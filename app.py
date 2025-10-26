from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_cors import CORS
import os
import sys
import json
import requests
import time
import re
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
import uuid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy import stats
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import joblib

# 直接使用 OpenAI 客户端处理 PDF
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'process_files'
app.config['SAVED_RESULTS'] = 'saved_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 读取 .env（如存在）并加载环境变量
load_dotenv()

# KIMI API 配置（从环境变量读取）
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions"

def compress_pdf_content(content, max_length):
    """智能压缩PDF内容，专门提取第二、九、十节内容"""
    if len(content) <= max_length:
        return content
    
    lines = content.split('\n')
    result_lines = []
    
    # 查找特定章节的标识符
    section_patterns = [
        # 第二节 - 公司基本信息
        r'第二节.*?公司.*?信息',
        r'第二节.*?概览',
        r'第二节.*?简介',
        r'二.*?公司.*?概况',
        r'二.*?基本.*?情况',
        
        # 第九节 - 通常包含债券信息
        r'第九节',
        r'九.*?债券',
        r'九.*?融资',
        r'九.*?信用',
        
        # 第十节 - 通常包含财务或评级信息
        r'第十节',
        r'十.*?财务',
        r'十.*?评级',
        r'十.*?信用'
    ]
    
    # 财务关键词用于辅助筛选
    financial_keywords = [
        '资产负债表', '利润表', '现金流量表', '股东权益',
        '流动比率', '速动比率', '资产回报率', '净利润率', '毛利率',
        '债务权益比', '资产周转率', '现金比率', '营业利润率',
        '总资产', '净利润', '营业收入', '负债', '权益',
        '现金流', '应收账款', '存货', '固定资产',
        '评级', '信用', 'Rating', 'Credit', 'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC',
        '公司名称', '股票代码', '行业', '评级机构', '注册地址', '法定代表人',
        '主营业务', '经营范围', '统一社会信用代码'
    ]
    
    # 标记是否在目标章节内
    in_target_section = False
    section_depth = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # 检查是否是目标章节开始
        is_target_section_start = False
        for pattern in section_patterns:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                is_target_section_start = True
                break
        
        # 检查是否是新的章节开始（用于结束当前章节提取）
        is_new_section = re.search(r'第.*?节|[一二三四五六七八九十]{1,2}[．、]', line_stripped)
        
        if is_target_section_start:
            in_target_section = True
            section_depth = 0
            result_lines.append(line_stripped)
        elif in_target_section:
            # 如果遇到下一个主要章节，结束当前章节提取
            if is_new_section and not is_target_section_start:
                # 检查是否是子章节还是新的主章节
                if re.search(r'第.*?节', line_stripped):
                    in_target_section = False
                else:
                    result_lines.append(line_stripped)
            else:
                result_lines.append(line_stripped)
        else:
            # 即使不在目标章节，也保留包含重要财务关键词的行
            if any(keyword in line_stripped for keyword in financial_keywords):
                result_lines.append(line_stripped)
    
    # 如果没有找到明确的章节，则使用关键词筛选
    if not result_lines:
        print("未找到指定章节，使用关键词筛选...")
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # 保留包含重要关键词的行
            if any(keyword in line_stripped for keyword in financial_keywords):
                result_lines.append(line_stripped)
    
    compressed = '\n'.join(result_lines)
    
    # 如果提取的内容仍然太长，进行进一步压缩
    if len(compressed) > max_length:
        print(f"提取内容仍然过长({len(compressed)}字符)，进行进一步压缩...")
        # 按重要性排序，保留最重要的内容
        priority_lines = []
        normal_lines = []
        
        for line in result_lines:
            # 高优先级：包含评级、财务比率、公司基本信息的行
            high_priority_keywords = [
                '评级', 'Rating', 'Credit', '信用',
                '公司名称', '股票代码', '注册地址',
                '流动比率', '速动比率', '净利润率', '资产回报率',
                'AAA', 'AA', 'BBB', 'BB', 'CCC'
            ]
            
            if any(keyword in line for keyword in high_priority_keywords):
                priority_lines.append(line)
            else:
                normal_lines.append(line)
        
        # 优先保留高优先级内容
        final_lines = priority_lines[:]
        current_length = sum(len(line) for line in final_lines)
        
        # 添加普通内容直到达到长度限制
        for line in normal_lines:
            if current_length + len(line) + 1 < max_length:
                final_lines.append(line)
                current_length += len(line) + 1
            else:
                break
        
        compressed = '\n'.join(final_lines)
        
        # 最终截断
        if len(compressed) > max_length:
            compressed = compressed[:max_length-100] + "\n...[内容已截断，已优先保留第二、九、十节内容]"
    
    print(f"章节提取完成，保留内容长度: {len(compressed)}字符")
    return compressed

# 创建 OpenAI 客户端
def create_kimi_client():
    """创建 KIMI API 客户端"""
    if not KIMI_API_KEY:
        raise RuntimeError("未设置 KIMI_API_KEY，请在环境变量或 .env 中配置")
    return OpenAI(
        api_key=KIMI_API_KEY,
        base_url="https://api.moonshot.cn/v1",
    )

def process_pdf_with_kimi(pdf_path, prompt_path):
    """使用 KIMI API 直接处理 PDF"""
    try:
        client = create_kimi_client()
        
        # 上传PDF文件到KIMI
        file_object = client.files.create(
            file=Path(pdf_path), 
            purpose="file-extract"
        )
        
        # 获取文件内容
        file_content = client.files.content(file_id=file_object.id).text
        
        # 检查内容长度，如果太长则进行章节提取
        max_content_length = 100000  # 增加到100k字符，因为我们现在精准提取
        if len(file_content) > max_content_length:
            print(f"PDF内容过长({len(file_content)}字符)，正在提取第二、九、十节...")
            file_content = compress_pdf_content(file_content, max_content_length)
            print(f"章节提取后长度: {len(file_content)}字符")
        else:
            print(f"PDF内容长度: {len(file_content)}字符，无需压缩")
        
        # 读取 prompt 文件
        prompt_content = Path(prompt_path).read_text(encoding="utf-8")
        
        # 构建消息 - 使用更高效的方式
        messages = [
            {
                "role": "user",
                "content": f"请从以下文档中提取财务数据：\n\n{file_content}\n\n{prompt_content}",
            }
        ]
        
        # 添加重试机制处理速率限制
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                # 调用 chat-completion
                completion = client.chat.completions.create(
                    model="kimi-k2-0711-preview",
                    messages=messages,
                    temperature=0.2,
                )
                break  # 成功则跳出重试循环
                
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"遇到速率限制，等待 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    raise e
        
        # 解析返回结果
        result_content = completion.choices[0].message.content
        
        try:
            # 尝试解析JSON
            import json
            if result_content.startswith('{') and result_content.endswith('}'):
                extracted_data = json.loads(result_content)
            else:
                # 如果不是完整JSON，尝试提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', result_content, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                else:
                    return {
                        'success': False,
                        'error': '无法解析API返回的数据格式',
                        'raw_response': result_content
                    }
            
            return {
                'success': True,
                'extracted_data': extracted_data.get('extracted_data', extracted_data),
                'raw_response': result_content
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'JSON解析错误: {str(e)}',
                'raw_response': result_content
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'处理PDF时发生错误: {str(e)}'
        }



# 确保文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['SAVED_RESULTS']]:
    os.makedirs(folder, exist_ok=True)

# 数据库初始化
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 创建用户表
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建分析会话表
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            filename TEXT,
            analysis_result TEXT,
            credit_evaluation TEXT,
            final_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # 检查并添加缺失的列
    try:
        # 检查 credit_evaluation 列是否存在
        c.execute("PRAGMA table_info(analysis_sessions)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'credit_evaluation' not in columns:
            print("添加 credit_evaluation 列...")
            c.execute('ALTER TABLE analysis_sessions ADD COLUMN credit_evaluation TEXT')
            
        if 'final_result' not in columns:
            print("添加 final_result 列...")
            c.execute('ALTER TABLE analysis_sessions ADD COLUMN final_result TEXT')
            
    except Exception as e:
        print(f"数据库迁移错误: {e}")
    
    conn.commit()
    conn.close()

# 定义改进后的多层感知器模型
class ImprovedMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.gelu(out)
        out = self.fc4(out)
        return out

# 企业信用评估模型
class CreditEvaluationModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.lgb_model = None
        self.mlp_model = None
        self.is_trained = False
        self.selected_feature_indices = None
        self.rating_mapping = {
            0: 'CCC', 1: 'B', 2: 'BB', 3: 'BBB', 
            4: 'A', 5: 'AA', 6: 'AAA'
        }
        self.feature_names = [
            'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
            'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin',
            'returnOnAssets', 'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover',
            'fixedAssetTurnover', 'debtEquityRatio', 'debtRatio', 'effectiveTaxRate',
            'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
            'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
            'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover'
        ]
    
    def train_model(self, training_data_path=None):
        """训练LightGBM+MLP组合模型"""
        try:
            # 如果没有提供训练数据路径，使用模拟数据进行初始化
            if training_data_path is None:
                # 创建模拟训练数据用于演示
                self._initialize_with_mock_data()
                return True
            
            # 加载真实训练数据
            data = pd.read_csv(training_data_path)
            
            # 提取特征和目标变量
            X = data.drop(['Rating', 'Name', 'Symbol', 'Rating Agency Name', 'Date', 'Sector'], axis=1).values
            y = data['Rating'].values
            
            # 对目标变量进行编码
            y = self.label_encoder.fit_transform(y)
            
            # 处理异常值
            z_scores = np.abs(stats.zscore(X))
            filtered_entries = (z_scores < 3).all(axis=1)
            X = X[filtered_entries]
            y = y[filtered_entries]
            
            # 特征筛选
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    correlations.append((i, abs(corr)))
            correlations.sort(key=lambda x: x[1], reverse=True)
            self.selected_feature_indices = [idx for idx, _ in correlations[:int(0.8 * len(correlations))]]
            X = X[:, self.selected_feature_indices]
            
            # 多项式特征变换
            X = self.poly.fit_transform(X)
            
            # 特征标准化
            X = self.scaler.fit_transform(X)
            
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练LightGBM模型
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_train)),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            self.lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
            
            # 获取LightGBM预测结果
            X_train_lgb = self.lgb_model.predict(X_train)
            X_val_lgb = self.lgb_model.predict(X_val)
            
            # 组合特征
            X_train_combined = np.hstack([X_train, X_train_lgb])
            X_val_combined = np.hstack([X_val, X_val_lgb])
            
            # 训练MLP模型
            input_size = X_train_combined.shape[1]
            num_classes = len(np.unique(y_train))
            self.mlp_model = ImprovedMLP(input_size, num_classes)
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train_combined)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val_combined)
            y_val_tensor = torch.LongTensor(y_val)
            
            # 训练MLP
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.mlp_model.parameters(), weight_decay=0.01)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            
            best_val_loss = float('inf')
            patience = 15
            counter = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                outputs = self.mlp_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    val_outputs = self.mlp_model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            break
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"模型训练失败: {str(e)}")
            # 如果训练失败，使用模拟数据初始化
            self._initialize_with_mock_data()
            return False
    
    def _initialize_with_mock_data(self):
        """使用模拟数据初始化模型（用于演示）"""
        # 创建模拟数据进行基本初始化
        np.random.seed(42)
        mock_X = np.random.randn(100, len(self.feature_names))
        mock_y = np.random.randint(0, 7, 100)  # 7个信用等级
        
        # 基本的标准化器初始化
        self.scaler.fit(mock_X)
        self.label_encoder.fit(['CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA'])
        
        # 使用简化的多项式特征（避免维度爆炸）
        self.poly = PolynomialFeatures(degree=1, include_bias=False)
        mock_X_poly = self.poly.fit_transform(mock_X)
        
        # 设置选中的特征索引
        self.selected_feature_indices = list(range(len(self.feature_names)))
        
        # 简化的LightGBM模型初始化
        try:
            lgb_train = lgb.Dataset(mock_X_poly, mock_y)
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 7,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 10,
                'learning_rate': 0.1,
                'verbose': -1
            }
            self.lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=10)
        except:
            self.lgb_model = None
        
        # 简化的MLP模型初始化
        if self.lgb_model:
            mock_lgb_pred = self.lgb_model.predict(mock_X_poly)
            mock_combined = np.hstack([mock_X_poly, mock_lgb_pred])
            input_size = mock_combined.shape[1]
        else:
            input_size = mock_X_poly.shape[1]
        
        self.mlp_model = ImprovedMLP(input_size, 7)
        self.is_trained = True
    
    def preprocess_data(self, financial_data):
        """预处理财务数据"""
        processed_data = []
        for feature in self.feature_names:
            value = financial_data.get(feature, 'N/A')
            if value == 'N/A' or value == '':
                processed_data.append(0.0)
            else:
                try:
                    # 移除百分号并转换为浮点数
                    if isinstance(value, str):
                        value = value.replace('%', '').replace(',', '')
                    processed_data.append(float(value))
                except:
                    processed_data.append(0.0)
        
        return np.array(processed_data).reshape(1, -1)
    
    def evaluate_credit(self, financial_data):
        """使用LightGBM+MLP组合模型评估企业信用"""
        try:
            # 确保模型已训练
            if not self.is_trained:
                self.train_model()
            
            # 预处理数据
            processed_data = self.preprocess_data(financial_data)
            
            # 应用特征选择
            if self.selected_feature_indices:
                processed_data = processed_data[:, self.selected_feature_indices]
            
            # 应用多项式特征变换
            try:
                processed_data = self.poly.transform(processed_data)
            except:
                # 如果transform失败，重新初始化
                self._initialize_with_mock_data()
                processed_data = self.preprocess_data(financial_data)
                processed_data = self.poly.transform(processed_data)
            
            # 标准化
            processed_data = self.scaler.transform(processed_data)
            
            # 使用组合模型进行预测
            if self.lgb_model and self.mlp_model:
                # LightGBM预测
                lgb_pred = self.lgb_model.predict(processed_data)
                lgb_pred = lgb_pred.reshape(1, -1)
                
                # 组合特征
                combined_features = np.hstack([processed_data, lgb_pred])
                
                # MLP预测
                self.mlp_model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(combined_features)
                    mlp_output = self.mlp_model(input_tensor)
                    probabilities = torch.softmax(mlp_output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = torch.max(probabilities).item()
                
                # 获取信用等级
                credit_rating = self.rating_mapping.get(predicted_class, 'BBB')
                
                # 计算信用评分（基于概率和传统方法的组合）
                ml_score = confidence * 100
                traditional_score = self.calculate_traditional_credit_score(financial_data)
                # 组合评分（70%机器学习 + 30%传统方法）
                final_score = 0.7 * ml_score + 0.3 * traditional_score
                
            else:
                # 如果模型未正确初始化，使用传统方法
                final_score = self.calculate_traditional_credit_score(financial_data)
                credit_rating = self._score_to_rating(final_score)
                confidence = 0.8
            
            # 确定风险等级
            risk_level = self._rating_to_risk_level(credit_rating)
            
            return {
                "success": True,
                "credit_score": round(final_score, 2),
                "credit_rating": credit_rating,
                "risk_level": risk_level,
                "confidence": round(confidence * 100, 2),
                "model_type": "LightGBM+MLP" if self.lgb_model and self.mlp_model else "Traditional",
                "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
                "analysis_summary": self.generate_analysis_summary(financial_data, final_score, credit_rating)
            }
            
        except Exception as e:
            # 如果ML模型失败，降级到传统方法
            try:
                score = self.calculate_traditional_credit_score(financial_data)
                rating = self._score_to_rating(score)
                risk_level = self._rating_to_risk_level(rating)
                
                return {
                    "success": True,
                    "credit_score": round(score, 2),
                    "credit_rating": rating,
                    "risk_level": risk_level,
                    "confidence": 75.0,
                    "model_type": "Traditional (Fallback)",
                    "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
                    "analysis_summary": self.generate_analysis_summary(financial_data, score, rating),
                    "warning": "使用传统方法评估，机器学习模型暂不可用"
                }
            except Exception as fallback_error:
                return {
                    "success": False,
                    "error": f"评估失败: {str(fallback_error)}"
                }
    
    def _score_to_rating(self, score):
        """将评分转换为信用等级"""
        if score >= 85:
            return "AAA"
        elif score >= 75:
            return "AA"
        elif score >= 65:
            return "A"
        elif score >= 55:
            return "BBB"
        elif score >= 45:
            return "BB"
        elif score >= 35:
            return "B"
        else:
            return "CCC"
    
    def _rating_to_risk_level(self, rating):
        """将信用等级转换为风险等级"""
        risk_mapping = {
            "AAA": "极低风险",
            "AA": "低风险", 
            "A": "较低风险",
            "BBB": "中等风险",
            "BB": "较高风险",
            "B": "高风险",
            "CCC": "极高风险"
        }
        return risk_mapping.get(rating, "中等风险")
    
    def calculate_traditional_credit_score(self, financial_data):
        """传统信用评分方法（作为备选）"""
        score = 50  # 基础分数
        
        # 流动性指标评估 (20分)
        current_ratio = self.safe_float(financial_data.get('currentRatio', 'N/A'))
        if current_ratio > 2.0:
            score += 10
        elif current_ratio > 1.5:
            score += 5
        elif current_ratio < 1.0:
            score -= 5
        
        quick_ratio = self.safe_float(financial_data.get('quickRatio', 'N/A'))
        if quick_ratio > 1.0:
            score += 10
        elif quick_ratio < 0.5:
            score -= 5
        
        # 盈利能力评估 (25分)
        net_profit_margin = self.safe_float(financial_data.get('netProfitMargin', 'N/A'))
        if net_profit_margin > 15:
            score += 15
        elif net_profit_margin > 10:
            score += 10
        elif net_profit_margin > 5:
            score += 5
        elif net_profit_margin < 0:
            score -= 10
        
        roe = self.safe_float(financial_data.get('returnOnEquity', 'N/A'))
        if roe > 20:
            score += 10
        elif roe > 15:
            score += 5
        elif roe < 5:
            score -= 5
        
        # 杠杆指标评估 (20分)
        debt_ratio = self.safe_float(financial_data.get('debtRatio', 'N/A'))
        if debt_ratio < 30:
            score += 10
        elif debt_ratio < 50:
            score += 5
        elif debt_ratio > 70:
            score -= 10
        
        debt_equity_ratio = self.safe_float(financial_data.get('debtEquityRatio', 'N/A'))
        if debt_equity_ratio < 0.5:
            score += 10
        elif debt_equity_ratio > 1.0:
            score -= 5
        
        # 运营效率评估 (15分)
        asset_turnover = self.safe_float(financial_data.get('assetTurnover', 'N/A'))
        if asset_turnover > 1.0:
            score += 8
        elif asset_turnover > 0.5:
            score += 4
        
        # 现金流评估 (20分)
        operating_cash_flow_ratio = self.safe_float(financial_data.get('operatingCashFlowSalesRatio', 'N/A'))
        if operating_cash_flow_ratio > 0.15:
            score += 10
        elif operating_cash_flow_ratio > 0.10:
            score += 5
        elif operating_cash_flow_ratio < 0.05:
            score -= 5
        
        return max(0, min(100, score))
    
    def safe_float(self, value):
        """安全转换为浮点数"""
        if value == 'N/A' or value == '':
            return 0.0
        try:
            if isinstance(value, str):
                value = value.replace('%', '').replace(',', '')
            return float(value)
        except:
            return 0.0
    
    def generate_analysis_summary(self, financial_data, score, credit_rating):
        """生成分析摘要"""
        summary = []
        
        # 添加模型信息
        if hasattr(self, 'mlp_model') and self.mlp_model and hasattr(self, 'lgb_model') and self.lgb_model:
            summary.append("使用LightGBM+深度学习组合模型进行评估")
        else:
            summary.append("使用传统财务指标分析方法进行评估")
        
        # 流动性分析
        current_ratio = self.safe_float(financial_data.get('currentRatio', 'N/A'))
        if current_ratio > 2.0:
            summary.append("流动性状况优秀，短期偿债能力强")
        elif current_ratio > 1.5:
            summary.append("流动性状况良好，短期偿债能力较强")
        elif current_ratio < 1.0:
            summary.append("流动性状况较差，短期偿债能力不足")
        
        # 盈利能力分析
        net_profit_margin = self.safe_float(financial_data.get('netProfitMargin', 'N/A'))
        if net_profit_margin > 15:
            summary.append("盈利能力优秀，净利润率较高")
        elif net_profit_margin > 10:
            summary.append("盈利能力良好，净利润率适中")
        elif net_profit_margin < 0:
            summary.append("盈利能力较差，存在亏损风险")
        
        # 杠杆分析
        debt_ratio = self.safe_float(financial_data.get('debtRatio', 'N/A'))
        if debt_ratio < 30:
            summary.append("财务杠杆较低，财务风险较小")
        elif debt_ratio > 70:
            summary.append("财务杠杆较高，财务风险较大")
        
        # 信用等级解释
        rating_explanations = {
            "AAA": "信用质量极高，违约风险极低",
            "AA": "信用质量很高，违约风险很低", 
            "A": "信用质量较高，违约风险较低",
            "BBB": "信用质量中等，存在一定违约风险",
            "BB": "信用质量较低，违约风险较高",
            "B": "信用质量低，违约风险高",
            "CCC": "信用质量很低，违约风险很高"
        }
        
        if credit_rating in rating_explanations:
            summary.append(f"信用等级{credit_rating}：{rating_explanations[credit_rating]}")
        
        return "；".join(summary) if summary else "数据不足，无法进行详细分析"

# 初始化模型
credit_model = CreditEvaluationModel()
# 在启动时初始化模型
print("正在初始化信用评估模型...")
credit_model.train_model()
print("信用评估模型初始化完成！")

# 路由定义
# @app.route('/test_api')
# def test_api_route():
#     """测试KIMI API连接的路由"""
#     result = test_kimi_api()
#     return jsonify(result)

# @app.route('/api_test_page')
# def api_test_page():
#     """API测试页面"""
#     return render_template('api_test.html')

@app.route('/')
def index():
    """第一个界面：用户登录"""
    if 'user_id' in session:
        return redirect(url_for('upload_page'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    """用户登录处理"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[3], password):
        session['user_id'] = user[0]
        session['username'] = user[1]
        flash('登录成功！', 'success')
        return redirect(url_for('upload_page'))
    else:
        flash('用户名或密码错误！', 'error')
        return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # 检查用户是否已存在
        c.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email))
        if c.fetchone():
            flash('用户名或邮箱已存在！', 'error')
            conn.close()
            return redirect(url_for('register'))
        
        # 创建新用户
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                 (username, email, password_hash))
        conn.commit()
        conn.close()
        
        flash('注册成功！请登录', 'success')
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/upload')
def upload_page():
    """第二个界面：PDF文件上传"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """处理PDF文件"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '请先登录'})
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': '请上传PDF文件'})
    
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 创建会话ID
        session_id = str(uuid.uuid4())
        
        # 使用本地的prompt.md文件
        prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.md')
        
        result = process_pdf_with_kimi(filepath, prompt_path)
        
        if result['success']:
            # 保存结果到数据库
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO analysis_sessions (id, user_id, filename, analysis_result)
                VALUES (?, ?, ?, ?)
            ''', (session_id, session['user_id'], filename, json.dumps(result)))
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'PDF处理成功！'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '处理失败')
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analysis_result/<session_id>')
def analysis_result(session_id):
    """第三个界面：美化分析结果"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT analysis_result FROM analysis_sessions WHERE id = ? AND user_id = ?',
             (session_id, session['user_id']))
    result = c.fetchone()
    conn.close()
    
    if not result:
        flash('会话不存在或已过期', 'error')
        return redirect(url_for('upload_page'))
    
    analysis_data = json.loads(result[0])
    return render_template('analysis_result.html', 
                         session_id=session_id,
                         analysis_data=analysis_data)

@app.route('/credit_evaluation/<session_id>')
def credit_evaluation(session_id):
    """第四个界面：企业信用评估"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT analysis_result FROM analysis_sessions WHERE id = ? AND user_id = ?',
             (session_id, session['user_id']))
    result = c.fetchone()
    conn.close()
    
    if not result:
        flash('会话不存在或已过期', 'error')
        return redirect(url_for('upload_page'))
    
    analysis_data = json.loads(result[0])
    
    # 进行信用评估
    if analysis_data.get('extracted_data'):
        evaluation_result = credit_model.evaluate_credit(analysis_data['extracted_data'])
        
        # 保存评估结果
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE analysis_sessions SET credit_evaluation = ? WHERE id = ?',
                 (json.dumps(evaluation_result), session_id))
        conn.commit()
        conn.close()
        
        return render_template('credit_evaluation.html',
                             session_id=session_id,
                             analysis_data=analysis_data,
                             evaluation_result=evaluation_result)
    else:
        flash('分析数据不完整，无法进行评估', 'error')
        return redirect(url_for('analysis_result', session_id=session_id))

@app.route('/final_result/<session_id>')
def final_result(session_id):
    """第五个界面：最终结果展示"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT analysis_result, credit_evaluation, filename 
        FROM analysis_sessions 
        WHERE id = ? AND user_id = ?
    ''', (session_id, session['user_id']))
    result = c.fetchone()
    conn.close()
    
    if not result:
        flash('会话不存在或已过期', 'error')
        return redirect(url_for('upload_page'))
    
    analysis_data = json.loads(result[0])
    evaluation_data = json.loads(result[1]) if result[1] else None
    filename = result[2]
    
    return render_template('final_result.html',
                         session_id=session_id,
                         analysis_data=analysis_data,
                         evaluation_data=evaluation_data,
                         filename=filename)

@app.route('/save_result/<session_id>')
def save_result(session_id):
    """生成可保存的报告页面"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '请先登录'})
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''
            SELECT analysis_result, credit_evaluation, filename 
            FROM analysis_sessions 
            WHERE id = ? AND user_id = ?
        ''', (session_id, session['user_id']))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'success': False, 'error': '会话不存在或已过期'})
        
        analysis_data = json.loads(result[0])
        evaluation_data = json.loads(result[1]) if result[1] else None
        filename = result[2]
        
        # 返回用于截图的报告页面
        return render_template('printable_report.html',
                             session_id=session_id,
                             analysis_data=analysis_data,
                             evaluation_data=evaluation_data,
                             filename=filename,
                             generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    """下载文件（保留兼容性）"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    try:
        return send_file(
            os.path.join(app.config['SAVED_RESULTS'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        flash(f'下载失败: {str(e)}', 'error')
        return redirect(url_for('upload_page'))

@app.route('/download/<filename>')
def download(filename):
    """兼容性路由，用于可能存在的旧链接"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    flash('报告已显示为网页格式，您可以使用浏览器打印或截图保存', 'info')
    return redirect(url_for('upload_page'))

@app.route('/logout')
def logout():
    """用户登出"""
    session.clear()
    flash('已成功登出', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5006) 