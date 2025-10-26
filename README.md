# 企业信用管理系统 - LightGBM+MLP智能评估版

一个基于AI和机器学习的企业财务信用评估Web应用，支持PDF文档上传、智能分析和高精度信用评级。

## 功能特点

### 🎯 五个核心界面
1. **用户登录界面** - 安全登录和注册
2. **PDF上传界面** - 支持拖拽上传PDF文件
3. **数据分析界面** - 美化展示KIMI API分析结果
4. **信用评估界面** - 基于LightGBM+MLP模型的智能评级
5. **最终报告界面** - 完整报告展示和图片保存

### 🔧 技术特性
- **智能PDF处理**: 自动将PDF转换为文本，支持大文件分节提取
- **AI分析**: 集成KIMI API进行智能财务数据提取
- **ML信用评估**: LightGBM + 多层感知器（MLP）组合模型
- **图片报告**: 高质量JPG格式报告生成和下载
- **响应式设计**: 现代化的Web界面

## 模型架构升级

### LightGBM + MLP 组合模型
- **LightGBM**: 用于特征工程和初步预测
- **多层感知器**: 用于深度学习和最终分类
- **模型降级机制**: 确保系统稳定性，ML模型失败时自动使用传统方法
- **特征工程**: 异常值处理、特征选择、多项式变换、标准化
- **性能优化**: 早停机制、学习率调度、批量归一化、正则化

### 评估结果增强
- **置信度**: 提供预测的可信度指标
- **模型类型**: 显示使用的具体模型（ML/传统）
- **智能分析**: 基于25个核心财务指标的综合评估
- **风险等级**: 7级评级系统（CCC到AAA）

## 系统架构

```
企业信用管理系统/
├── app.py                  # 主应用文件（LightGBM+MLP集成）
├── templates/              # HTML模板
│   ├── login.html          # 登录界面
│   ├── register.html       # 注册界面
│   ├── upload.html         # 上传界面
│   ├── analysis_result.html # 分析结果界面
│   ├── credit_evaluation.html # 信用评估界面
│   ├── final_result.html   # 最终报告界面
│   └── printable_report.html # 可打印报告界面
├── static/                 # 静态资源
├── uploads/                # 上传文件存储
├── saved_results/          # 生成的图片报告
├── users.db               # 用户数据库
├── 启动系统.bat            # Windows启动脚本
└── README.md              # 使用说明
```

## 安装和运行

### 1. 环境要求
- Python 3.8+
- Windows/Linux/macOS

### 2. 安装依赖
使用虚拟环境并安装依赖（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 启动系统

#### Windows用户

优先方式：PowerShell 启动

```powershell
# 首次使用，复制并填写 KIMI Key
Copy-Item .env.example .env
# 编辑 .env，填入你的 KIMI_API_KEY

# 进入虚拟环境并运行
.\.venv\Scripts\Activate.ps1
python app.py
```

或双击运行 `启动系统.bat`（会尝试直接用系统 Python 启动）。

#### 手动启动
```powershell
python app.py
```

### 4. 访问系统
打开浏览器访问: <http://localhost:5006>

## 使用流程

### 第一步：用户注册/登录

- 首次使用需要注册账户
- 输入用户名、邮箱和密码
- 登录后进入主界面

### 第二步：上传PDF文件

- 支持拖拽上传或点击选择
- 仅支持PDF格式文件
- 系统自动处理文件

### 第三步：查看分析结果

- 展示从PDF提取的财务数据
- 包括公司基本信息、财务指标等
- 数据按类别分组展示

### 第四步：信用评估
- 基于财务数据计算信用评分
- 显示信用等级和风险等级
- 提供详细的分析摘要

### 第五步：生成最终报告

- 完整的评估报告展示
- 可选择保存为高质量JPG图片
- 支持下载和分享

## 核心功能详解

### CreditEvaluationModel类

```python
class CreditEvaluationModel:
    def train_model()                    # 训练LightGBM+MLP组合模型
    def evaluate_credit(financial_data)  # 使用ML模型进行信用评估
    def calculate_traditional_credit_score()  # 传统评估方法（备用）
```

### 信用评估模型

- **评分范围**: 0-100分
- **评级等级**: CCC, B, BB, BBB, A, AA, AAA (7级评级)
- **评估维度**:
  - 流动性指标 (20分)
  - 盈利能力 (25分)
  - 杠杆指标 (20分)
  - 运营效率 (15分)
  - 现金流 (20分)

### 智能PDF处理

- 支持大文件智能分节提取（第二、九、十节）
- 自动处理KIMI API令牌限制
- 高效的内容压缩和筛选算法

### KIMI API集成

- 基于专业提示词模板
- 自动提取25+个核心财务指标
- 支持多种数据格式和异常处理

## 数据库结构

### 用户表 (users)

- id: 主键
- username: 用户名
- email: 邮箱
- password_hash: 密码哈希
- created_at: 创建时间

### 分析会话表 (analysis_sessions)

- id: 会话ID
- user_id: 用户ID
- filename: 文件名
- analysis_result: 分析结果
- credit_evaluation: 信用评估结果
- final_result: 最终结果（可选）
- created_at: 创建时间

## 配置说明

### KIMI API配置

在 `app.py` 中修改：

```python
# 推荐使用环境变量或 .env 文件
# 在项目根目录（web_kimiai）创建 .env 写入：
# KIMI_API_KEY=sk-xxxx
```

### 系统配置

```python
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SAVED_RESULTS'] = 'saved_results'
```

## Python依赖

### 核心依赖包

- **torch** - PyTorch深度学习框架
- **lightgbm** - 梯度提升框架
- **scikit-learn** - 机器学习工具
- **pandas, numpy, scipy** - 数据处理
- **flask, flask-cors** - Web框架
- **openai** - KIMI API客户端
- **PyPDF2** - PDF处理

### 安装命令（可选直装）

```powershell
pip install torch lightgbm scikit-learn pandas numpy scipy flask flask-cors openai PyPDF2 python-dotenv
```

或使用 `requirements.txt`：

```powershell
pip install -r requirements.txt
```

注意：PyTorch 安装可能根据你的环境自动选择 CPU/CUDA 版本，若失败，请参考 <https://pytorch.org> 获取对应命令。

## 常见问题

- 启动报错“未设置 KIMI_API_KEY”：
  - 复制 `.env.example` 为 `.env` 并填入 KIMI_API_KEY，或在系统环境变量中设置。
- 端口占用：
  - 本项目监听 5006 端口，若被占用，可在 `app.py` 修改 `port=5006` 为其他端口。
- OpenAI/KIMI 依赖错误：
  - 已固定 `openai` 版本到 1.x，请确保卸载旧的 0.x：`pip uninstall -y openai` 后重装。

## 系统特性

### 智能特性

- **自适应处理**: 根据PDF大小自动选择处理策略
- **模型降级**: ML模型失败时自动使用传统方法
- **容错设计**: 处理各种数据格式和异常情况
- **高精度评估**: 基于25个核心财务指标

### 性能优势

- **准确性提升**: ML模型能够发现传统方法难以捕捉的复杂模式
- **自适应性**: 模型可以根据新数据进行调整
- **鲁棒性**: 多重备用机制确保系统始终可用
- **可解释性**: 保留传统分析方法作为对比


