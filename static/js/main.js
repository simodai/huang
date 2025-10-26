// 企业信用评估系统 - 自定义JavaScript

// 拖拽上传功能
function initDragDropUpload() {
    const dropArea = document.querySelector('.drag-drop-area');
    const fileInput = document.querySelector('#file');
    
    if (dropArea && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        dropArea.addEventListener('drop', handleDrop, false);
    }
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.querySelector('.drag-drop-area').classList.add('dragover');
}

function unhighlight(e) {
    document.querySelector('.drag-drop-area').classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const fileInput = document.querySelector('#file');
        fileInput.files = files;
        updateFileName(files[0].name);
    }
}

function updateFileName(fileName) {
    const fileNameDisplay = document.querySelector('.file-name-display');
    if (fileNameDisplay) {
        fileNameDisplay.textContent = `已选择文件: ${fileName}`;
        fileNameDisplay.style.display = 'block';
    }
}

// 进度条动画
function animateProgressBar(element, targetValue, duration = 2000) {
    const startValue = 0;
    const startTime = performance.now();
    
    function updateProgress(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentValue = startValue + (targetValue - startValue) * progress;
        
        element.style.width = currentValue + '%';
        element.setAttribute('aria-valuenow', currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(updateProgress);
        }
    }
    
    requestAnimationFrame(updateProgress);
}

// 数字计数动画
function animateNumber(element, targetValue, duration = 2000) {
    const startValue = 0;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentValue = startValue + (targetValue - startValue) * progress;
        
        element.textContent = Math.round(currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// 表单验证
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;
    
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.classList.add('is-invalid');
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// 显示加载动画
function showLoading(message = '处理中...') {
    const loadingHtml = `
        <div class="loading-overlay" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        ">
            <div class="loading-content text-center text-white">
                <div class="spinner-border mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p>${message}</p>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', loadingHtml);
}

function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// 初始化所有功能
document.addEventListener('DOMContentLoaded', function() {
    // 初始化拖拽上传
    initDragDropUpload();
    
    // 文件输入变化监听
    const fileInput = document.querySelector('#file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                updateFileName(e.target.files[0].name);
            }
        });
    }
    
    // 进度条动画
    const progressBars = document.querySelectorAll('.progress-bar[data-value]');
    progressBars.forEach(bar => {
        const targetValue = parseFloat(bar.getAttribute('data-value'));
        setTimeout(() => {
            animateProgressBar(bar, targetValue);
        }, 500);
    });
    
    // 数字动画
    const animatedNumbers = document.querySelectorAll('.animated-number[data-value]');
    animatedNumbers.forEach(element => {
        const targetValue = parseFloat(element.getAttribute('data-value'));
        setTimeout(() => {
            animateNumber(element, targetValue);
        }, 500);
    });
    
    // 淡入动画
    const fadeElements = document.querySelectorAll('.fade-in-up');
    fadeElements.forEach((element, index) => {
        setTimeout(() => {
            element.style.opacity = '1';
        }, index * 100);
    });
});

// 工具函数
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatPercent(num) {
    return (num * 100).toFixed(2) + '%';
}

function formatCurrency(num) {
    return '¥' + formatNumber(num.toFixed(2));
}
