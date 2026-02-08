# Quick Start Guide

## 方法 1：使用启动脚本（推荐）

### Mac/Linux:
```bash
cd /Users/andyzhang/Desktop/corp
./launch_website.sh
```

### Windows:
```bash
cd C:\Users\...\Desktop\corp
pip install -r requirements.txt
streamlit run mvr_website.py
```

## 方法 2：手动启动

### 1. 安装依赖
```bash
cd /Users/andyzhang/Desktop/corp
pip3 install streamlit pandas numpy networkx matplotlib scikit-learn
```

### 2. 启动网站
```bash
streamlit run mvr_website.py
```

网站将自动在浏览器中打开：**http://localhost:8501**

---

## 网站使用说明

### 左侧边栏（参数设置）
1. **R (Repetitions)**: 重复次数，默认3000
   - 建议：3000（完整分析）或 500（快速测试）
   
2. **T (Iterations)**: 每次重复的迭代次数，默认1500
   - 建议：1500（完整分析）或 500（快速测试）
   
3. **Enable Early Stopping**: 早停机制
   - 勾选：如果500轮没有新的optimal rankings则停止
   - 不勾选：运行完整的R轮
   
4. **K-means Method**: 选择层级分类方法
   - **Bonhomme et al. (2019)**: 论文方法，自动选择K
   - **Overall Std**: 基于总体方差的阈值
   - **Elbow (Manual)**: 先看图，手动选K（推荐K=4）
   - **Simple Variance**: 简单方差阈值

5. **Choose K**: 如果选择了Elbow方法，这里手动输入K值

### 主页面（结果展示）

点击 **Run Analysis** 后会显示：

#### Step 1: MVR - Finding Optimal Rankings
- 进度条显示运行状态
- Convergence图：显示发现optimal rankings的过程
- Coverage指标：找到的rankings占理论最大值的比例

#### Step 2: Job Position Variance
- 左图：每个职位的平均排名（带误差棒）
- 右图：每个职位的位置方差（红色=高方差，绿色=低方差）

#### Step 3: K-means Clustering
- 显示选定方法的Q(K)或Inertia曲线
- 红色虚线：阈值
- 绿色虚线：选定的K值

#### Step 4: Job Cluster Visualization
- 散点图：显示每个职位分配到哪一层
- 可展开查看详细的层级分配

---

## 常见参数组合

### 快速测试（10秒）
```
R = 500
T = 500
Early Stop = True
K-means = Bonhomme et al. (2019)
```

### 标准分析（30-60秒）
```
R = 3000
T = 1500
Early Stop = False
K-means = Elbow (Manual), K = 4
```

### 深度分析（2-3分钟）
```
R = 5000
T = 2000
Early Stop = False
试用所有4种K-means方法并比较
```

---

## 部署到云端（可选）

### Streamlit Cloud（免费）
1. 将代码推送到GitHub
2. 访问 https://share.streamlit.io
3. 连接GitHub仓库
4. 选择 `mvr_website.py` 作为主文件
5. 点击 Deploy

你会得到一个公开的URL，可以分享给审稿人！

---

## 故障排除

**问题1：pip install失败**
```bash
# 尝试升级pip
pip3 install --upgrade pip
# 然后重新安装
pip3 install -r requirements.txt
```

**问题2：网站启动但打不开**
- 检查终端输出的URL
- 尝试手动打开：http://localhost:8501
- 或尝试：http://127.0.0.1:8501

**问题3：运行太慢**
- 降低R和T的值（例如R=1000, T=1000）
- 启用Early Stopping
- 使用更快的机器

**问题4：Plot不显示**
```bash
pip3 install matplotlib --upgrade
```

---

## 文件说明

- `mvr_website.py` - 主程序文件（Streamlit应用）
- `mvr_website_prototype.ipynb` - Jupyter notebook原型
- `requirements.txt` - Python依赖包列表
- `launch_website.sh` - 自动启动脚本
- `README.md` - 项目说明文档
- `QUICKSTART.md` - 本文件（快速启动指南）

---

## 下一步建议

1. **本地测试**：先在本地运行，确保一切正常
2. **调整参数**：尝试不同的R、T和K-means方法
3. **截图保存**：将结果图表截图，用于论文
4. **云端部署**：部署到Streamlit Cloud，获得公开URL
5. **论文引用**：在论文中引用这个交互式工具的URL

---

**祝使用愉快！如有问题，请检查终端输出的错误信息。**
