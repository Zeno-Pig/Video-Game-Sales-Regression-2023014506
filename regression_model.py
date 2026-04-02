import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文显示（可选，防止绘图乱码）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
file_path = r"C:\Users\Administrator\Desktop\回归分析\vgsales.csv"
print(f"正在读取文件: {file_path}")
df = pd.read_csv(file_path)

# 2. 数据清洗与预处理
print("正在进行数据清洗...")

# 剔除年份或销量缺失的数据
df = df.dropna(subset=['Year', 'Global_Sales', 'Genre', 'Platform'])
df = df[df['Year'] != 'N/A']

# 转换年份为数值型
df['Year'] = pd.to_numeric(df['Year']).astype(int)

# 数据分布通常呈现严重长尾（极少数游戏销量极高），回归前进行对数变换可使模型更稳健
# 我们创建一个新列：对数销量 (log1p 处理销量为0的情况)
df['Log_Global_Sales'] = np.log1p(df['Global_Sales'])

# 3. 特征工程 (One-Hot Encoding)
# 选择特征变量
X_raw = df[['Year', 'Genre', 'Platform']]

# 对类别变量进行独热编码，drop_first=True 用于避免多重共线性陷阱（虚拟变量陷阱）
X_encoded = pd.get_dummies(X_raw, columns=['Genre', 'Platform'], drop_first=True)

# 目标变量
y = df['Log_Global_Sales'] # 使用对数销量效果更好，若想回归原始销量，可改为 df['Global_Sales']

# 4. 划分数据集 (训练集 80%, 测试集 20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. 模型求解 (使用 statsmodels 获取详细统计信息)
# statsmodels 需要手动添加常数项 (截距)
X_train_with_constant = sm.add_constant(X_train)
X_test_with_constant = sm.add_constant(X_test)

# 转换为浮点型确保计算兼容性
model = sm.OLS(y_train.astype(float), X_train_with_constant.astype(float)).fit()

# 6. 输出回归分析报告
print("\n" + "="*60)
print("回归分析总结报告")
print("="*60)
print(model.summary())

# 7. 模型预测与评估
y_pred = model.predict(X_test_with_constant)

# 将对数预测值还原回原始销量单位进行误差计算
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred.astype(float))

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print("\n" + "-"*30)
print(f"模型 R-squared (对数空间): {r2:.4f}")
print(f"均方根误差 (原始空间 - 百万份): {rmse:.4f}")
print("-"*30)

# 8. 结果可视化
plt.figure(figsize=(12, 6))

# 图1：真实值 vs 预测值
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实销量 (对数)')
plt.ylabel('预测销量 (对数)')
plt.title('对数空间：真实值 vs 预测值')

# 图2：残差分布分析
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='green')
plt.xlabel('残差 (误差)')
plt.title('模型残差分布 (Residuals)')

plt.tight_layout()
plt.show()

# 9. 提取显著性变量
print("\n--- 显著性影响因素分析 (P < 0.05) ---")
significant_params = model.pvalues[model.pvalues < 0.05].sort_values()
for var, p in significant_params.items():
    coef = model.params[var]
    impact = "正向影响" if coef > 0 else "负向影响"
    print(f"变量: {var:<30} | P值: {p:.4e} | 影响方向: {impact}")