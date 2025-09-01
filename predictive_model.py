import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def remove_outliers(data, threshold=4):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev
    outlier_rows = np.any((data < lower_bound) | (data > upper_bound), axis=1)
    cleaned_data = data[~outlier_rows]
    return cleaned_data

def evaluate_performance(y_true, y_pred, dataset_type):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    within_threshold = np.mean(np.abs(y_true - y_pred) <= 0.2)
    print(f"{dataset_type} 集:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"在误差范围内的预测比例: {within_threshold:.4f}\n")
    return mse, rmse, r2, within_threshold

def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Standard Line')
    #plt.plot([y_true.min(), y_true.max()], [y_true.min() + 0.2, y_true.max() + 0.2], 'r--', lw=2, label='+0.2 Error Line')
    #plt.plot([y_true.min(), y_true.max()], [y_true.min() - 0.2, y_true.max() - 0.2], 'r--', lw=2, label='-0.2 Error Line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

# 加载增强后的数据集
enhanced_data = pd.read_excel("processed_data.xlsx", engine='openpyxl')

# 分离特征和输出
X = enhanced_data.iloc[:, :-1].values
y = enhanced_data.iloc[:, -1].values

# 删除异常值
cleaned_data = remove_outliers(np.hstack((X, y.reshape(-1, 1))))
cleaned_df = pd.DataFrame(cleaned_data, columns=enhanced_data.columns)

# 保存处理后的数据到Excel文件
cleaned_df.to_excel('cleaned_data_no_noise.xlsx', index=False)

# 分离清理后的特征和输出
X_cleaned = cleaned_df.iloc[:, :-1].values
y_cleaned = cleaned_df.iloc[:, -1].values

# 使用原始数据进行划分
X_train, X_temp, y_train, y_temp = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=9)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=8)

# 使用标准化
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

# 标准化输出列
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 使用网格搜索进行超参数调优
param_grid = {
    'hidden_layer_sizes': [(64, 32), (100, 50), (50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 2000]
}

mlp = MLPRegressor(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train_scaled)

# 打印最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_mlp = grid_search.best_estimator_
best_mlp.fit(X_train, y_train_scaled)

# 预测
y_train_pred_scaled = best_mlp.predict(X_train)
y_val_pred_scaled = best_mlp.predict(X_val)
y_test_pred_scaled = best_mlp.predict(X_test)

# 恢复原始数据范围
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

y_train = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
y_val = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# 计算误差
train_errors = np.abs(y_train - y_train_pred)
val_errors = np.abs(y_val - y_val_pred)
test_errors = np.abs(y_test - y_test_pred)

# 定义一个误差阈值，选择误差大于阈值的数据作为异常值
threshold = np.percentile(train_errors, 75)  # 95% 分位数作为阈值

# 找出误差大于阈值的样本
train_outliers = train_errors > threshold
val_outliers = val_errors > threshold
test_outliers = test_errors > threshold

# 打印出异常值数据
print("训练集中的异常值:")
print(pd.DataFrame(np.hstack((X_train[train_outliers], y_train[train_outliers].reshape(-1, 1))), columns=enhanced_data.columns).to_string(index=False))
print("\n验证集中的异常值:")
print(pd.DataFrame(np.hstack((X_val[val_outliers], y_val[val_outliers].reshape(-1, 1))), columns=enhanced_data.columns).to_string(index=False))
print("\n测试集中的异常值:")
print(pd.DataFrame(np.hstack((X_test[test_outliers], y_test[test_outliers].reshape(-1, 1))), columns=enhanced_data.columns).to_string(index=False))

# 删除异常值数据
X_train_no_outliers = X_train[~train_outliers]
y_train_no_outliers = y_train[~train_outliers]

X_val_no_outliers = X_val[~val_outliers]
y_val_no_outliers = y_val[~val_outliers]

X_test_no_outliers = X_test[~test_outliers]
y_test_no_outliers = y_test[~test_outliers]

# 重新标准化
X_train_no_outliers = scaler_X.fit_transform(X_train_no_outliers)
X_val_no_outliers = scaler_X.transform(X_val_no_outliers)
X_test_no_outliers = scaler_X.transform(X_test_no_outliers)
y_train_no_outliers_scaled = scaler_y.fit_transform(y_train_no_outliers.reshape(-1, 1)).flatten()
y_val_no_outliers_scaled = scaler_y.transform(y_val_no_outliers.reshape(-1, 1)).flatten()
y_test_no_outliers_scaled = scaler_y.transform(y_test_no_outliers.reshape(-1, 1)).flatten()

# 使用最佳参数重新训练模型
best_mlp.fit(X_train_no_outliers, y_train_no_outliers_scaled)

# 预测
y_train_pred_scaled_no_outliers = best_mlp.predict(X_train_no_outliers)
y_val_pred_scaled_no_outliers = best_mlp.predict(X_val_no_outliers)
y_test_pred_scaled_no_outliers = best_mlp.predict(X_test_no_outliers)

# 恢复原始数据范围
y_train_pred_no_outliers = scaler_y.inverse_transform(y_train_pred_scaled_no_outliers.reshape(-1, 1)).flatten()
y_val_pred_no_outliers = scaler_y.inverse_transform(y_val_pred_scaled_no_outliers.reshape(-1, 1)).flatten()
y_test_pred_no_outliers = scaler_y.inverse_transform(y_test_pred_scaled_no_outliers.reshape(-1, 1)).flatten()

y_train_no_outliers = scaler_y.inverse_transform(y_train_no_outliers_scaled.reshape(-1, 1)).flatten()
y_val_no_outliers = scaler_y.inverse_transform(y_val_no_outliers_scaled.reshape(-1, 1)).flatten()
y_test_no_outliers = scaler_y.inverse_transform(y_test_no_outliers_scaled.reshape(-1, 1)).flatten()

# 重新计算评价指标并打印
evaluate_performance(y_train_no_outliers, y_train_pred_no_outliers, "训练")
evaluate_performance(y_val_no_outliers, y_val_pred_no_outliers, "验证")
evaluate_performance(y_test_no_outliers, y_test_pred_no_outliers, "测试")

# 可视化结果
plot_results(y_train_no_outliers, y_train_pred_no_outliers, "训练集预测结果")
plot_results(y_val_no_outliers, y_val_pred_no_outliers, "验证集预测结果")
plot_results(y_test_no_outliers, y_test_pred_no_outliers, "测试集预测结果")

# 保存训练好的模型
joblib.dump(best_mlp, 'best_mlp_model_no_outliers.joblib')
joblib.dump(scaler_X, 'scaler_X.joblib')
joblib.dump(scaler_y, 'scaler_y.joblib')
