import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据读取模块
# ==========================================
def load_and_label_data(directory_path):
    """读取CSV/Excel文件并根据文件名自动打标签"""
    all_data = []
    files = glob.glob(os.path.join(directory_path, "*.*"))
    
    for file in files:
        file_name = os.path.basename(file)
        # 标签逻辑：Healthy, Crack, Erosion, Imbalance
        if 'H-' in file_name or 'H for' in file_name: label = 'Healthy'
        elif 'crack' in file_name.lower(): label = 'Crack'
        elif 'erosion' in file_name.lower(): label = 'Erosion'
        elif 'twist' in file_name.lower(): label = 'Twist'
        elif 'unbalance' in file_name.lower(): label = 'Unbalance'
        else: continue
            
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file, sep=';', decimal='.', encoding='utf-8')
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file)
            
            df.columns = ['Time', 'Amplitude']
            df['Label'] = label
            df['Source'] = file_name
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return pd.concat(all_data, ignore_index=True)

# ==========================================
# 2. 预处理与重叠采样模块
# ==========================================
def preprocess_pipeline(df, window_size=500, overlap=100):
    """滑动窗口切分、标准化及标签编码"""
    segments, labels = [], []
    
    print(f"正在处理数据文件，总行数: {len(df)}")
    
    for source in df['Source'].unique():
        subset = df[df['Source'] == source]
        data = subset['Amplitude'].values
        target = subset['Label'].iloc[0]
        
        # --- 修正点：即使行数刚好等于 window_size，也至少取一个样本 ---
        if len(data) < window_size:
            print(f"警告：文件 {source} 行数({len(data)})小于窗口大小({window_size})，已跳过")
            continue
        
        # 如果行数正好等于 window_size，强制取这唯一的 500 个点
        if len(data) == window_size:
            segments.append(data)
            labels.append(target)
        else:
            # 正常滑动窗口逻辑
            for i in range(0, len(data) - window_size + 1, overlap):
                segments.append(data[i : i + window_size])
                labels.append(target)
            
    X = np.array(segments)
    y_raw = np.array(labels)
    
    if len(X) < 5:  # 如果样本太少，报错提示
        raise ValueError(f"生成的总样本数太少（仅 {len(X)} 个），请减小 window_size 或减小 overlap 步长！")

    print(f"成功生成样本总数: {len(X)}")

    # 编码与标准化
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    scaler = StandardScaler()
    # 注意：StandardScaler 需要 2D 输入，处理后再变回原样
    X_scaled = scaler.fit_transform(X)
    
    # 适配1D-CNN形状: (样本数, 时间步长, 通道数)
    X_final = X_scaled.reshape(-1, window_size, 1)
    
    return X_final, y, le

# ==========================================
# 3. CNN-LSTM 模型架构模块
# ==========================================
def build_cnn_lstm(input_shape, num_classes):
    """构建CNN-LSTM混合神经网络"""
    model = models.Sequential([
        # 特征提取层 (CNN)
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # 时序建模层 (LSTM)
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.5),
        
        # 分类决策层
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. 训练与评估可视化模块
# ==========================================
def evaluate_model(model, X_test, y_test, label_encoder):
    """生成混淆矩阵及分类报告（用于论文插图）"""
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix of Wind Turbine Fault Diagnosis')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# ==========================================
# 5. 主程序调用逻辑（Main Pipeline）
# ==========================================
def main_pipeline(data_path):
    # 1. 读取
    raw_df = load_and_label_data(data_path)
    if raw_df.empty:
        print("错误：未能在指定目录读取到任何有效 CSV 或 Excel 文件！")
        return None, None
    
    # 2. 采样与划分
    try:
        X, y, le = preprocess_pipeline(raw_df, window_size=200, overlap=5) 
        
        # 增加 stratify 参数，确保训练集和测试集中各工况比例一致
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception as e:
        print(f"采样或划分数据集失败: {e}")
        return None, None
    
    # 3. 建模
    model = build_cnn_lstm(input_shape=(X.shape[1], 1), num_classes=len(le.classes_))
    
    # 4. 训练
    print("开始训练模型...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,
    restore_best_weights=True # 自动回滚
    )

    history = model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping] # 早停
    )
    
    evaluate_model(model, X_test, y_test, le)
    
    return model, history

# 运行主程序
if __name__ == "__main__":
    trained_model, train_history = main_pipeline(r'D:\py_test\毕设\data')