import pandas as pd
import numpy as np
import os
import glob
import scipy.stats as stats
from scipy.fft import fft
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def extract_enhanced_features(signal, fs=1000):

    feat = []
    
    feat.append(np.mean(signal))            # 均值
    feat.append(np.std(signal))             # 标准差
    feat.append(np.sqrt(np.mean(signal**2)))# 有效值 (RMS)
    feat.append(stats.kurtosis(signal))     # 峰度
    feat.append(stats.skew(signal))         # 偏度
    # 方根幅值 (SRA)
    sra = (np.mean(np.sqrt(np.abs(signal))))**2
    feat.append(sra)
    # 裕度因子 (Margin Factor)
    feat.append(np.max(np.abs(signal)) / (sra + 1e-6))

    # 频域特征
    n = len(signal)
    f_val = np.abs(fft(signal))[:n//2]
    total_energy = np.sum(f_val**2) + 1e-6
    feat.append(np.sum(f_val[:10]**2) / total_energy)  # 低频能量占比
    feat.append(np.sum(f_val[10:50]**2) / total_energy) # 中频能量占比
    feat.append(np.sum(f_val[50:]**2) / total_energy)  # 高频能量占比

    # --- C. 时频域特征
    # 使用 db4 小波进行 4 层分解
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    energies = [np.sum(c**2) for c in coeffs]
    total_e = sum(energies) + 1e-6
    probs = [e/total_e for e in energies]
    # 小波熵：反映信号波动的混乱程度
    entropy = -np.sum([p * np.log(p+1e-6) for p in probs])
    feat.append(entropy)
    feat.extend(probs) # 将各层能量比例也作为特征输入
        
    return np.array(feat)


def load_and_preprocess_with_resampling(directory_path, window_size=200, step_size=2):
    all_features, all_labels = [], []
    files = glob.glob(os.path.join(directory_path, "*.*"))
    
    print(f"开始重采样提取特征（步长={step_size}）...")
    
    for file in files:
        file_name = os.path.basename(file)
        # 标签逻辑
        if 'H-' in file_name or 'H for' in file_name: label = 'Healthy'
        elif 'crack' in file_name.lower(): label = 'Crack'
        elif 'erosion' in file_name.lower(): label = 'Erosion'
        elif 'twist' in file_name.lower() or 'twsist' in file_name.lower(): label = 'Twist'
        elif 'unbalance' in file_name.lower(): label = 'Unbalance'
        else: continue
            
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file, sep=';', decimal='.', encoding='utf-8')
            else:
                df = pd.read_excel(file)
                
            data = df.iloc[:, 1].values # 提取 Amplitude 列
            
            # 
            for i in range(0, len(data) - window_size + 1, step_size):
                window = data[i : i + window_size]
                features = extract_enhanced_features(window)
                all_features.append(features)
                all_labels.append(label)
        except Exception as e:
            print(f"处理文件 {file_name} 出错: {e}")

    X = np.array(all_features)
    y_raw = np.array(all_labels)
    
    # 标签编码与标准化
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"【采样成功】生成样本总数: {len(X)}, 覆盖特征维度: {X.shape[1]}")
    return X_scaled, y, le


def build_robust_net(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 调低学习率以保证训练稳定性
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def main_pipeline(data_path):
    X, y, le = load_and_preprocess_with_resampling(data_path, window_size=200, step_size=2)
    
    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 建模
    model = build_robust_net(input_dim=X.shape[1], num_classes=len(le.classes_))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 5. 评估结果
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("="*30)
    
    # 混淆矩阵可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Fault Diagnosis Confusion Matrix (Overlapping Sampling)')
    plt.show()
    
    return model

if __name__ == "__main__":
    DATA_PATH = r'D:\py_test\毕设\data'
    final_model = main_pipeline(DATA_PATH)
    
    # 保存结果
    if not os.path.exists(r'D:\py_test\毕设\model'): os.makedirs(r'D:\py_test\毕设\model')
    final_model.save(r'D:\py_test\毕设\model\enhanced_feature_model.keras')