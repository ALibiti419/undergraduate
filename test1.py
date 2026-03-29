import pandas as pd
import numpy as np
import os
import glob
import re
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_label_data(directory_path):
    all_data = []
    files = glob.glob(os.path.join(directory_path, "*.*"))
    
    for file in files:
        file_name = os.path.basename(file)
        # 标签逻辑
        if 'H-' in file_name or 'H for' in file_name: label = 'Healthy'
        elif 'crack' in file_name.lower(): label = 'Crack'
        elif 'erosion' in file_name.lower(): label = 'Erosion'
        elif 'twist' in file_name.lower() or 'twsist' in file_name.lower(): label = 'Twist'
        elif 'unbalance' in file_name.lower(): label = 'Unbalance'
        else: continue
        

        vw_match = re.search(r'Vw[a-zA-Z]*=([0-9.]+)', file_name)
        
        if vw_match:
            vw_str = vw_match.group(1)
            if vw_str.endswith('.'):
                vw_str = vw_str[:-1]
            try:
                vw_val = float(vw_str)
            except:
                vw_val = 0.0
        else:
            vw_val = 0.0
            
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file, sep=';', decimal='.', encoding='utf-8')
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file)
            
            df.columns = ['Time', 'Amplitude']
            df['Label'] = label
            df['Vw'] = vw_val
            df['Source'] = file_name
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return pd.concat(all_data, ignore_index=True)


def preprocess_pipeline(df, window_size=200, overlap=5):
    segments, labels, winds = [], [], []
    
    for source in df['Source'].unique():
        subset = df[df['Source'] == source]
        data = subset['Amplitude'].values
        target = subset['Label'].iloc[0]
        vw = subset['Vw'].iloc[0]
        
        if len(data) < window_size: continue
        
        for i in range(0, len(data) - window_size + 1, overlap):
            segments.append(data[i : i + window_size])
            labels.append(target)
            winds.append(vw)
            
    X_vib = np.array(segments)
    y_raw = np.array(labels)
    X_vw = np.array(winds).reshape(-1, 1)
    
    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # 振动信号标准化
    scaler_vib = StandardScaler()
    X_vib_scaled = scaler_vib.fit_transform(X_vib).reshape(-1, window_size, 1)
    
    # 风速标准化
    scaler_vw = StandardScaler()
    X_vw_scaled = scaler_vw.fit_transform(X_vw)
    
    return X_vib_scaled, X_vw_scaled, y, le


def build_multi_input_model(vib_shape, num_classes):
    # 振动信号处理 (CNN-LSTM)
    vib_input = Input(shape=vib_shape, name='Vibration_Input')
    x = layers.Conv1D(64, 3, activation='relu')(vib_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
    
    # 风速数值处理
    vw_input = Input(shape=(1,), name='WindSpeed_Input')
    y = layers.Dense(16, activation='relu')(vw_input)
    
    # 特征融合
    combined = layers.concatenate([x, y])
    
    # 共同决策层
    z = layers.Dense(32, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    model = models.Model(inputs=[vib_input, vw_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main_pipeline(data_path):
    raw_df = load_and_label_data(data_path)

    X_vib, X_vw, y, le = preprocess_pipeline(raw_df, window_size=250, overlap=10)
    
    # 划分数据集
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    
    X_vib_train, X_vib_test = X_vib[train_idx], X_vib[test_idx]
    X_vw_train, X_vw_test = X_vw[train_idx], X_vw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = build_multi_input_model(vib_shape=(X_vib.shape[1], 1), num_classes=len(le.classes_))
    
    print("开始训练多输入模型...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        [X_vib_train, X_vw_train], y_train,
        epochs=30, batch_size=16,   # 10， 8
        validation_data=([X_vib_test, X_vw_test], y_test),
        callbacks=[early_stopping]
    )
    
    y_pred = np.argmax(model.predict([X_vib_test, X_vw_test]), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return model

if __name__ == "__main__":
    data_path = r'D:\py_test\毕设\data'
    save_path = r'D:\py_test\毕设\model'

    trained_model = main_pipeline(data_path)
    model_file = os.path.join(save_path, 'wind_turbine_fault_model.keras')
    trained_model.save(model_file)
    
    print(f"模型已成功保存至: {model_file}")
    
    trained_model.save_weights(os.path.join(save_path, 'model_weights.weights.h5'))