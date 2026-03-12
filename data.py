import pandas as pd
import os
import glob

# 1. 设置数据集路径
dataset_path = '.\data'  # 请替换为存放CSV文件的实际路径

def load_wind_turbine_data(path):
    all_data = []
    
    # 获取目录下所有的 csv 和 xlsx 文件
    files = glob.glob(os.path.join(path, "*.*"))
    
    for file in files:
        file_name = os.path.basename(file)
        
        # 提取工况分类 (Label)
        if 'H-' in file_name or 'H for' in file_name:
            label = 'Healthy'
        elif 'crack' in file_name.lower():
            label = 'Crack'
        elif 'erosion' in file_name.lower():
            label = 'Erosion'
        elif 'twist' in file_name.lower():
            label = 'Imbalance'
        else:
            label = 'Unknown'
            
        # 读取数据
        try:
            if file.endswith('.csv'):
                # 根据截图，CSV 使用分号分隔
                df = pd.read_csv(file, sep=';', decimal='.', encoding='utf-8')
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file)
            
            # 统一列名（假设第一列是时间，第二列是振动值/电压）
            df.columns = ['Time', 'Vibration']
            
            # 添加标签和来源文件信息
            df['Label'] = label
            df['Source'] = file_name
            
            all_data.append(df)
            print(f"成功读取: {file_name}，工况识别为: {label}")
            
        except Exception as e:
            print(f"读取文件 {file_name} 出错: {e}")

    # 合并为一个大的 DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# 2. 执行读取
df_combined = load_wind_turbine_data(dataset_path)

# 3. 查看基本信息
print("\n数据读取完毕！")
print(df_combined.info())
print("\n工况样本分布统计：")
print(df_combined['Label'].value_counts())

# 4. 展示前5行数据
print("\n数据样例：")
print(df_combined.head())