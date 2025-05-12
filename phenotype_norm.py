import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# phenotype_file = '/data1/wangchengrui/data/gdnky/phenotype_all.txt'
# phenotype_file = '/data1/wangchengrui/rice1w/Data_shanghai.csv'
phenotype_file = '/data1/wangchengrui/data/hznd/phenos.csv'

# 设置Seaborn样式
sns.set(style="whitegrid")
# 读取数据
data = pd.read_csv(phenotype_file)

# 提取表型数据
# phenotypes = data.columns[2:8]
# phenotypes = ['grain_thickness', 'grain_length', 'grain_weight', 'transparence', 'chalkiness_degress', 'AC']
# phenotypes = ['Heading_date','Plant_height','Culm_length','Panicle_length','Panicle_number', 'Grain_yield']
phenotypes = ['Heading_date','Plant_height', 'Num_panicles', 'Yield', 'Grain_weight', 'Grain_length']

# 标准化处理
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data[phenotypes]), columns=phenotypes)


# 使用分位数截断法处理极端离群点
def truncate_outliers(df, lower_quantile=0.02, upper_quantile=0.98):
    lower_bound = df.quantile(lower_quantile)
    upper_bound = df.quantile(upper_quantile)
    return df.clip(lower=lower_bound, upper=upper_bound, axis=1), lower_bound, upper_bound


# 标准化前的箱线图
def plot_boxplot(data, title, colors, output_dir, flag):
    plt.figure(figsize=(12, 6), dpi=600)
    sns.boxplot(data=data, palette=colors, linewidth=1.5, width=0.5)
    # plt.title(title, fontsize=16)
    # plt.xlabel('Phenotypes', fontsize=18)
    plt.ylabel('Values', fontsize=20)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # 保存图片
    output_path = os.path.join(output_dir, f'{flag}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=600)

    # 关闭图形以释放内存
    plt.close()

# 自定义颜色方案
colors_dict = [
    '#104680', '#317CB7', '#6DADD1', '#8EC2E0',
    '#B6D7E8', '#C4E0ED', '#FBE3D5', '#F0D0C4',
    '#F6B293', '#EE8D75', '#DC6D57', '#B72230',
    '#6D011F'
]

# 绘制标准化前的箱线图
plot_boxplot(data[phenotypes], 'Boxplot of Phenotypes Before Standardization for rice4k', colors_dict,
             output_dir='/data1/wangchengrui/', flag='rice4k_bn')

# 标准化后的数据并截断离群点
data_truncated, lower_bounds, upper_bounds = truncate_outliers(data_standardized)

# 绘制标准化后的箱线图（去除极端离群点）
plot_boxplot(data_truncated, 'Boxplot of Phenotypes After Standardization (Outliers Truncated) for rice18k',
             colors_dict, output_dir='/data1/wangchengrui/', flag='rice4k_an')
