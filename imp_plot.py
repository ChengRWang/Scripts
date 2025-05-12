import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os
import numpy as np

# 设置matplotlib后端为Agg以提高性能
mpl.use('Agg')

# 特征名称映射字典
trait_name_dict = {
    'Grain_length': 'Grain Length',
    'Grain_width': 'Grain Width',
    'Grain_thickness': 'Grain Thickness',
    'Grain_weight': 'Grain Weight',
    'Heading_date': 'Heading Date',
    'Plant_height': 'Plant Height',
    'Spikelet_length': 'Spikelet Length',
    'Num_effective_panicles': 'Number of Effective Panicles',
    'Num_panicles': 'Number of Panicles',
    'Yield': 'Yield'
}

# 定义颜色列表
colors_dict = [
    '#104680', '#317CB7', '#6DADD1', '#8EC2E0', '#B6D7E8',
    '#C4E0ED', '#FBE3D5', '#F0D0C4', '#F6B293', '#EE8D75',
    '#DC6D57', '#B72230', '#6D011F'
]


def plot_feature_importance(file_path, output_dir, top_n=10):
    try:
        print(f"\nProcessing file: {os.path.basename(file_path)}")

        # 读取数据
        print("Reading data...")
        df = pd.read_csv(file_path)
        print(f"Found {len(df)} features")

        # 获取特征名和重要性值
        features = df.iloc[:, 0]
        importance = df.iloc[:, 1]

        # 标准化重要性值到0-1之间
        importance_normalized = (importance - importance.min()) / (importance.max() - importance.min())

        # 排序（从大到小）并选择前top_n个特征
        sorted_idx = importance_normalized.argsort()[::-1]  # 反转顺序，使最重要的在前
        features = features.iloc[sorted_idx[:top_n]]
        importance_normalized = importance_normalized.iloc[sorted_idx[:top_n]]

        # 反转顺序以便在图中从上到下显示重要性递减
        features = features[::-1]
        importance_normalized = importance_normalized[::-1]

        # 创建颜色映射
        colors = colors_dict[:top_n][::-1]  # 反转颜色顺序以匹配特征顺序

        print("Creating plot...")
        # 设置固定的图片高度
        figure_height = 6  # 固定高度为6英寸

        # 创建图形
        fig, ax = plt.subplots(figsize=(6, figure_height))

        # 绘制横向柱状图
        bars = ax.barh(range(len(features)), importance_normalized, color=colors)

        # 设置y轴标签
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)

        # 设置x轴标签
        ax.set_xlabel('Normalized Feature Importance', size=16)
        ax.tick_params(axis='x', labelsize=14)  # 设置x轴字体大小为12
        ax.tick_params(axis='y', labelsize=14)  # 设置y轴字体大小为12

        # 获取文件名中的性状名称
        # trait_name = os.path.basename(file_path).split('_feature_importance')[0]
        # trait_name = os.path.basename(file_path).split('_shap')[0]
        trait_name = os.path.basename(file_path).replace('exp_', '').split('_shap')[0]
        formatted_trait_name = trait_name_dict.get(trait_name, trait_name.replace('_', ' '))
        # ax.set_title(f'Gene Importance Ranking for {formatted_trait_name} in CatBoost\n(Top {top_n} Features) ')
        # ax.set_title(f'水稻{formatted_trait_name}相关的TOP 10单核苷酸多态性特征')

        # 添加网格线
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # 添加数值标签
        for i, v in enumerate(importance_normalized):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')

        # 调整布局
        print("Adjusting layout...")
        plt.tight_layout()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存图片
        # output_path = os.path.join(output_dir, f'{trait_name}_top{top_n}_feature_importance.pdf')
        output_path = os.path.join(output_dir, f'{trait_name}_top{top_n}_shap.pdf')
        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
        print("Done!")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")


# 处理所有feature_importance文件
print("Starting to process files...")
# data_dir = '/data1/wangchengrui/rice1w/SHAP'
# data_dir = '/data1/wangchengrui/Results/gdnky/exp'
# data_dir = '/data1/wangchengrui/hznd_data/catboost'
data_dir = '/data1/wangchengrui/final_results/eqtl/rice4k.eQTL_GWAS.results/exp_shap'
# feature_importance_files = glob.glob(os.path.join(data_dir, '*_feature_importance_sorted.csv'))
feature_importance_files = glob.glob(os.path.join(data_dir, '*_shap_values_sorted.csv'))
print(f"Found {len(feature_importance_files)} files to process")

# 设置要显示的特征数量
TOP_N_FEATURES = 10

for i, file_path in enumerate(feature_importance_files, 1):
    print(f"\nProcessing file {i} of {len(feature_importance_files)}")
    plot_feature_importance(file_path, data_dir, top_n=TOP_N_FEATURES)

print("\nAll files processed!")