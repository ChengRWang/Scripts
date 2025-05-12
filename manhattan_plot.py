import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_manht_input(file_path):
    """
    读取.manht_input格式的文件
    """
    df = pd.read_csv(file_path, sep='\s+')
    return df


def remove_outliers(data, threshold=2):
    """
    去除离群点
    """
    # 计算每个点的p值与其邻近点的差异
    data['p_diff'] = data['p'].diff().abs()

    # 标记离群点
    outliers = data[data['p_diff'] > threshold]

    # 返回去除离群点后的数据
    return data[~data.index.isin(outliers.index)]


def plot_manhattan_for_traits(input_dir, output_dir=None, threshold_value=None):
    """
    为每个表型绘制曼哈顿图

    Parameters:
    -----------
    input_dir : str
        包含.manht_input文件的目录路径
    output_dir : str, optional
        输出图片目录，默认为input_dir
    """
    # 颜色配置
    colors_dict = [
        '#44045A', '#413E85', '#30688D', '#1F928B', '#35B777',
        '#91D542', '#F8E620', '#E69F00', '#56B4E9', '#009E73',
        '#F0E442', '#CC79A7'
    ]

    # 获取所有.manht_input文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.manht_input')]

    # 如果未指定输出目录，使用输入目录
    if output_dir is None:
        output_dir = input_dir

        # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 为每个文件绘制曼哈顿图
    for file in input_files:
        # 提取表型名称（去掉前缀和后缀）
        trait_name = file.replace('mlm_output_', '').replace('.manht_input', '')

        # 读取数据
        file_path = os.path.join(input_dir, file)
        data = read_manht_input(file_path)

        # 去除离群点
        data = remove_outliers(data)

        # 对p值取负对数
        data['-log10_p'] = -np.log10(data['p'])

        # 设置图形样式
        plt.figure(figsize=(12, 6), dpi=600)
        plt.style.use('seaborn-v0_8-white')

        # 按染色体分组并上色
        unique_chrs = sorted(data['Chr'].unique())
        chr_colors = {chr: colors_dict[i % len(colors_dict)] for i, chr in enumerate(unique_chrs)}

        # 计算每条染色体的累积位置
        data['cumulative_pos'] = 0
        cumulative_length = 0
        chr_cumulative_pos = {}

        for chr in unique_chrs:
            chr_data = data[data['Chr'] == chr]
            chr_cumulative_pos[chr] = cumulative_length + chr_data['Pos'].max() / 2
            data.loc[data['Chr'] == chr, 'cumulative_pos'] = chr_data['Pos'] + cumulative_length
            cumulative_length += chr_data['Pos'].max()

            # 绘制散点
        for chr in unique_chrs:
            chr_data = data[data['Chr'] == chr]
            plt.scatter(
                chr_data['cumulative_pos'],
                chr_data['-log10_p'],
                color=chr_colors[chr],
                alpha=0.7,
                edgecolors='none',
                s=20
            )

            # 添加显著性阈值线
        if threshold_value is not None:
            plt.axhline(y=threshold_value, color='red', linestyle='--', linewidth=1,
                        label=f'Significance threshold (-log10(p) = {threshold_value:.2f})')

            # 设置x轴刻度
        plt.xticks(
            [chr_cumulative_pos[chr] for chr in unique_chrs],
            [str(chr) for chr in unique_chrs],
            rotation=0
        )

        # 设置标签和标题
        plt.xlabel('Chromosome', fontsize=16)
        plt.ylabel('-log10(p-value)', fontsize=16)
        # plt.title(f'Manhattan Plot - {trait_name}', fontsize=14, fontweight='bold')

        max_log10p = data['-log10_p'].max()
        plt.ylim(0, max_log10p * 1.1)  # 比最大值多10%

        # 添加图例，并指定具体位置
        if threshold_value is not None:
            plt.legend(loc='upper right')

            # 调整布局
        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(output_dir, f'new_manhattan_plot_{trait_name}_th{threshold_value}.pdf')
        plt.savefig(output_path, bbox_inches='tight', format='pdf')

        # 关闭图形以释放内存
        plt.close()

        print(f"Generated Manhattan plot for {trait_name}")


input_dir = '/data1/wangchengrui/hznd_data/gwas_input/input_new'
output_dir = '/data1/wangchengrui/hznd_data/gwas_input/results_new'
plot_manhattan_for_traits(input_dir, output_dir, threshold_value=8)