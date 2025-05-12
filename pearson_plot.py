import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置seaborn样式
sns.set_style("whitegrid")

# 新的颜色方案
colors_dict = [
    '#104680',
    '#317CB7',
    '#6DADD1',
    '#8EC2E0',
    '#B6D7E8',
    '#C4E0ED',
    '#FBE3D5',
    '#F0D0C4',
    '#F6B293',
    '#EE8D75',
    '#DC6D57',
    '#B72230',
    '#6D011F',
    '#4B000E'
]


def process_directory(dir_path, dir_name):
    # files = [f for f in os.listdir(dir_path) if f.endswith('_metrics.csv')]
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    all_data = []

    # exclude_traits = ['Grain_thickness', 'Num_effective_panicles', 'Num_panicles', 'Spikelet_length',
    #                   'Brown_rice_ratio', 'Polished_rice_ratio', 'Complete_polished_rice_ratio', 'Transparence',
    #                   'LSvsT', 'Chalkiness_degree', 'AC', 'Gel_consistency', 'Alkali_spreading',
    #                   'Leaf_length', 'Leaf_angle', 'Leaf_width', 'Panicle_length', 'Grain_protein_content']
    # exclude_traits = []

    for file in files:
        # trait = file.split('_metrics')[0].split('_')[-1]
        # trait = file.split('_metrics')[0]
        trait = file.split('.')[1]
        # if dir_name == 'exp_ml':
        #     trait = file.split('_metrics')[0].split('transpose.tsv_')[1]
        # elif dir_name == 'snp_ml':
        #     trait = file.split('_metrics')[0].split('hz.snp.1w_')[1]
        # elif dir_name == 'merge_ml':
        #     trait = file.split('_metrics')[0].split('merge_')[1]

        # if trait in exclude_traits:
        #     continue

        df = pd.read_csv(os.path.join(dir_path, file))
        df['Trait'] = trait
        df['Model'] = df['Model'].replace('RRBLUP', 'rrBLUP')

        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)

    color_dict = {
        'CatBoost': colors_dict[0],
        'GradientBoosting': colors_dict[1],
        'RandomForest': colors_dict[2],
        'LightGBM': colors_dict[3],
        'XGBoost': colors_dict[4],
        'KNeighborsRegressor': colors_dict[5],
        'Lasso': colors_dict[6],
        'ElasticNet': colors_dict[7],
        'LinearRegression': colors_dict[8],
        'Ridge': colors_dict[9],
        'DeepGS': colors_dict[10],
        'DNNGP': colors_dict[11],
        'TransformerGP': colors_dict[12],
    }

    plt.figure(figsize=(16, 9))  # 增加图形宽度以适应右侧图例
    if 'exp' in dir_name:
        h_order = ['CatBoost', 'GradientBoosting', 'RandomForest', 'LightGBM',
                   'XGBoost', 'KNeighborsRegressor', 'Lasso', 'ElasticNet',
                   'LinearRegression', 'Ridge', 'DeepGS', 'DNNGP', 'TransformerGP']
    else:
        h_order = ['CatBoost', 'GradientBoosting', 'RandomForest', 'LightGBM',
                   'XGBoost', 'KNeighborsRegressor', 'Lasso', 'ElasticNet',
                   'LinearRegression', 'Ridge', 'rrBLUP', 'DeepGS', 'DNNGP', 'TransformerGP']

    ax = sns.barplot(x='Trait', y='Pearson_Mean', hue='Model', data=combined_data,
                     palette=colors_dict, hue_order=h_order)

    # 将图例移到右侧且不遮挡图像
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='16')

    traits = combined_data['Trait'].unique()
    for trait in traits:
        trait_data = combined_data[combined_data['Trait'] == trait]
        # catboost_value = trait_data[trait_data['Model'] == 'CatBoost']['Pearson_Mean']
        # transformer_value = trait_data[trait_data['Model'] == 'TransformerGP']['Pearson_Mean']
        catboost_value = trait_data[trait_data['Model'] == 'CatBoost']['Pearson_Mean'].values[0]
        transformer_value = trait_data[trait_data['Model'] == 'TransformerGP']['Pearson_Mean'].values[0]
        max_value = trait_data['Pearson_Mean'].max()
        max_model = trait_data.loc[trait_data['Pearson_Mean'] == max_value, 'Model'].values[0]

        trait_index = list(traits).index(trait)
        plt.text(trait_index - 0.2, catboost_value, f'{catboost_value:.3f}',
                 ha='center', va='bottom', fontsize='10')

        plt.text(trait_index + 0.2, transformer_value, f'{transformer_value:.3f}',
                 ha='center', va='bottom', fontsize='10')

        # 如果 CatBoost 和 TransformerGP 都不是最高精度，标注最高精度
        if max_model not in ['CatBoost', 'TransformerGP']:
            plt.text(trait_index, max_value, f'{max_value:.3f}',
                     ha='center', va='bottom')

    title_map = {
        'exp_gd': 'Transcriptome Features Performance Comparison',
        'snp_gd': 'SNP Features Performance Comparison',
        'merge_gd': 'Merge Features Performance Comparison'
    }

    # plt.title(title_map[dir_name])
    plt.ylabel('Pearson Correlation', size=20)
    # plt.xlabel('Traits', size=16)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴字体大小为12
    ax.tick_params(axis='y', labelsize=20)  # 设置y轴字体大小为12

    # 分别计算CatBoost优于每个模型的优势
    models = combined_data['Model'].unique()
    for model in models:
        if model != 'CatBoost':
            trait_advantages = []
            print(f"\n{dir_name} - CatBoost vs {model} comparison:")
            for trait in traits:
                trait_data = combined_data[combined_data['Trait'] == trait]
                catboost_value = trait_data[trait_data['Model'] == 'CatBoost']['Pearson_Mean'].values[0]
                model_value = trait_data[trait_data['Model'] == model]['Pearson_Mean'].values[0]
                advantage = catboost_value - model_value
                trait_advantages.append(advantage)
                print(f"Trait: {trait}, Advantage: {advantage:.4f}")
            avg_advantage = np.mean(trait_advantages)
            print(f"Average advantage over {model}: {avg_advantage:.4f}")

    plt.savefig(f'./pc/{dir_name}_pearson.pdf', bbox_inches='tight', dpi=600, format='pdf')
    plt.close()

    return combined_data


def compare_feature_types(exp_data, snp_data, merge_data):
    # 只获取CatBoost的数据
    model = 'TransformerGP'
    # model = 'CatBoost'
    exp_catboost = exp_data[exp_data['Model'] == model].copy()
    snp_catboost = snp_data[snp_data['Model'] == model].copy()
    merge_catboost = merge_data[merge_data['Model'] == model].copy()

    # 添加特征类型标签
    exp_catboost['Feature_Type'] = 'Transcriptome'
    snp_catboost['Feature_Type'] = 'SNP'
    merge_catboost['Feature_Type'] = 'Merge'

    # 合并数据
    combined_data = pd.concat([exp_catboost, snp_catboost, merge_catboost])
    print(combined_data)

    plt.figure(figsize=(16, 9))

    # 创建颜色映射字典
    color_dict = {
        'Transcriptome': colors_dict[2],  # '#104680'
        'SNP': colors_dict[5],           # '#317CB7'
        'Merge': colors_dict[8]          # '#6DADD1'
    }

    ax = sns.barplot(x='Trait', y='Pearson_Mean', hue='Feature_Type',
                     data=combined_data, palette=color_dict, hue_order=['Transcriptome', 'SNP', 'Merge'])

    # 将图例移到右侧且不遮挡图像
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='18')

    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')

    # plt.title('TransformerGP Performance Across Different Feature Types')

    # plt.xlabel('Traits', size=16)
    plt.ylabel('Pearson Correlation', size=16)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴字体大小为12
    ax.tick_params(axis='y', labelsize=20)  # 设置y轴字体大小为12

    plt.savefig(f'./pc/rice333_{model}_comparison.png', bbox_inches='tight', dpi=600)
    plt.close()

    # 打印每个表型的比较结果
    print("\nComparison across feature types:")
    for trait in combined_data['Trait'].unique():
        print(f"\nTrait: {trait}")
        trait_data = combined_data[combined_data['Trait'] == trait]
        for feature_type in ['Transcriptome', 'SNP', 'Merge']:
            value = trait_data[trait_data['Feature_Type'] == feature_type]['Pearson_Mean'].values[0]
            print(f"{feature_type}: {value:.4f}")


def summarize_catboost_advantage_by_type(data, data_type):
    data.to_csv('dataframe.csv', index=False)
    # 计算CatBoost的平均值
    catboost_mean = data[data['Model'] == 'CatBoost']['Pearson_Mean'].mean() * 100

    # 计算相对于其他模型的优势
    advantages = {}
    other_models = data['Model'].unique()
    for model in other_models:
        if model != 'CatBoost':
            model_mean = data[data['Model'] == model]['Pearson_Mean'].mean() * 100
            advantage = catboost_mean - model_mean
            advantages[model] = advantage

    # 生成描述文本
    advantage_text = ', '.join([f"{model}({advantage:.2f}%)" for model, advantage in advantages.items()])

    summary = f"在{data_type}特征中，CatBoost模型的预测精度显著高于其他模型，其均值达到了{catboost_mean:.2f}%，分别高于模型{advantage_text}。"

    print(f"\n{data_type} Results Summary:")
    print(summary)

    return {
        'catboost_mean': catboost_mean,
        'advantages': advantages
    }


# 处理目录并保存数据
# base_path = r'E:\Results'   #4k
# snp_data = process_directory(os.path.join(base_path, 'snp_ml'), 'snp_ml')
# exp_data = process_directory(os.path.join(base_path, 'exp_ml'), 'exp_ml')
# merge_data = process_directory(os.path.join(base_path, 'merge_ml'), 'merge_ml')
base_path = r'E:\py_dataprocess\py_dataprocess\pycaret_env'
snp_data = process_directory(os.path.join(base_path, '18k'), '18k')
# snp_data = process_directory(os.path.join(base_path, 'snp_gd'), 'snp_gd')
# exp_data = process_directory(os.path.join(base_path, 'exp_gd'), 'exp_gd')
# merge_data = process_directory(os.path.join(base_path, 'merge_gd'), 'merge_gd')
# exp_data.to_csv('exp_4k.csv', index=False)
# snp_data.to_csv('snp_4k.csv', index=False)
# merge_data.to_csv('merge_4k.csv', index=False)

# # 生成SNP和Transcriptome的总结
# summary_stats_snp = summarize_catboost_advantage_by_type(snp_data, "SNP")
# summary_stats_trans = summarize_catboost_advantage_by_type(exp_data, "Transcriptome")
#
# 比较特征类型
# compare_feature_types(exp_data, snp_data, merge_data)

# comprehensive_data = compare_feature_types_comprehensive(exp_data, snp_data, merge_data)