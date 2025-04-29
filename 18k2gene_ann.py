import pandas as pd
import pysam
import csv
import logging
import re
from multiprocessing import Pool, Manager

"""
    Rice18k 根据SNP SHAP值文件和原始vcf文件，从gtf文件中找到对应的基因区域
"""

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def read_feature_importance(file_path):
    """
    读取特征重要性文件

    参数:
        file_path (str): 特征重要性CSV文件路径

    返回:
        set: 特征集合, DataFrame: 特征重要性数据框
    """
    try:
        df = pd.read_csv(file_path)
        if 'Feature' not in df.columns:
            logging.error("特征文件缺少 'Feature' 列")
            return set(), pd.DataFrame()
        features = set(df['Feature'])
        if not features:
            logging.warning("特征文件中未找到任何特征")
        logging.info(f"成功读取 {len(features)} 个特征")
        return features, df
    except Exception as e:
        logging.error(f"读取特征文件时发生错误: {e}")
        return set(), pd.DataFrame()


def read_gtf(gtf_path, vcf_path):
    """
    读取GTF文件并构建基因区域映射

    参数:
        gtf_path (str): GTF文件路径
        vcf_path (str): VCF文件路径以验证染色体名称

    返回:
        dict: 基因区域映射
    """
    try:
        # 获取VCF文件中的有效染色体名称
        vcf = pysam.VariantFile(vcf_path)
        valid_chroms = set(vcf.header.contigs.keys())

        gene_region_map = {}
        with open(gtf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):  # 跳过注释行
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9 or fields[2] != 'gene':  # 只处理gene行
                    continue

                chrom = fields[0].replace('chr', '')  # 移除chr前缀
                if chrom not in valid_chroms:
                    logging.warning(f"跳过不在VCF文件中的染色体: {chrom}")
                    continue

                # 提取基因ID
                gene_id_match = re.search(r'gene_id "([^"]+)"', fields[8])
                if not gene_id_match:
                    logging.warning(f"无法解析基因ID: {fields[8]}")
                    continue
                gene_id = gene_id_match.group(1)

                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]

                if chrom not in gene_region_map:
                    gene_region_map[chrom] = []
                gene_region_map[chrom].append({
                    'gene_id': gene_id,
                    'start': start,
                    'end': end,
                    'strand': strand
                })

        logging.info(f"成功读取 {len(gene_region_map)} 条染色体的基因区域")
        return gene_region_map
    except Exception as e:
        logging.error(f"读取GTF文件时发生错误: {e}")
        return {}


def process_chromosome(chrom, gene_region_map, vcf_path, shared_features, output_path, upstream=2000, downstream=2000):
    """
    处理单个染色体的VCF文件并找到基因区域

    参数:
        chrom (str): 染色体名称
        gene_region_map (dict): 基因区域映射
        vcf_path (str): VCF文件路径
        shared_features (multiprocessing.Manager.list): 共享特征集合
        output_path (str): 输出文件路径
        upstream (int): 上游区域长度
        downstream (int): 下游区域长度
    """
    try:
        vcf = pysam.VariantFile(vcf_path)
        with open(output_path, 'a', newline='') as csvfile:
            fieldnames = ['Feature', 'Chromosome', 'Gene_ID', 'Gene_Start', 'Gene_End', 'Strand', 'Feature_Position']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()

            # 遍历基因区域
            for gene_region in gene_region_map.get(chrom, []):
                start = max(1, gene_region['start'] - upstream)
                end = gene_region['end'] + downstream

                for record in vcf.fetch(chrom, start, end):
                    if record.id and record.id in shared_features:
                        result_entry = {
                            'Feature': record.id,
                            'Chromosome': f'chr{chrom}',
                            'Gene_ID': gene_region['gene_id'],
                            'Gene_Start': gene_region['start'],
                            'Gene_End': gene_region['end'],
                            'Strand': gene_region['strand'],
                            'Feature_Position': record.pos
                        }
                        writer.writerow(result_entry)
                        shared_features.remove(record.id)

            # 记录未匹配的特征
            for record in vcf.fetch(chrom):
                if record.id and record.id in shared_features:
                    result_entry = {
                        'Feature': record.id,
                        'Chromosome': f'chr{chrom}',
                        'Gene_ID': '',
                        'Gene_Start': '',
                        'Gene_End': '',
                        'Strand': '',
                        'Feature_Position': record.pos
                    }
                    writer.writerow(result_entry)
                    shared_features.remove(record.id)

        logging.info(f"成功处理染色体 {chrom} 的特征位置")
    except Exception as e:
        logging.error(f"处理染色体 {chrom} 时发生错误: {e}")


def main():
    # 文件路径
    phenotype = 'Heading_date'
    feature_file = f'/data1/wangchengrui/rice1w/SHAP/{phenotype}_feature_importance_sorted.csv'
    vcf_file = '/data1/wangchengrui/rice1w/LD_NAM_Magic.vcf.gz'
    gtf_file = '/data1/wangchengrui/refs/osa/osa_IRGSP_1.annotation.gtf'
    intermediate_output_file = f'/data1/wangchengrui/rice1w/SHAP/{phenotype}_intermediate_mapping_results.csv'
    final_output_file = f'/data1/wangchengrui/rice1w/SHAP/{phenotype}_final_mapping_results.csv'

    # 读取特征
    features, feature_importance_df = read_feature_importance(feature_file)
    if not features:
        logging.error("未找到任何特征，程序终止")
        return

    # 读取GFF文件并构建基因区域映射
    gene_region_map = read_gtf(gtf_file, vcf_file)
    if not gene_region_map:
        logging.error("未找到任何基因区域，程序终止")
        return

    # 使用多进程处理
    manager = Manager()
    shared_features = manager.list(features)

    with Pool() as pool:
        pool.starmap(process_chromosome,
                     [(chrom, gene_region_map, vcf_file, shared_features, intermediate_output_file, 2000, 2000)
                      for chrom in gene_region_map.keys()],
                     chunksize=1)

    # 将剩余未匹配的特征转换回普通集合
    remaining_features = set(shared_features)
    if remaining_features:
        logging.warning(f"以下特征未找到对应的基因区域: {remaining_features}")

    # 合并中间结果文件
    try:
        mapping_results_df = pd.read_csv(intermediate_output_file)
        final_results_df = mapping_results_df.merge(feature_importance_df, on='Feature', how='left')
        final_results_df = final_results_df.sort_values('Importance_Value', ascending=False)
        final_results_df.to_csv(final_output_file, index=False)
        logging.info(f"最终结果已保存到 {final_output_file}")
    except Exception as e:
        logging.error(f"合并和排序结果时发生错误: {e}")


if __name__ == '__main__':
    main()