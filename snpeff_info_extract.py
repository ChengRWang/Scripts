"""
脚本功能：
1. 从压缩的VCF文件中提取SNP信息，并结合SNPEff注释字段解析变异的功能影响。
2. 通过加载多个表型的SHAP值文件，筛选出每个表型的Top N SNPs。
3. 对筛选出的SNPs进行注释解析，并将结果保存为CSV文件。
4. 提供统计信息，包括区域分布、影响程度分布和可能的顺式作用元件数量。
5. 优化：只提取第一个替代碱基（ALT字段的第一个碱基）的信息。
"""

import pandas as pd
import gzip
import re
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_snpeff_annotation(ann_field: str) -> dict:
    """
    解析SNPEff的注释字段，提取详细信息
    """
    # 分割注释字段，确保字段数量足够
    fields = ann_field.split('|')
    if len(fields) < 11:
        fields.extend([''] * (11 - len(fields)))  # 补充空字段
    elif len(fields) > 11:
        logger.warning(f"SNPEff注释字段数量超出预期: {len(fields)}")
        fields = fields[:11]  # 截断到前11个字段

    # 提取基本注释信息
    annotation_info = {
        'Allele': fields[0] if fields[0] else '',
        'Annotation': fields[1] if fields[1] else '',
        'Impact': fields[2] if fields[2] else '',
        'Gene_Name': fields[3] if fields[3] else '',
        'Gene_ID': fields[4] if fields[4] else '',
        'Feature_Type': fields[5] if fields[5] else '',
        'Feature_ID': fields[6] if fields[6] else '',
        'Transcript_BioType': fields[7] if fields[7] else '',
        'Rank': fields[8] if fields[8] else '',
        'HGVS_c': fields[9] if fields[9] else '',
        'HGVS_p': fields[10] if fields[10] else '',
    }

    # 区域分类字典
    region_classification = {
        'intergenic_region': '基因间区',
        'intron_variant': '内含子',
        'upstream_gene_variant': '上游区域',
        'downstream_gene_variant': '下游区域',
        'splice_acceptor_variant': '剪接受体位点',
        'splice_donor_variant': '剪接供体位点',
        '5_prime_UTR_variant': '5\'UTR区域',
        '3_prime_UTR_variant': '3\'UTR区域',
        'missense_variant': '错义突变',
        'synonymous_variant': '同义突变',
        'stop_gained': '终止密码子获得',
        'stop_lost': '终止密码子缺失',
        'start_lost': '起始密码子缺失',
        'frameshift_variant': '移码突变',
        'splice_region_variant': '剪接区域',
        'non_coding_transcript_variant': '非编码转录本',
        'non_coding_transcript_exon_variant': '非编码转录本外显子',
        'conservative_inframe_deletion': '保守的框内缺失',
        'conservative_inframe_insertion': '保守的框内插入',
        'disruptive_inframe_deletion': '破坏性框内缺失',
        'disruptive_inframe_insertion': '破坏性框内插入',
        '5_prime_UTR_premature_start_codon_gain_variant': '5\'UTR过早起始密码子获得',
        'start_retained_variant': '起始密码子保留',
        'stop_retained_variant': '终止密码子保留',
        'transcript_ablation': '转录本消除',
        'gene_fusion': '基因融合',
        'bidirectional_gene_fusion': '双向基因融合',
        'intragenic_variant': '基因内变异'
    }

    annotation_info['Region'] = region_classification.get(fields[1], '其他区域')

    # 判断是否可能为顺式作用元件
    cis_regulatory_elements = {
        'upstream_gene_variant': True,
        'downstream_gene_variant': True,
        '5_prime_UTR_variant': True,
        '3_prime_UTR_variant': True,
        'intergenic_region': True,
        '5_prime_UTR_premature_start_codon_gain_variant': True
    }

    annotation_info['Possible_Cis_Element'] = 'Yes' if fields[1] in cis_regulatory_elements else 'No'

    # 添加功能影响预测
    impact_description = {
        'HIGH': '高影响',
        'MODERATE': '中等影响',
        'LOW': '低影响',
        'MODIFIER': '修饰性影响'
    }

    annotation_info['Impact_Description'] = impact_description.get(fields[2], '未知影响')

    return annotation_info


def load_shap_files(phenotypes: List[str], shap_dir: str, top_n: int = 1000) -> Dict[str, dict]:
    """
    加载多个表型的SHAP值文件并创建SNP映射字典
    """
    all_snp_mappings = {}
    snp_pattern = re.compile(r'(vg\d+)_[ATCG]')

    for phenotype in phenotypes:
        shap_file = f'{shap_dir}/{phenotype}_v2_shap_values_sorted.csv'
        logger.info(f"Loading SHAP file for {phenotype}: {shap_file}")

        try:
            snp_mapping = {}
            with open(shap_file, 'r') as f:
                header = f.readline().strip().split(',')
                feature_idx = header.index('Feature')
                shap_value_idx = header.index('SHAP_Value')

                for line in f:
                    if len(snp_mapping) >= top_n:
                        break  # 达到top_n后停止读取

                    fields = line.strip().split(',')
                    feature = fields[feature_idx]
                    match = snp_pattern.match(feature)
                    if match:
                        base_id = match.group(1)
                        snp_mapping[base_id] = {
                            'full_id': feature,
                            'shap_value': float(fields[shap_value_idx])
                        }

            all_snp_mappings[phenotype] = snp_mapping
            logger.info(f"Loaded {len(snp_mapping)} SNPs for {phenotype}")

        except Exception as e:
            logger.error(f"Error loading SHAP file for {phenotype}: {e}")
            continue

    return all_snp_mappings


def process_multiple_phenotypes(
        phenotypes: List[str],
        shap_dir: str,
        vcf_gz_file: str,
        output_dir: str,
        top_n: int = 1000
) -> Dict[str, Tuple[pd.DataFrame, dict]]:
    """
    同时处理多个表型的SNP注释信息，只遍历一次VCF文件
    """
    logger.info(f"开始处理{len(phenotypes)}个表型的数据")

    # 加载所有表型的SHAP值文件
    all_snp_mappings = load_shap_files(phenotypes, shap_dir, top_n)

    # 创建结果存储结构
    results = {phenotype: [] for phenotype in phenotypes}

    # 读取VCF.GZ文件
    logger.info(f"开始处理VCF文件: {vcf_gz_file}")
    processed_count = 0

    with gzip.open(vcf_gz_file, 'rt') as vcf:
        for line in vcf:
            if line.startswith('#'):
                continue

            processed_count += 1
            if processed_count % 100000 == 0:
                logger.info(f"已处理 {processed_count} 行")

            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = fields[1]
            snp_id = fields[2]
            ref = fields[3]
            alt = fields[4].split(',')[0]  # 只提取第一个替代碱基
            info = fields[7]

            # 检查该SNP是否在任何表型的top SNPs中
            for phenotype, snp_mapping in all_snp_mappings.items():
                if snp_id in snp_mapping:
                    try:
                        # 解析ANN字段
                        if 'ANN=' not in info:
                            logger.warning(f"SNP {snp_id} 的INFO字段中未找到ANN字段")
                            continue
                        ann_field = info.split('ANN=')[1].split(';')[0]
                        annotation_info = parse_snpeff_annotation(ann_field)

                        # 合并基本信息和注释信息
                        result = {
                            'SNP_ID': snp_mapping[snp_id]['full_id'],
                            'CHROM': chrom,
                            'POS': pos,
                            'REF': ref,
                            'ALT': alt,
                            'SHAP_Value': snp_mapping[snp_id]['shap_value'],
                            **annotation_info
                        }

                        results[phenotype].append(result)

                    except Exception as e:
                        logger.warning(f"处理SNP {snp_id}时出错: {e}")
                        continue

    # 处理结果并生成统计信息
    final_results = {}
    for phenotype in phenotypes:
        if not results[phenotype]:
            logger.warning(f"表型 {phenotype} 没有找到匹配的SNP")
            continue

        # 创建DataFrame并排序
        result_df = pd.DataFrame(results[phenotype])
        result_df = result_df.sort_values('SHAP_Value', ascending=False)

        # 保存结果
        output_file = f'{output_dir}/{phenotype}_v2_shap_info.csv'
        result_df.to_csv(output_file, index=False)
        logger.info(f"已保存{phenotype}的结果到: {output_file}")

        # 生成统计信息
        stats = {
            'Total_Matched_SNPs': len(results[phenotype]),
            'Region_Distribution': result_df['Region'].value_counts(),
            'Impact_Distribution': result_df['Impact'].value_counts(),
            'Cis_Element_Count': result_df['Possible_Cis_Element'].value_counts()
        }

        final_results[phenotype] = (result_df, stats)

        logger.info(f"{phenotype}统计信息:")
        logger.info(f"找到并匹配的SNP数量: {len(results[phenotype])}")
        logger.info(f"可能缺失的SNP数量: {top_n - len(results[phenotype])}")

    return final_results


# 使用示例
if __name__ == "__main__":
    # 定义参数
    phenotypes = ['Plant_height', 'Heading_date']  # 添加所有需要分析的表型
    shap_dir = '/data1/wangchengrui/hznd_data/snp_shap'
    vcf_gz_file = '/data2/users/luojinjing/Tangping/nodel_variants.ann.vcf.gz'
    output_dir = '/data1/wangchengrui/hznd_data'

    # 处理多个表型
    results = process_multiple_phenotypes(
        phenotypes=phenotypes,
        shap_dir=shap_dir,
        vcf_gz_file=vcf_gz_file,
        output_dir=output_dir,
        top_n=1000
    )

    # 输出每个表型的统计信息
    for phenotype, (df, stats) in results.items():
        print(f"\n{phenotype} 统计信息:")
        print(f"总SNP数: {stats['Total_Matched_SNPs']}")
        print("\n区域分布:")
        print(stats['Region_Distribution'])
        print("\n影响程度分布:")
        print(stats['Impact_Distribution'])