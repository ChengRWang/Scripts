"""
脚本功能：
1. 从SNP数据文件、基因数据文件和GFF注释文件中加载数据。
2. 分析SNP和基因之间的连锁关系，基于距离阈值（默认1Mb）筛选潜在的连锁对。
3. 将分析结果保存为CSV文件，并生成统计摘要。
4. 提供灵活的日志记录和错误处理机制，便于调试和监控。

主要功能模块：
- `GeneSnpLinkageAnalyzer` 类：负责加载数据、计算连锁关系并生成分析结果。
- `main` 函数：主程序入口，协调数据加载、分析和结果保存。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import sys
from datetime import datetime


# 设置日志
def setup_logger(log_file: str = "linkage_analysis.log"):
    """
    配置日志记录器，支持将日志同时输出到文件和控制台。
    :param log_file: 日志文件路径
    :return: 配置好的日志记录器实例
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 将日志写入文件
            logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
        ]
    )
    return logging.getLogger(__name__)


class GeneSnpLinkageAnalyzer:
    def __init__(self, threshold: int = 1000000):
        """
        初始化分析器，设置连锁分析的距离阈值和日志记录器。
        :param threshold: SNP和基因之间判定为连锁的最大距离（默认1Mb）
        """
        self.threshold = threshold
        self.logger = setup_logger()

    def load_snp_data(self, snp_file: str) -> pd.DataFrame:
        """
        加载SNP数据，确保必要的列存在并标准化染色体格式。
        :param snp_file: SNP数据文件路径
        :return: 包含SNP信息的DataFrame
        """
        try:
            snp_df = pd.read_csv(snp_file)
            # 确保必要的列存在
            required_cols = ['SNP_ID', 'CHROM', 'POS', 'REF', 'ALT']
            missing_cols = [col for col in required_cols if col not in snp_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # 标准化染色体列的格式（转换为小写）
            snp_df['CHROM'] = snp_df['CHROM'].str.lower()

            self.logger.info(f"Successfully loaded {len(snp_df)} SNPs")
            return snp_df
        except Exception as e:
            self.logger.error(f"Error loading SNP data: {str(e)}")
            raise

    def load_gene_data(self, gene_file: str) -> pd.DataFrame:
        """
        加载基因数据。
        :param gene_file: 基因数据文件路径
        :return: 包含基因信息的DataFrame
        """
        try:
            gene_df = pd.read_csv(gene_file)
            self.logger.info(f"Successfully loaded {len(gene_df)} genes")
            return gene_df
        except Exception as e:
            self.logger.error(f"Error loading gene data: {str(e)}")
            raise

    def load_gff_data(self, gff_file: str) -> pd.DataFrame:
        """
        加载GFF注释文件，解析attributes字段并提取基因ID和名称。
        :param gff_file: GFF文件路径
        :return: 处理后的GFF DataFrame
        """
        try:
            # 跳过注释行，读取GFF文件
            gff_df = pd.read_csv(gff_file,
                                 sep='\t',
                                 comment='#',
                                 names=['seqid', 'source', 'type', 'start', 'end',
                                        'score', 'strand', 'phase', 'attributes'])

            # 只保留基因相关条目（'gene' 和 'mRNA'）
            gff_df = gff_df[gff_df['type'].isin(['gene', 'mRNA'])]

            # 解析 attributes 列
            def parse_attributes(attr_str: str) -> Dict[str, str]:
                """解析 GFF 的 attributes 字段"""
                try:
                    return dict(item.split('=') for item in attr_str.split(';'))
                except Exception:
                    return {}

            gff_df['parsed_attrs'] = gff_df['attributes'].apply(parse_attributes)
            gff_df['gene_id'] = gff_df['parsed_attrs'].apply(lambda x: x.get('ID', None))
            gff_df['gene_name'] = gff_df['parsed_attrs'].apply(lambda x: x.get('Name', None))

            # 标准化染色体ID（转换为小写）
            gff_df['seqid'] = gff_df['seqid'].str.lower()

            self.logger.info(f"Successfully loaded GFF data with {len(gff_df)} entries")
            return gff_df
        except Exception as e:
            self.logger.error(f"Error loading GFF data: {str(e)}")
            raise

    def find_gene_position(self, gene_id: str, gff_df: pd.DataFrame) -> Optional[Dict]:
        """
        在GFF文件中查找基因的位置信息。
        :param gene_id: 基因ID
        :param gff_df: GFF DataFrame
        :return: 包含基因位置信息的字典（染色体、起始位置、结束位置、链方向）
        """
        try:
            # 快速查找基因ID
            gene_entry = gff_df[gff_df['gene_id'] == gene_id]
            if not gene_entry.empty:
                entry = gene_entry.iloc[0]
                return {
                    'chr': entry['seqid'],
                    'start': entry['start'],
                    'end': entry['end'],
                    'strand': entry['strand']
                }
            return None
        except Exception as e:
            self.logger.warning(f"Error finding position for gene {gene_id}: {str(e)}")
            return None

    def calculate_distance(self, snp_pos: int, gene_start: int, gene_end: int) -> int:
        """
        计算SNP到基因的最小距离。
        :param snp_pos: SNP的位置
        :param gene_start: 基因的起始位置
        :param gene_end: 基因的结束位置
        :return: SNP到基因的最小距离
        """
        if gene_start <= snp_pos <= gene_end:
            return 0
        return min(abs(snp_pos - gene_start), abs(snp_pos - gene_end))

    def analyze_linkage(self, snp_df: pd.DataFrame, gene_df: pd.DataFrame,
                        gff_df: pd.DataFrame) -> pd.DataFrame:
        """
        分析SNP和基因之间的连锁关系，基于距离阈值筛选潜在的连锁对。
        :param snp_df: SNP DataFrame
        :param gene_df: 基因 DataFrame
        :param gff_df: GFF DataFrame
        :return: 包含连锁分析结果的DataFrame
        """
        results = []
        total_combinations = len(snp_df) * len(gene_df)
        processed = 0

        self.logger.info("Starting linkage analysis...")

        # 按染色体分组以提高效率
        snp_grouped = snp_df.groupby('CHROM')
        gene_grouped = gene_df.groupby('RAP_ID')

        for chrom, snp_group in snp_grouped:
            gene_entries = gene_grouped.filter(lambda x: x['RAP_ID'].iloc[0] in gff_df['gene_id'].values)
            for _, snp in snp_group.iterrows():
                for _, gene in gene_entries.iterrows():
                    processed += 1
                    if processed % 1000 == 0:
                        progress = (processed / total_combinations) * 100
                        self.logger.info(f"Progress: {progress:.2f}%")

                    gene_pos = self.find_gene_position(gene['RAP_ID'], gff_df)
                    if gene_pos is None:
                        continue

                    if snp['CHROM'] == gene_pos['chr']:
                        distance = self.calculate_distance(snp['POS'],
                                                           gene_pos['start'],
                                                           gene_pos['end'])
                        if distance <= self.threshold:
                            results.append({
                                'SNP_ID': snp['SNP_ID'],
                                'SNP_Chr': snp['CHROM'],
                                'SNP_Pos': snp['POS'],
                                'Gene_Name': gene['Gene_Name'],
                                'Gene_ID': gene['RAP_ID'],
                                'Gene_Chr': gene_pos['chr'],
                                'Gene_Start': gene_pos['start'],
                                'Gene_End': gene_pos['end'],
                                'Distance': distance
                            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('Distance')

        self.logger.info(f"Analysis complete. Found {len(results_df)} potential linkages")
        return results_df


def main():
    """
    主函数，负责协调数据加载、分析和结果保存。
    """
    # 创建输出目录
    phenotype = 'Heading_date'
    output_dir = Path("/data1/wangchengrui/hznd_data/shap_info")
    output_dir.mkdir(exist_ok=True)

    # 创建分析器实例
    analyzer = GeneSnpLinkageAnalyzer(threshold=1000000)

    try:
        # 加载数据
        snp_df = analyzer.load_snp_data(f"/data1/wangchengrui/hznd_data/shap_info/{phenotype}_shap_info.csv")
        gene_df = analyzer.load_gene_data(f"/data1/wangchengrui/hznd_data/shap_info/{phenotype}.txt")
        gff_df = analyzer.load_gff_data("/data1/wangchengrui/refs/osa/osa_IRGSP_1.annotation.gff")

        # 进行分析
        results_df = analyzer.analyze_linkage(snp_df, gene_df, gff_df)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"linkage_results_{phenotype}_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)

        # 生成统计摘要
        summary_file = output_dir / f"analysis_summary_{phenotype}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("Linkage Analysis Summary\n")
            f.write("=======================\n")
            f.write(f"Total SNPs analyzed: {len(snp_df)}\n")
            f.write(f"Total genes analyzed: {len(gene_df)}\n")
            f.write(f"Total linkages found: {len(results_df)}\n")
            f.write(f"Distance threshold: {analyzer.threshold:,} bp\n")

            if not results_df.empty:
                f.write("\nTop 10 closest linkages:\n")
                for _, row in results_df.head(10).iterrows():
                    f.write(f"\nSNP {row['SNP_ID']} - Gene {row['Gene_Name']}\n")
                    f.write(f"Distance: {row['Distance']:,} bp\n")

        analyzer.logger.info(f"Results saved to {output_file}")
        analyzer.logger.info(f"Summary saved to {summary_file}")

    except Exception as e:
        analyzer.logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()