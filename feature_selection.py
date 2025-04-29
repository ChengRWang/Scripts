"""
脚本功能：
1. 从SnpEff注释文件中计算SNP的优先级，基于不同策略（如区域优先级、影响优先级等）。
2. 从VCF文件中筛选优先级最高的SNP，并生成编码矩阵。
3. 将编码矩阵与样本信息和SNP ID结合，生成最终的转置CSV文件。
4. 提供灵活的参数配置，支持用户自定义输入文件路径和输出目录。

主要功能模块：
- SNPSelector 类：负责处理SnpEff注释文件、VCF文件，并计算SNP优先级。
- add_snp_and_sample_info 函数：将VCF文件中的样本信息和SNP ID添加到编码矩阵中。
- transpose_csv 函数：转置CSV文件，使每一行为一个样本，列包括IID和SNP名。
"""

import gzip
import os
import numpy as np
import pandas as pd
import json

class SNPSelector:
    def __init__(self, vcf_file, snpeff_file, ldsnp_file, output_dir, total_snps=10000):
        """
        初始化SNP选择器
        :param vcf_file: 输入的VCF文件路径
        :param snpeff_file: SnpEff注释文件路径
        :param ldsnp_file: LD修剪后的SNP列表文件路径
        :param output_dir: 输出目录
        :param total_snps: 需要选择的SNP总数
        """
        os.makedirs(output_dir, exist_ok=True)
        self.vcf_file = vcf_file
        self.snpeff_file = snpeff_file
        self.ldsnp_list = self._read_ldsnp_list(ldsnp_file)
        self.output_dir = output_dir
        self.total_snps = total_snps

    def _open_file(self, file_path):
        """根据文件扩展名选择打开方式"""
        return gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')

    def _read_ldsnp_list(self, ldsnp_file):
        """读取LD修剪后的SNP列表"""
        with open(ldsnp_file, 'r') as f:
            return set(line.strip() for line in f)

    def _calculate_priority(self, info_fields, method):
        """
        计算SNP优先级
        :param info_fields: INFO字段字典
        :param method: 优先级计算方法（region, impact, combined）
        :return: 优先级分数
        """
        impact_priority = {'HIGH': 4, 'MODERATE': 3, 'LOW': 2, 'MODIFIER': 1}
        mutation_priority = {
            'missense_variant': 'EXON',
            'synonymous_variant': 'EXON',
            'splice_region_variant': 'SPLICE_SITE_REGION',
            'intron_variant': 'INTRON',
            'upstream_gene_variant': 'UPSTREAM',
            'downstream_gene_variant': 'DOWNSTREAM',
            'frameshift_variant': 'EXON',
            'start_lost': 'GENE',
            'stop_gained': 'GENE',
            'stop_lost': 'GENE',
            'non_coding_transcript_exon_variant': 'EXON',
            'protein_altering_variant': 'EXON',
            'intergenic_variant': 'INTERGENIC',
            '3_prime_UTR_truncation': 'UTR_3_PRIME',
            '3_prime_UTR_variant': 'UTR_3_PRIME',
            '5_prime_UTR_premature_start_codon_gain_variant': 'UTR_5_PRIME',
            '5_prime_UTR_truncation': 'UTR_5_PRIME',
            '5_prime_UTR_variant': 'UTR_5_PRIME',
            'bidirectional_gene_fusion': 'GENE',
            'conservative_inframe_deletion': 'EXON',
            'conservative_inframe_insertion': 'EXON',
            'disruptive_inframe_deletion': 'EXON',
            'disruptive_inframe_insertion': 'EXON',
            'gene_fusion': 'GENE',
            'initiator_codon_variant': 'GENE',
            'exon_loss_variant': 'EXON',
            'transcript_ablation': 'TRANSCRIPT',
            'SPLICE_SITE_ACCEPTOR': 'SPLICE_SITE_ACCEPTOR',
            'SPLICE_SITE_DONOR': 'SPLICE_SITE_DONOR',
            'SPLICE_SITE_REGION': 'SPLICE_SITE_REGION',
            'TRANSCRIPT': 'TRANSCRIPT',
            'UTR_3_PRIME': 'UTR_3_PRIME',
            'UTR_5_PRIME': 'UTR_5_PRIME'
        }
        region_priority = {
            'DOWNSTREAM': 1,  # 下游区域，优先级较低
            'EXON': 5,        # 外显子，优先级较高
            'GENE': 5,        # 基因区域，优先级较高
            'INTERGENIC': 1,  # 基因间区域，优先级最低
            'INTRON': 3,      # 内含子，优先级中等
            'SPLICE_SITE_ACCEPTOR': 4,  # 剪接接受位点，优先级较高
            'SPLICE_SITE_DONOR': 4,     # 剪接供体位点，优先级较高
            'SPLICE_SITE_REGION': 4,    # 剪接区域，优先级较高
            'TRANSCRIPT': 3,            # 转录区域，优先级中等
            'UPSTREAM': 1,              # 上游区域，优先级较低
            'UTR_3_PRIME': 3,           # 3'非编码区，优先级中等
            'UTR_5_PRIME': 3            # 5'非编码区，优先级中等
        }

        ann_info = info_fields.get('ANN', '').split('|') if 'ANN' in info_fields else []
        impact = ann_info[2] if len(ann_info) > 2 else 'UNKNOWN'
        features = ann_info[1] if len(ann_info) > 4 else 'UNKNOWN'
        feature_list = features.split('&')
        maf = float(info_fields.get('AF', 0.5)) if 'AF' in info_fields else 0.5

        if method == 'region':
            feature_score = sum(region_priority.get(f, 0) for f in feature_list) / len(
                feature_list) if feature_list != ['UNKNOWN'] else 0
            return feature_score * 0.7 + maf * 0.3
        elif method == 'impact':
            return impact_priority.get(impact, 0) * 0.7 + maf * 0.3
        elif method == 'combined':
            feature_score = sum(region_priority.get(f, 0) for f in feature_list) / len(
                feature_list) if feature_list != ['UNKNOWN'] else 0
            impact_score = impact_priority.get(impact, 0)
            return (feature_score + impact_score) * 0.5 + maf * 0.3

    def process_snpeff(self):
        """遍历SnpEff文件并计算优先级"""
        priorities = {
            'region': {}, 'impact': {}, 'combined': {},
            'random': {}, 'coding': {}, 'regulatory': {}
        }
        coding_region_keys = [
            'missense_variant', 'synonymous_variant', 'frameshift_variant',
            'non_coding_transcript_exon_variant', 'protein_altering_variant',
            'conservative_inframe_deletion', 'conservative_inframe_insertion',
            'disruptive_inframe_deletion', 'disruptive_inframe_insertion',
            'exon_loss_variant', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'
        ]
        regulatory_region_keys = [
            'splice_region_variant', 'intron_variant', 'upstream_gene_variant',
            'downstream_gene_variant', 'start_lost', 'stop_gained', 'stop_lost',
            'intergenic_variant', '3_prime_UTR_truncation', '3_prime_UTR_variant',
            '5_prime_UTR_premature_start_codon_gain_variant', '5_prime_UTR_truncation',
            '5_prime_UTR_variant', 'bidirectional_gene_fusion', 'gene_fusion',
            'initiator_codon_variant', 'transcript_ablation', 'TRANSCRIPT',
            'UTR_3_PRIME', 'UTR_5_PRIME'
        ]
        all_ld_snps = list(self.ldsnp_list)  # 用于随机选择
        with self._open_file(self.snpeff_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                snp_id = fields[2] if fields[2] != '.' else f"{fields[0]}_{fields[1]}"
                if snp_id not in self.ldsnp_list:
                    continue
                info_fields = dict(
                    item.split('=') if '=' in item else (item, True)
                    for item in fields[7].split(';')
                )
                feature = info_fields.get('ANN', '').split('|')[1] if 'ANN' in info_fields else 'UNKNOWN'
                for method in ['region', 'impact', 'combined']:
                    priorities[method][snp_id] = self._calculate_priority(info_fields, method)
                # 策略：挑选编码区优先级最高的SNP
                if set(feature.split('&')).intersection(coding_region_keys):
                    priorities['coding'][snp_id] = self._calculate_priority(info_fields, 'combined')
                # 策略：挑选调控区优先级最高的SNP
                if set(feature.split('&')).intersection(regulatory_region_keys):
                    priorities['regulatory'][snp_id] = self._calculate_priority(info_fields, 'combined')

        # 策略：随机挑选
        random_selected = np.random.choice(all_ld_snps, self.total_snps, replace=False)
        for snp_id in random_selected:
            priorities['random'][snp_id] = 1  # 随机选择无优先级，默认设置为1

        # 将优先级字典写入文件
        output_file = f'{self.output_dir}/priorities_output.json'  # 输出文件路径
        with open(output_file, 'w') as f:
            json.dump(priorities, f, indent=4)
        return priorities

    def process_vcf(self, priorities):
        """遍历VCF文件并选取优先级最高的SNP"""
        selected_snps = {
            'region': [], 'impact': [], 'combined': [],
            'random': [], 'coding': [], 'regulatory': []
        }
        encoded_data = {method: [] for method in selected_snps.keys()}
        with self._open_file(self.vcf_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                snp_id = fields[2] if fields[2] != '.' else f"{fields[0]}_{fields[1]}"
                for method in selected_snps.keys():
                    if snp_id in priorities[method]:
                        selected_snps[method].append((priorities[method][snp_id], snp_id))
                        genotype_row = []
                        for gt in fields[9:]:
                            alleles = gt.split(':')[0].split('/')
                            genotype_code = sum(
                                int(allele) if allele.isdigit() else -1 for allele in alleles
                            ) if '.' not in gt else -1
                            genotype_row.append(genotype_code)
                        encoded_data[method].append((snp_id, genotype_row))

        # 筛选优先级最高的 SNP
        for method in selected_snps.keys():
            selected_snps[method] = sorted(selected_snps[method], key=lambda x: -x[0])[:self.total_snps]
            encoded_data[method] = [
                genotype for snp_id, genotype in encoded_data[method]
                if snp_id in {snp[1] for snp in selected_snps[method]}
            ]

        return selected_snps, encoded_data

    def save_results(self, selected_snps, encoded_data):
        """保存结果到CSV文件"""
        for method in selected_snps.keys():
            snp_ids = [snp[1] for snp in selected_snps[method]]
            # 保存SNP列表
            snp_file = os.path.join(self.output_dir, f'{method}_selected_snps.csv')
            pd.DataFrame({'SNP_ID': snp_ids}).to_csv(snp_file, index=False)
            print(f"SNP列表已保存到 {snp_file}")
            # 保存加性编码矩阵
            encoded_file = os.path.join(self.output_dir, f'{method}_encoded_matrix.csv')
            np.savetxt(encoded_file, np.array(encoded_data[method]), delimiter=',', fmt='%d')
            print(f"编码矩阵已保存到 {encoded_file}")

    def run(self):
        """主流程"""
        # 检查是否已经存在之前计算的优先级文件
        priority_file = f'{self.output_dir}/priorities_output.json'
        if os.path.exists(priority_file):
            print("检测到已存在的优先级文件，正在读取...")
            with open(priority_file, 'r') as f:
                priorities = json.load(f)
            print("优先级文件读取完成，开始处理VCF文件...")
        else:
            print("没有检测到优先级文件，开始处理SnpEff文件...")
            priorities = self.process_snpeff()
            print("优先级计算完成，开始处理VCF文件...")

        # 接下来的处理步骤保持不变
        selected_snps, encoded_data = self.process_vcf(priorities)
        print("VCF文件处理完成，开始保存结果...")
        self.save_results(selected_snps, encoded_data)
        print("所有结果已保存完成。")


def add_snp_and_sample_info(vcf_file, encoded_file, snp_file, output_file):
    """
    将VCF文件中的样本信息和SNP_ID添加到编码矩阵中
    :param vcf_file: 原始VCF文件路径
    :param encoded_file: 编码矩阵文件路径
    :param snp_file: 选择的SNP列表文件路径
    :param output_file: 带SNP_ID和样本信息的新文件路径
    """
    # 获取样本信息
    with gzip.open(vcf_file, 'rt') if vcf_file.endswith('.gz') else open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#CHROM'):
                # VCF文件样本ID从第10列开始
                sample_ids = line.strip().split('\t')[9:]
                break

    # 读取现有编码矩阵
    encoded_matrix = pd.read_csv(encoded_file, header=None)
    # 读取SNP_ID列表
    snp_ids = pd.read_csv(snp_file)['SNP_ID']

    # 检查编码矩阵的行数和SNP_ID列表的长度是否一致
    if len(encoded_matrix) != len(snp_ids):
        raise ValueError("编码矩阵的行数与SNP_ID列表的长度不一致！")

    # 将SNP_ID添加为第一列
    encoded_matrix.insert(0, 'SNP_ID', snp_ids)
    # 添加列标题（样本ID），注意样本ID从第二列开始
    encoded_matrix.columns = ['SNP_ID'] + sample_ids

    # 保存结果到新文件
    encoded_matrix.to_csv(output_file, index=False)
    print(f"带SNP_ID和样本信息的文件已保存到 {output_file}")


def transpose_csv(input_csv, output_csv):
    """
    转置CSV文件，每一行为一个样本，列包括IID和SNP名
    :param input_csv: 输入的CSV文件路径
    :param output_csv: 转置后的CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    # 转置操作
    transposed_df = df.set_index('SNP_ID').transpose()
    # 添加IID列
    transposed_df.insert(0, 'IID', transposed_df.index)
    # 保存转置后的CSV
    transposed_df.to_csv(output_csv, index=False)
    print(f"转置后的CSV已保存到 {output_csv}")


# 主程序入口
def main():
    vcf_file = 'data1/wangchengrui/data/hznd/rice4k_geno_no_del.vcf.gz'
    snpeff_file = 'data2/users/luojinjing/Tangping/nodel_variants.ann.vcf.gz'
    ldsnp_file = 'data1/wangchengrui/rice4k_LD/pruned_data.prune.in'
    output_dir = 'data1/wangchengrui/rice4k_LD/output/'
    total_snps = 10000  # 可调整参数

    selector = SNPSelector(vcf_file, snpeff_file, ldsnp_file, output_dir, total_snps)
    selector.run()

    for method in ['region', 'impact', 'combined', 'coding', 'regulatory']:
        encoded_file = f'{output_dir}{method}_encoded_matrix.csv'
        snp_file = f'{output_dir}{method}_selected_snps.csv'
        output_file = f'{output_dir}{method}_encoded_matrix_with_samples.csv'
        trans = f'{output_dir}{method}_trans.csv'

        add_snp_and_sample_info(vcf_file, encoded_file, snp_file, output_file)
        transpose_csv(output_file, trans)


if __name__ == '__main__':
    main()