import pandas as pd
import numpy as np
import re
import os
import logging

"""根据gwas结果、以及gff文件，找到显著性位点集中的基因区域"""

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def extract_gene_id(attributes):
    """从attributes字符串中提取GeneID，并去掉后缀部分"""
    try:
        if not isinstance(attributes, str) or not attributes:
            return None  # 如果输入无效，返回None

        # 支持大小写不敏感的匹配
        match = re.search(r'geneID=([^;]+)', attributes, re.IGNORECASE)
        if match:
            gene_id = match.group(1)  # 获取geneID
            return re.sub(r'-\d+$', '', gene_id)  # 去掉后缀部分
        return None  # 如果未找到geneID，返回None
    except Exception as e:
        logging.error(f"提取基因ID时出错: {str(e)}")
        return None


def read_gwas_results(file_path):
    """读取GWAS结果文件并计算-log10(p)"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 尝试多种分隔符以提高兼容性
        df = pd.read_csv(file_path, sep=None, engine='python')
        required_cols = ['Marker', 'Chr', 'Pos', 'p']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        df = df.dropna(subset=['p'])
        df['log10p'] = -np.log10(df['p'].clip(lower=np.finfo(float).tiny))  # 防止对0取对数
        df['Chr'] = pd.to_numeric(df['Chr'], errors='coerce')
        df['Pos'] = pd.to_numeric(df['Pos'], errors='coerce')
        df = df.dropna(subset=['Chr', 'Pos'])

        # 检查log10p是否合理
        if (df['log10p'] < 0).any():
            raise ValueError("Found negative values in log10p column. Please check the p-values.")

        return df

    except Exception as e:
        logging.error(f"Error reading GWAS results: {str(e)}")
        raise


def read_gene_positions(gff_file):
    """读取基因位置信息"""
    try:
        # 检查文件是否存在
        if not os.path.exists(gff_file):
            raise FileNotFoundError(f"File not found: {gff_file}")

        cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
        gff = pd.read_csv(gff_file, sep='\t', comment='#', names=cols)
        genes = gff[gff['type'] == 'mRNA'].copy()

        # 提取染色体编号，支持非数字形式
        genes['chr'] = genes['seqid'].str.replace('chr', '', case=False).str.extract('(\d+|X|Y)').astype(str)

        # 提取geneID并去重
        genes['gene_id'] = genes['attributes'].apply(extract_gene_id)
        genes = genes.dropna(subset=['gene_id'])  # 过滤掉无效的geneID

        genes['start'] = pd.to_numeric(genes['start'], errors='coerce')
        genes['end'] = pd.to_numeric(genes['end'], errors='coerce')
        genes = genes.dropna(subset=['chr', 'start', 'end'])

        return genes

    except Exception as e:
        logging.error(f"Error reading GFF file: {str(e)}")
        raise


def analyze_significant_regions(gwas_df, genes_df, log10p_threshold=3, flanking_distance=2000):
    """分析显著性区域内的SNP"""
    significant_regions = []
    gene_order = []  # 用于存储基因ID的顺序

    significant_snps = gwas_df[gwas_df['log10p'] >= log10p_threshold].sort_values(['Chr', 'Pos'])

    for _, snp in significant_snps.iterrows():
        genes_in_region = genes_df[
            (genes_df['chr'] == snp['Chr']) &
            (genes_df['start'] - flanking_distance <= snp['Pos']) &
            (genes_df['end'] + flanking_distance >= snp['Pos'])
        ]

        logging.info(f"Analyzing SNP: {snp['Marker']} at position {snp['Pos']} on chromosome {snp['Chr']}")
        logging.info(f"Found genes in region: {genes_in_region[['gene_id', 'start', 'end']]}\n")

        for _, gene in genes_in_region.iterrows():
            # 检查是否已经处理过该基因
            if any(r['gene_id'] == gene['gene_id'] for r in significant_regions):
                continue

            # 记录基因ID的顺序
            if gene['gene_id'] not in gene_order:
                gene_order.append(gene['gene_id'])

            region_start = gene['start'] - flanking_distance
            region_end = gene['end'] + flanking_distance

            region_snps = gwas_df[
                (gwas_df['Chr'] == snp['Chr']) &
                (gwas_df['Pos'] >= region_start) &
                (gwas_df['Pos'] <= region_end) &
                (gwas_df['log10p'] >= log10p_threshold)
            ]

            if region_snps.empty:
                continue  # 如果区域内没有SNP，跳过

            peak_snp = region_snps.loc[region_snps['log10p'].idxmax()]

            region_info = {
                'chr': int(snp['Chr']),
                'gene_id': gene['gene_id'],
                'gene_start': int(gene['start']),
                'gene_end': int(gene['end']),
                'region_start': int(region_start),
                'region_end': int(region_end),
                'significant_snp_count': len(region_snps),
                'peak_snp': peak_snp['Marker'],
                'peak_snp_pos': int(peak_snp['Pos']),
                'peak_log10p': peak_snp['log10p'],
                'peak_p': peak_snp['p'],
                'snp_list': region_snps['Marker'].tolist()
            }
            significant_regions.append(region_info)

    significant_regions.sort(key=lambda x: x['significant_snp_count'], reverse=True)

    return significant_regions, gene_order  # 返回基因顺序


def main():
    try:
        phenotype = 'Plant_height'
        gwas_file = f'/data1/wangchengrui/hznd_data/gwas_input/input_ori/mlm_output_{phenotype}.manht_input'
        gff_file = f'/data1/wangchengrui/refs/osa/osa_IRGSP_1.annotation.gff'
        output_file = f'/data1/wangchengrui/hznd_data/gwas_input/result_ori/{phenotype}_peak_analysis_results.txt'
        gene_list_file = f'/data1/wangchengrui/hznd_data/gwas_input/result_ori/{phenotype}_gene_list.txt'

        log10p_threshold = 6
        flanking_distance = 2000

        logging.info("Reading GWAS results...")
        gwas_df = read_gwas_results(gwas_file)

        logging.info("Reading gene annotations...")
        genes_df = read_gene_positions(gff_file)

        logging.info("Analyzing significant regions...")
        significant_regions, gene_list = analyze_significant_regions(
            gwas_df,
            genes_df,
            log10p_threshold=log10p_threshold,
            flanking_distance=flanking_distance
        )

        logging.info(f"\nWriting results to {output_file}...")

        with open(output_file, 'w') as f, open(gene_list_file, 'w') as f2:
            f2.write("Unique Gene List\n")
            f.write("GWAS Gene Region Analysis Results\n")
            f.write("================================\n\n")

            f.write("Analysis Parameters:\n")
            f.write(f"- Significance threshold: -log10(p) > {log10p_threshold}\n")
            f.write(f"- Gene flanking distance: {flanking_distance}bp\n\n")

            f.write(f"Found {len(significant_regions)} significant gene regions\n\n")

            for i, region in enumerate(significant_regions, 1):
                gene_id = region['gene_id']
                output = f"\nSignificant Region {i}:\n"
                output += f"Chromosome: {region['chr']}\n"
                output += f"GeneID: {region['gene_id']}\n"
                output += f"Gene Position: {region['gene_start']}-{region['gene_end']}\n"
                output += f"Region (with flanking): {region['region_start']}-{region['region_end']}\n"
                output += f"Number of Significant SNPs: {region['significant_snp_count']}\n"
                output += f"Peak SNP: {region['peak_snp']}\n"
                output += f"Peak SNP Position: {region['peak_snp_pos']}\n"
                output += f"Peak Significance: -log10(p) = {region['peak_log10p']:.2f} (p = {region['peak_p']:.2e})\n"
                output += f"Significant SNPs in region: {', '.join(region['snp_list'])}\n"
                output += "-" * 80 + "\n"

                logging.info(output)
                f.write(output)
                f2.write(f'{gene_id}\n')

        logging.info(f"\nAnalysis completed. Results saved to {output_file}")

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()