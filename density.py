import gzip
import os
from concurrent.futures import ProcessPoolExecutor
import csv
import numpy as np

def count_total_lines(file_path):
    """统计 .vcf.gz 文件的总行数"""
    total_lines = 0
    with gzip.open(file_path, 'rt') as f:
        for _ in f:
            total_lines += 1
    return total_lines

def extract_metadata_and_header(file_path):
    """
    提取元信息行（## 开头）和列标题行（#CHROM 开头）
    返回值：
        metadata_lines: 元信息行列表
        header_line: 列标题行内容
    """
    metadata_lines = []
    header_line = None
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("##"):
                metadata_lines.append(line)
            elif line.startswith("#CHROM"):
                header_line = line
                break
            else:
                break
    return metadata_lines, header_line

def read_phenotype_ids(phenotype_file):
    """
    从 phenotype 文件中读取 ID 列
    返回值：
        ids: 样本 ID 列表
    """
    ids = []
    with open(phenotype_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['ID']:
                ids.append(row['ID'])
    return ids

def process_chunk(chunk_start, chunk_end, file_path):
    """处理文件的一个分块，返回所有非注释行"""
    selected_lines = []
    with gzip.open(file_path, 'rt') as f:
        # 跳过到 chunk_start 行
        for _ in range(chunk_start):
            next(f)
        # 处理当前分块
        for line_number in range(chunk_start, chunk_end):
            try:
                line = next(f)
                if not line.startswith("#"):
                    selected_lines.append(line)
            except StopIteration:
                break
    return selected_lines

def main(vcf_file, output_dir, target_rows_list, phenotype_file):
    # 1. 计算文件总行数
    total_lines = count_total_lines(vcf_file)
    print(f"Total lines: {total_lines}")

    # 2. 提取元信息行和列标题行
    metadata_lines, header_line = extract_metadata_and_header(vcf_file)
    if header_line is None:
        raise ValueError("Missing #CHROM header line in the input VCF file.")
    print(f"Metadata lines: {len(metadata_lines)}")
    print(f"Header line: {header_line.strip()}")

    # 3. 读取 phenotype 文件中的样本 ID
    sample_ids = read_phenotype_ids(phenotype_file)
    print(f"Sample IDs loaded: {len(sample_ids)}")

    # 4. 修改表头行中的样本名
    header_parts = header_line.strip().split('\t')
    original_samples = header_parts[9:]  # 原始样本名
    if len(sample_ids) != len(original_samples):
        raise ValueError(f"Sample count mismatch: VCF has {len(original_samples)} samples, phenotype file has {len(sample_ids)} IDs.")
    header_parts[9:] = sample_ids  # 替换为新的样本名
    new_header_line = '\t'.join(header_parts) + '\n'
    print(f"Updated header line: {new_header_line.strip()}")

    # 5. 多进程处理文件，提取所有 SNP 数据
    num_processes = min(os.cpu_count(), 4)  # 限制最大进程数为 4
    chunk_size = total_lines // num_processes

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(num_processes):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size if i < num_processes - 1 else total_lines
            future = executor.submit(process_chunk, chunk_start, chunk_end, vcf_file)
            futures.append(future)

        # 收集结果
        for future in futures:
            results.extend(future.result())

    # 按染色体排序所有 SNP 数据
    results = sorted(results, key=lambda x: x.split('\t')[0])

    # 6. 从最大数据集逐步生成更小的数据集
    subsets = {}
    current_subset = results  # 当前数据集初始化为最大数据集
    for target_rows in sorted(target_rows_list, reverse=True):  # 从大到小生成
        if len(current_subset) <= target_rows:
            subsets[target_rows] = current_subset  # 如果当前数据集小于目标数量，直接使用
        else:
            # 使用 numpy.linspace 精确选择目标数量的 SNP
            indices = np.linspace(0, len(current_subset) - 1, target_rows, dtype=int)
            subsets[target_rows] = [current_subset[i] for i in indices]
        current_subset = subsets[target_rows]  # 更新当前数据集为新生成的子集

    # 7. 写入多个输出文件
    for target_rows in target_rows_list:
        output_file = os.path.join(output_dir, f"rice.snp.{target_rows}.vcf.gz")
        with gzip.open(output_file, 'wt') as f:
            # 写入元信息行
            for line in metadata_lines:
                f.write(line)
            # 写入更新后的列标题行
            f.write(new_header_line)
            # 写入选中的数据行
            for line in subsets[target_rows]:
                f.write(line)

        print(f"New VCF file created: {output_file} with {len(subsets[target_rows])} SNPs")

if __name__ == "__main__":
    target_rows_list = [
        100, 200, 500, 800, 1000, 1500, 2000, 3000,
        5000, 8000, 10000, 12000, 15000, 18000,
        20000, 22000, 25000, 28000, 30000
    ]  # 目标保留的 SNP 数量列表
    vcf_file = "/data1/wangchengrui/rice1w/LD_NAM_Magic.vcf.gz"  # 输入的 VCF 压缩文件路径
    output_dir = "/data1/wangchengrui/rice1w/density"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    phenotype_file = "/data1/wangchengrui/rice1w/Data_shanghai.csv"
    main(vcf_file, output_dir, target_rows_list, phenotype_file)