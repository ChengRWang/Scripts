import gzip
import os
from concurrent.futures import ProcessPoolExecutor

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

def process_chunk(chunk_start, chunk_end, file_path, step):
    """处理文件的一个分块"""
    selected_lines = []
    with gzip.open(file_path, 'rt') as f:
        # 跳过到 chunk_start 行
        for _ in range(chunk_start):
            next(f)
        # 处理当前分块
        for line_number in range(chunk_start, chunk_end):
            try:
                line = next(f)
                if not line.startswith("#") and line_number % step == 0:
                    selected_lines.append(line)
            except StopIteration:
                break
    return selected_lines

def main(vcf_file, output_file, target_rows):
    # 1. 计算文件总行数
    total_lines = count_total_lines(vcf_file)
    print(f"Total lines: {total_lines}")

    # 2. 提取元信息行和列标题行
    metadata_lines, header_line = extract_metadata_and_header(vcf_file)
    if header_line is None:
        raise ValueError("Missing #CHROM header line in the input VCF file.")
    print(f"Metadata lines: {len(metadata_lines)}")
    print(f"Header line: {header_line.strip()}")

    # 3. 计算步长
    data_lines = total_lines - len(metadata_lines) - 1  # 减去 #CHROM 行
    step = max(1, data_lines // (target_rows - len(metadata_lines) - 1))
    print(f"Step size: {step}")

    # 4. 多进程处理文件
    num_processes = min(os.cpu_count(), 4)  # 限制最大进程数为 4
    chunk_size = total_lines // num_processes

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(num_processes):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size if i < num_processes - 1 else total_lines
            future = executor.submit(process_chunk, chunk_start, chunk_end, vcf_file, step)
            futures.append(future)

        # 收集结果
        for future in futures:
            results.extend(future.result())

    # 5. 写入新文件
    with gzip.open(output_file, 'wt') as f:
        # 写入元信息行
        for line in metadata_lines:
            f.write(line)
        # 写入列标题行
        f.write(header_line)
        # 写入选中的数据行
        for line in sorted(results, key=lambda x: x.split('\t')[0]):  # 按染色体排序
            f.write(line)

    print(f"New VCF file created: {output_file}")


if __name__ == "__main__":
    target_rows = 10000  # 目标保留的行数
    vcf_file = "/data1/wangchengrui/data/hznd/rice4k_eQTL.vcf.gz"  # 输入的 VCF 压缩文件路径
    output_file = f"/data1/wangchengrui/final_results/eqtl/rice4k.eqtl.{target_rows}.vcf.gz"  # 输出的 VCF 压缩文件路径
    main(vcf_file, output_file, target_rows)
