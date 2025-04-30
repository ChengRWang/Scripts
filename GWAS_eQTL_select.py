import os
import gzip
import pandas as pd

# 定义路径
gwas_dir = "/data1/wangchengrui/final_results/yuanshigwas"
eqtl_cis_file = "/path/to/Rice_all_eQTL_cis.txt"
eqtl_trans_file = "/path/to/Rice_all_eQTL_trans.txt"
vcf_file = "/path/to/input.vcf.gz"
output_dir = "/path/to/output"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取eQTL文件
eqtl_cis = pd.read_csv(eqtl_cis_file, sep="\t")
eqtl_trans = pd.read_csv(eqtl_trans_file, sep="\t")

# 合并cis和trans eQTL数据
eqtl_all = pd.concat([eqtl_cis, eqtl_trans])

# 获取所有GWAS文件
gwas_files = [f for f in os.listdir(gwas_dir) if f.startswith("mlm_output_") and f.endswith(".txt")]

# 存储每个表型对应的SNP列表
trait_snp_dict = {}

for gwas_file in gwas_files:
    # 读取GWAS结果文件
    gwas_path = os.path.join(gwas_dir, gwas_file)
    gwas_data = pd.read_csv(gwas_path, sep="\t")
    
    # 提取表型名称
    trait_name = gwas_file.split("_")[2].split(".")[0]
    
    # 按Effect绝对值排序，筛选前2万个SNP
    top_snps = gwas_data.groupby("Marker").first().reset_index()
    top_snps["Effect_Abs"] = top_snps["Effect"].abs()
    top_snps = top_snps.sort_values(by="Effect_Abs", ascending=False).head(20000)
    
    # 获取SNP列表
    snp_list = top_snps["Marker"].tolist()
    
    # 匹配eQTL中的SNP
    matched_snps = eqtl_all[eqtl_all["SNP"].isin(snp_list)]["SNP"].unique()
    
    # 如果匹配的SNP不足1万，从GWAS中补充
    if len(matched_snps) < 10000:
        remaining_count = 10000 - len(matched_snps)
        # 从未匹配的GWAS SNP中按Effect排序选择补充
        unmatched_snps = top_snps[~top_snps["Marker"].isin(matched_snps)]
        supplemental_snps = unmatched_snps.head(remaining_count)["Marker"].tolist()
        matched_snps = list(matched_snps) + supplemental_snps
    
    # 确保最终SNP数量不超过1万
    matched_snps = matched_snps[:10000]
    
    # 存储到字典中
    trait_snp_dict[trait_name] = set(matched_snps)

# 打开VCF文件并遍历
with gzip.open(vcf_file, "rt") as vcf_in:
    # 初始化输出文件句柄
    output_files = {}
    header_lines = []
    
    for line in vcf_in:
        # 保存VCF头部信息
        if line.startswith("#"):
            header_lines.append(line)
            continue
        
        # 解析VCF记录
        fields = line.strip().split("\t")
        chrom, pos, snp_id = fields[0], fields[1], fields[2]
        
        # 检查SNP是否在某个表型的SNP列表中
        for trait_name, snp_set in trait_snp_dict.items():
            if snp_id in snp_set:
                # 如果第一次写入，先写入头部信息
                if trait_name not in output_files:
                    output_vcf = os.path.join(output_dir, f"{trait_name}_filtered.vcf")
                    output_files[trait_name] = open(output_vcf, "w")
                    output_files[trait_name].writelines(header_lines)
                
                # 写入选中的SNP记录
                output_files[trait_name].write(line)
    
    # 关闭所有输出文件
    for file_handle in output_files.values():
        file_handle.close()

print("All tasks completed.")