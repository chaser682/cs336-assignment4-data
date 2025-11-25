import random
import os

def read_clean_lines(file_path, min_length=20):
    """
    读取文本文件，返回清洗后的行列表。
    参数:
        file_path: 文件路径
        min_length: 过滤掉长度小于该值的行（太短的文本通常是噪声）
    """
    lines = []
    print(f"正在读取: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # 去除首尾空白
            text = line.strip()
            
            # 简单清洗：确保 FastText 能够正确识别（FastText 默认以换行符分隔样本）
            # 将中间可能存在的制表符等转为空格
            text = text.replace('\n', ' ').replace('\r', '')
            
            # 过滤掉过短的行
            if len(text) > min_length:
                lines.append(text)
                
    print(f"  - 读取到 {len(lines)} 行有效数据")
    return lines

def build_balanced_dataset():
    # === 配置路径 ===
    wiki_path = "var/high_quality_wiki_reference.txt"  # 正例
    cc_path = "var/low_quality_cc.txt"                 # 负例
    output_path = "var/quality_train.txt"              # 最终生成的训练文件

    # 1. 读取数据到内存
    wiki_lines = read_clean_lines(wiki_path)
    cc_lines = read_clean_lines(cc_path)

    # 2. 检查数据量
    if not wiki_lines or not cc_lines:
        print("停止：其中一个数据集为空，无法构建训练集。")
        return

    # 3. 计算平衡数量 (Downsampling)
    # 取两者中的较小值，保证正负样本 1:1
    min_count = min(len(wiki_lines), len(cc_lines))
    
    print("-" * 40)
    print(f"Wiki (正例) 总数: {len(wiki_lines)}")
    print(f"CC   (负例) 总数: {len(cc_lines)}")
    print(f"平衡后每类数量: {min_count}")
    print("-" * 40)

    # 4. 随机采样
    # 从较大的数据集中随机抽取 min_count 条，较小的全取
    balanced_wiki = random.sample(wiki_lines, min_count)
    balanced_cc = random.sample(cc_lines, min_count)

    # 5. 格式化并合并
    # 格式要求: __label__标签名 文本内容
    train_data = []
    
    for line in balanced_wiki:
        train_data.append(f"__label__wiki {line}\n")
        
    for line in balanced_cc:
        train_data.append(f"__label__cc {line}\n")

    # 6. 全局打乱 (Shuffle)
    # 这对训练非常重要
    random.shuffle(train_data)

    # 7. 写入最终文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)

    print(f"成功！训练数据集已生成至: {output_path}")
    print(f"总样本数: {len(train_data)}")

if __name__ == "__main__":
    build_balanced_dataset()