import random
import os
from fastwarc.warc import ArchiveIterator, WarcRecordType
from adapters import run_extract_text_from_html_bytes

def extract_valid_texts_from_warc(warc_path):
    """
    读取 WARC 文件，提取所有符合条件的文本并存储在列表中返回。
    """
    valid_texts = []
    print(f"正在读取并提取: {warc_path} ...")
    
    if not os.path.exists(warc_path):
        print(f"警告: 文件不存在 {warc_path}")
        return []

    try:
        with open(warc_path, 'rb') as stream:
            for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
                try:
                    # 获取 HTML 字节
                    html_bytes = record.reader.read()
                    
                    # 提取文本
                    text = run_extract_text_from_html_bytes(html_bytes)
                    
                    # 过滤逻辑：长度大于 50 且非空
                    if text and len(text.strip()) > 50:
                        # 清洗换行符，FastText 要求一行一条
                        clean_text = text.replace("\n", " ").strip()
                        valid_texts.append(clean_text)
                except Exception as e:
                    # 捕获单条记录处理可能的异常，防止整个过程由于坏数据中断
                    continue
    except Exception as e:
        print(f"读取 WARC 文件出错: {e}")

    print(f"  - 从 {os.path.basename(warc_path)} 提取到 {len(valid_texts)} 条有效文本")
    return valid_texts

def build_balanced_dataset():
    wiki_warc_path = "var/subsampled_positive_wiki.warc.gz"
    cc_warc_path = "var/low_quality_cc.warc.gz"
    output_train_file = "var/quality_train.txt"
    
    # 1. 分别提取数据到内存
    wiki_texts = extract_valid_texts_from_warc(wiki_warc_path)
    cc_texts = extract_valid_texts_from_warc(cc_warc_path)

    # 2. 检查数据是否为空
    if not wiki_texts or not cc_texts:
        print("错误：无法生成训练集，因为其中一个数据集为空。")
        return

    # 3. 计算平衡数量 (取最小值)
    min_count = min(len(wiki_texts), len(cc_texts))
    print(f"--------------------------------------------------")
    print(f"正例 (Wiki) 数量: {len(wiki_texts)}")
    print(f"负例 (CC)   数量: {len(cc_texts)}")
    print(f"平衡后每类数量: {min_count}")
    print(f"--------------------------------------------------")

    # 4. 随机采样以平衡数据
    # random.sample 会从列表中随机抽取 k 个不重复的元素
    balanced_wiki = random.sample(wiki_texts, min_count)
    balanced_cc = random.sample(cc_texts, min_count)

    # 5. 构建最终带标签的数据行
    final_lines = []
    for text in balanced_wiki:
        final_lines.append(f"__label__wiki {text}\n")
    
    for text in balanced_cc:
        final_lines.append(f"__label__cc {text}\n")

    # 6. 全局打乱 (Shuffle)
    # 避免训练数据前一半全是正例，后一半全是负例
    random.shuffle(final_lines)

    # 7. 写入文件
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    with open(output_train_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(final_lines)

    print(f"平衡训练集已生成: {output_train_file}")
    print(f"总行数: {len(final_lines)}")

if __name__ == "__main__":
    build_balanced_dataset()