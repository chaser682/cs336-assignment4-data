from __future__ import annotations

import os
from typing import Any

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
import hashlib
from pathlib import Path
import shutil
import struct
import unicodedata
import numpy as np
from typing import List, Set, Tuple


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    从 HTML 字节流中提取纯文本。
    会自动检测编码将字节流转换为字符串，然后提取文本。
    """
    if not html_bytes:
        return ""
    try:
        # 1. 优先尝试标准的 UTF-8 解码（最常见且效率最高）
        html_string = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 2. 如果 UTF-8 失败，使用 Resiliparse 检测实际编码
        detected_encoding = detect_encoding(html_bytes)
        
        # 3. 使用检测到的编码进行解码
        # 使用 errors='replace' 是为了防止即使检测了编码，仍有部分非法字节导致程序崩溃
        html_string = html_bytes.decode(detected_encoding, errors='replace')

    # 4. 使用 extract_plain_text 提取文本
    return extract_plain_text(html_string)


def run_identify_language(text: str) -> tuple[Any, float]:
    '''
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "var/lid.176.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)
    predictions = model.predict([text])
    '''
    # 1. 定义模型路径
    model_path = "var/lid.176.bin"
    # 2. 加载模型
    _FASTTEXT_MODEL = fasttext.load_model(model_path)
    # 3. 预处理：FastText 预测时，输入文本不能包含换行符，否则会报错或准确率下降
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return "unknown", 0.0
    # 4. 执行预测
    # predict 返回格式通常为: (('__label__en',), array([0.88]))
    labels, scores = _FASTTEXT_MODEL.predict(clean_text)
    # 5. 提取结果
    if not labels:
        return "unknown", 0.0
    raw_label = labels[0]
    score = scores[0]
    # 6. 映射转换
    # FastText 输出格式为 "__label__en"，测试期望 "en"
    language_code = raw_label.replace("__label__", "")
    # print(f"Identified language: {language_code} with score: {score}")
    return language_code, float(score)


def run_mask_emails(text: str) -> tuple[str, int]:
    """
    将所有电子邮件地址替换为占位符 |||EMAIL_ADDRESS|||
    """
    if not text:
        return text, 0
    
    # 一个广泛使用的、实用的电子邮件正则表达式
    # 允许字母、数字、点、下划线、加号、减号作为用户名
    # 域名部分允许字母、数字、点和减号
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # re.subn 返回 (处理后的字符串, 替换次数)
    return re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    将所有电话号码替换为占位符 |||PHONE_NUMBER|||
    兼容常见的美国格式，如 (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
    """
    if not text:
        return text, 0

    # 电话号码正则解释：
    # (?:\b\+?1[\s.-]?)?     -> 可选的美国国家代码 (+1 或 1)，后面跟可选的分隔符
    # (?:\(\d{3}\)|\b\d{3})  -> 区号，可以是 (123) 或者 123
    # [\s.-]?                -> 区号后的分隔符 (空格、点或短横线)
    # \d{3}                  -> 中间三位数字
    # [\s.-]?                -> 分隔符
    # \d{4}\b                -> 最后四位数字，以单词边界结束
    phone_pattern = r'(?:\b\+?1[\s.-]?)?(?:\(\d{3}\)|\b\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b'
    
    return re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)


def run_mask_ips(text: str) -> tuple[str, int]:
    """
    将所有 IPv4 地址替换为占位符 |||IP_ADDRESS|||
    严格检查每个段是否在 0-255 之间
    """
    if not text:
        return text, 0

    # 构建匹配 0-255 的正则片段
    # 25[0-5]       匹配 250-255
    # 2[0-4][0-9]   匹配 200-249
    # [01]?[0-9][0-9]? 匹配 0-199 (包括个位数和两位数)
    octet = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    
    # 组合四个片段，中间用 \. 连接
    # 前后加上 \b 确保不会匹配到像 123.123.123.1234 这样过长的数字串的一部分
    ip_pattern = fr'\b{octet}\.{octet}\.{octet}\.{octet}\b'
    
    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    '''
    使用NSFW 分类器
    过滤不适合工作场景（NSFW）内容
    '''
    # 1. 定义模型路径
    model_path = "var/jigsaw_fasttext_bigrams_nsfw_final.bin"
    # 2. 加载模型
    model = fasttext.load_model(model_path)
    # 3. 预处理：FastText 预测时，输入文本不能包含换行符，否则会报错或准确率下降
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return "unknown", 0.0
    # 4. 执行预测
    labels, scores = model.predict(clean_text)
    # 5. 提取结果
    if not labels:
        return "unknown", 0.0
    raw_label = labels[0]
    score = scores[0]
    # 6. 映射转换
    label = raw_label.replace("__label__", "")
    return label, float(score)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    '''
    使用仇恨言论分类器
    过滤有毒言论内容
    '''
    # 1. 定义模型路径
    model_path = "var/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    # 2. 加载模型
    model = fasttext.load_model(model_path)
    # 3. 预处理：FastText 预测时，输入文本不能包含换行符，否则会报错或准确率下降
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return "unknown", 0.0
    # 4. 执行预测
    labels, scores = model.predict(clean_text)
    # 5. 提取结果
    if not labels:
        return "unknown", 0.0
    raw_label = labels[0]
    score = scores[0]
    # 6. 映射转换
    label = raw_label.replace("__label__", "")
    return label, float(score)


def run_classify_quality(text: str) -> tuple[Any, float]:
    '''
    使用 维基百科数据集 训练的fasttext质量分类器
    来评估文本质量
    '''
    # 1. 定义模型路径
    model_path = "var/quality_classifier.bin"
    # 2. 加载模型
    model = fasttext.load_model(model_path)
    # 3. 预处理：FastText 预测时，输入文本不能包含换行符，否则会报错或准确率下降
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return "unknown", 0.0
    # 4. 执行预测
    labels, scores = model.predict(clean_text)
    # 5. 提取结果
    if not labels:
        return "unknown", 0.0
    raw_label = labels[0]
    score = scores[0]
    # 6. 映射转换
    label = raw_label.replace("__label__", "")
    return label, float(score)


def run_gopher_quality_filter(text: str) -> bool:
    """
    根据 Gopher 论文 [Rae et al., 2021] 的启发式规则过滤文本。
    
    规则：
    1. 单词数在 [50, 100,000] 之间。
    2. 平均单词长度在 [3, 10] 字符之间。
    3. 以省略号 ("...") 结尾的行占比 <= 30%。
    4. 包含至少一个字母的单词占比 >= 80%。
    """
    # 使用空白字符分词获取单词列表
    words = text.split()
    n_words = len(words)

    # 规则 1: 单词数检查
    # 移除单词数少于 50 或多于 100,000 的文档
    if n_words < 50 or n_words > 100000:
        return False

    # 规则 2: 平均单词长度检查
    # 移除平均单词长度超出 3-10 个字符范围的文档
    total_chars = sum(len(w) for w in words)
    mean_word_len = total_chars / n_words
    if mean_word_len < 3 or mean_word_len > 10:
        return False

    # 规则 3: 省略号结尾行占比检查
    # 移除以省略号（“...”）结尾的行占比超过 30% 的文档
    lines = text.splitlines()
    if lines:
        # 使用 strip() 忽略行尾的空白符
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith('...'))
        if (ellipsis_lines / len(lines)) > 0.3:
            return False

    # 规则 4: 字母单词占比检查
    # 移除包含至少一个字母的单词占比低于 80% 的文档
    alpha_word_count = sum(1 for w in words if any(c.isalpha() for c in w))
    if (alpha_word_count / n_words) < 0.8:
        return False

    # 如果通过所有检查
    return True


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    精确行去重：统计所有文件中每一行的频率（使用哈希作为 key 以节省内存），
    然后将每个文件重写到输出目录，仅保留只出现过一次的行。

    Args:
        input_files (list[os.PathLike]): 输入文本文件路径列表
        output_directory (os.PathLike): 输出目录（保持输入文件文件名不变）
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # 第一步：统计每一行的频率（用哈希作为 Key）
    counter: dict[str, int] = {}

    for file_path in input_files:
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_norm = line.rstrip("\n")  # 保留行内容但去掉换行符
                h = hashlib.sha256(line_norm.encode("utf-8")).hexdigest()
                counter[h] = counter.get(h, 0) + 1

    # 第二步：重写文件，仅保留频率=1 的行
    for file_path in input_files:
        file_path = Path(file_path)
        out_path = output_directory / file_path.name

        with file_path.open("r", encoding="utf-8") as f_in, \
             out_path.open("w", encoding="utf-8") as f_out:

            for line in f_in:
                line_norm = line.rstrip("\n")
                h = hashlib.sha256(line_norm.encode("utf-8")).hexdigest()
                if counter.get(h, 0) == 1:
                    f_out.write(line)


# === 辅助类：并查集 (Union-Find) ===
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

# === 文本归一化处理 ===
def normalize_text(text: str) -> str:
    # 1. 应用 NFD Unicode 归一化
    text = unicodedata.normalize("NFD", text)
    # 2. 移除重音符号 (combining characters)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    # 3. 小写转换
    text = text.lower()
    # 4. 移除标点 (保留字符和空白)
    text = re.sub(r'[^\w\s]', '', text)
    # 5. 归一化空白字符 (将连续空白替换为单个空格，去除首尾空白)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === N-Gram 生成 ===
def get_ngrams(text: str, n: int) -> Set[bytes]:
    words = text.split()
    if len(words) < n:
        # 如果文本过短，处理为一个整体
        return {text.encode('utf-8')}
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        # 使用空格连接单词形成 n-gram
        ngram_str = " ".join(words[i : i + n])
        ngrams.add(ngram_str.encode('utf-8'))
    return ngrams

# === 核心函数 ===
def run_minhash_deduplication(
    input_files: List[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    使用 MinHash 和 LSH 对文档进行模糊去重。
    """
    input_files = [Path(p) for p in input_files]
    output_directory = Path(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    num_docs = len(input_files)
    if num_docs == 0:
        return

    rows_per_band = num_hashes // num_bands
    
    # 1. 预处理：读取文件，归一化，生成 N-grams
    # doc_ngrams 存储每个文档的 n-gram 集合 (用于后续精确 Jaccard 计算)
    doc_ngrams: List[Set[bytes]] = []
    
    for filepath in input_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            normalized = normalize_text(content)
            doc_ngrams.append(get_ngrams(normalized, ngrams))

    # 2. 生成 MinHash 签名
    # 使用 61 位梅森素数
    mersenne_prime = (1 << 61) - 1
    max_hash = (1 << 32) - 1
    
    # 生成随机哈希参数 a, b
    # shape: (num_hashes, 1)
    rng = np.random.RandomState(42)
    a = rng.randint(1, mersenne_prime, size=(num_hashes, 1), dtype=np.uint64)
    b = rng.randint(0, mersenne_prime, size=(num_hashes, 1), dtype=np.uint64)

    # 初始化签名矩阵 (num_hashes, num_docs)，填充最大值
    signatures = np.full((num_hashes, num_docs), mersenne_prime, dtype=np.uint64)

    for doc_idx, grams in enumerate(doc_ngrams):
        if not grams:
            continue
        
        # 将 n-grams 转换为整数哈希列表
        # 使用 sha1 取前 8 字节作为基础哈希，保证稳定性
        gram_hashes = []
        for g in grams:
            h = hashlib.sha1(g).digest()
            # 取前 4 字节转为 unsigned int
            val = struct.unpack("I", h[:4])[0]
            gram_hashes.append(val)
        
        gram_hashes = np.array(gram_hashes, dtype=np.uint64) # shape: (num_grams,)
        
        # 向量化计算 MinHash
        # hash_vals = (a * gram + b) % prime
        # 这是一个广播操作: (num_hashes, 1) * (1, num_grams) -> (num_hashes, num_grams)
        # 注意：如果不分批次，大文档可能会导致内存爆炸。这里简单处理，假设内存足够。
        
        # 为了避免内存过大，我们按哈希函数分批计算或者直接利用 numpy 的 min
        # hash_vals = (a * gram_hashes + b) % mersenne_prime
        # signatures[:, doc_idx] = hash_vals.min(axis=1)
        
        # 优化内存版本：
        hash_values = (a @ gram_hashes.reshape(1, -1) + b) % mersenne_prime
        signatures[:, doc_idx] = hash_values.min(axis=1)

    # 3. LSH (Locality Sensitive Hashing)
    candidate_pairs = set()
    
    for band_idx in range(num_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        
        # 获取当前 band 的所有文档签名片段
        band_signatures = signatures[start_row:end_row, :] # (rows_per_band, num_docs)
        
        # 将列向量转换为 tuple 以便作为字典 key
        buckets = {}
        for doc_idx in range(num_docs):
            # 将 numpy array 转为 tuple (hashable)
            band_sig = tuple(band_signatures[:, doc_idx].tolist())
            
            if band_sig not in buckets:
                buckets[band_sig] = []
            buckets[band_sig].append(doc_idx)
            
        # 同一个 bucket 内的文档两两构成候选对
        for doc_indices in buckets.values():
            if len(doc_indices) > 1:
                for i in range(len(doc_indices)):
                    for j in range(i + 1, len(doc_indices)):
                        idx1, idx2 = doc_indices[i], doc_indices[j]
                        # 保证 idx1 < idx2 避免重复
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        candidate_pairs.add((idx1, idx2))

    # 4. 验证候选对并聚类
    uf = UnionFind(num_docs)
    
    for i, j in candidate_pairs:
        set_a = doc_ngrams[i]
        set_b = doc_ngrams[j]
        
        if not set_a or not set_b:
            continue
            
        # 计算杰卡德相似度
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity >= jaccard_threshold:
            uf.union(i, j)

    # 5. 输出结果
    # 找到每个聚类的代表（通常取 root 对应的文档，或者聚类中第一个出现的文档）
    kept_indices = set()
    for i in range(num_docs):
        root = uf.find(i)
        # 我们的策略：保留聚类中的代表元素 (root)。
        # UnionFind 的实现保证了同一个集合最终指向同一个 root。
        # 我们只保留那些自己就是 root 的节点，或者统一映射到 root。
        # 简单的做法：收集所有 root
        kept_indices.add(root)

    for i in kept_indices:
        src_path = input_files[i]
        dst_path = output_directory / src_path.name
        # 复制原始文件
        shutil.copy2(src_path, dst_path)
