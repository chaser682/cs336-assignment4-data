import gzip
import random

def sample_urls(input_path, output_path, n=5000):
    print(f"正在从 {input_path} 中采样 {n} 个 URL...")
    selected_urls = []
    
    # 蓄水池采样算法 (Reservoir Sampling) 适合读取大文件
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < n:
                selected_urls.append(line)
            else:
                j = random.randint(0, i)
                if j < n:
                    selected_urls[j] = line
            
            if i % 1000000 == 0:
                print(f"已扫描 {i} 行...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(selected_urls)
    print(f"采样完成，已保存至 {output_path}")

if __name__ == "__main__":
    # 确保你修改了下面的路径为你实际的文件路径
    sample_urls("var/enwiki-20240420-extracted_urls.txt.gz", "subsampled_positive_wiki.txt", n=5000)

'''
页面下载：
wget --timeout=5 \
     -i var/subsampled_positive_wiki.txt \
     --warc-file=var/subsampled_positive_wiki.warc.gz \
     -O /dev/null
'''