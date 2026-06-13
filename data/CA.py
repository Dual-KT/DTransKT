import pandas as pd
from typing import List, Dict
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import logging
import os
import re

def get_concept_embedding(concept_text, pooling_strategy='cls', device='cpu'):
    # 从镜像加载分词器和模型
    model_path = './Bert'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    device = device
    model.to(device)
    model.eval()  # 设置为评估模式

    inputs = tokenizer(
        concept_text,
        return_tensors = "pt",
        padding = 'max_length',
        truncation = True,
        max_length = 128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)  # 直接调用模型，无需.bert

    # 从outputs中获取last_hidden_state
    hidden_states = outputs.last_hidden_state

    # 池化逻辑保持不变
    if pooling_strategy == 'cls':
        embedding = hidden_states[:, 0, :]
    elif pooling_strategy == 'mean':
        attention_mask = inputs['attention_mask']
        embedding = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim = True)
    elif pooling_strategy == 'max':
        attention_mask = inputs['attention_mask']
        embedding = hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9).max(1)[0]
    else:
        raise ValueError(f"不支持的池化策略: {pooling_strategy}")

    return embedding.cpu().numpy().flatten()


def read_txt_in_blocks(file_path: str) -> List[Dict]:
    """
    按块读取文本文件，每6行组成一条完整记录
    :param file_path: 文本文件路径
    :return: 包含所有记录的列表，每条记录为字典
    """
    data = []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        while True:
            # 读取6行组成一条完整记录
            lines = [f.readline().strip() for _ in range(6)]

            # 检查是否到达文件末尾
            if all(not line for line in lines):  # 全为空行
                break

            # 过滤空行并解析
            try:
                # 解析每行数据
                user_info = lines[0]
                seq_problems = lines[1]
                seq_skills = lines[2]
                seq_ans = lines[3]
                seq_start_time = lines[4]
                seq_response_cost = lines[5]
                # 构建记录字典
                record = {
                    'user': user_info.split(',')[0],
                    'seq_len': int(user_info.split(',')[1]),
                    'seq_problems': seq_problems,
                    'seq_skills': seq_skills,
                    'seq_ans': seq_ans,
                    'seq_start_time': seq_start_time,
                    'seq_response_cost': seq_response_cost
                }
                data.append(record)

            except Exception as e:
                # 打印错误信息并跳过当前块
                print(f"解析错误: {e}")
                print(f"错误行: {lines}")
                continue

    data = pd.DataFrame(data)
    # print(data.info())
    # print(data.iloc[12])

    return data


def calculate_similarity(concept1_embedding, concept2_embedding):
    """计算两个概念之间的余弦相似度"""
    # 重塑向量为二维数组以适应cosine_similarity函数
    sim = cosine_similarity(
        concept1_embedding.reshape(1, -1),
        concept2_embedding.reshape(1, -1)
    )
    return sim[0][0]


def build_similarity_matrix(source_concepts, target_concepts):
    print("-" * 50, '构建源学科与目标学科概念间的相似度矩阵', "-" * 50)
    matrix = np.zeros((len(source_concepts), len(target_concepts)))

    for i, source_concept in enumerate(source_concepts):
        source_embedding = get_concept_embedding(source_concept)
        for j, target_concept in enumerate(target_concepts):
            target_embedding = get_concept_embedding(target_concept)
            similarity = calculate_similarity(source_embedding, target_embedding)
            matrix[i, j] = similarity

    return matrix


def generate_attention_weights(similarity_matrix, temperature=1.0):
    print("-" * 50, "生成注意力权重矩阵", "-" * 50)
    """
    生成注意力权重矩阵

    参数:
    similarity_matrix: 概念相似度矩阵
    temperature: 温度参数，控制分布的平滑度
    """
    # 应用温度缩放
    scaled_similarity = similarity_matrix / temperature

    # 对每行应用softmax，使每个源概念对目标概念的权重和为1
    exp_similarity = np.exp(scaled_similarity)
    attention_weights = exp_similarity / np.sum(exp_similarity, axis = 1, keepdims = True)

    return attention_weights


def enhance_text_with_attention(source_concepts, target_concepts, attention_weights, problem_text):
    """增强文本中的学科概念（调试增强版）"""
    print("\n===== 开始概念增强处理 =====")
    print(f"原始文本: {problem_text[:80]}...")

    # 验证输入有效性
    if not source_concepts or not target_concepts:
        print("警告: 源概念或目标概念列表为空")
        return problem_text

    # 构建概念映射表 (源概念 -> (目标概念, 权重))
    concept_mapping = {}
    for i, source_concept in enumerate(source_concepts):
        max_weight = attention_weights[i].max()
        max_indices = np.where(np.abs(attention_weights[i] - max_weight) < 1e-8)[0]
        max_weight_idx = max_indices[0]

        target_concept = target_concepts[max_weight_idx]
        weight_value = attention_weights[i, max_weight_idx]
        concept_mapping[source_concept] = (target_concept, weight_value)
        # print(f"概念映射: '{source_concept}' -> '{target_concept}@{weight_value:.4f}'")

    # 按概念长度降序排序
    sorted_indices = sorted(
        range(len(source_concepts)),
        key = lambda i: -len(source_concepts[i])
    )

    # 统一使用小写进行概念检测
    lower_text = problem_text.lower()
    concept_found = any(concept.lower() in lower_text for concept in source_concepts)

    if not concept_found:
        print("警告: 文本中未找到任何源学科概念")
        return problem_text

    # 使用原始文本进行替换
    enhanced_text = problem_text
    print("\n===== 开始替换过程 =====")

    for i in sorted_indices:
        source_concept = source_concepts[i]

        if source_concept.lower() in lower_text:
            target_concept, weight_value = concept_mapping[source_concept]
            tag = f"{target_concept}@{weight_value*len(target_concepts):.2f}"

            print(f"\n处理概念: '{source_concept}'")
            print(f"  目标概念: '{target_concept}@{weight_value:.4f}'")

            # 使用更灵活的边界匹配
            escaped_concept = re.escape(source_concept)
            pattern = re.compile(rf'(?:^|(?<=\W)){escaped_concept}(?:$|(?=\W))', re.IGNORECASE)

            # 调试：检查匹配情况
            matches = list(pattern.finditer(enhanced_text))
            if not matches:
                print(f"  警告: 宽松匹配仍未找到任何内容")
                # 尝试查找概念的部分匹配，帮助诊断问题
                partial_pattern = re.compile(rf'{escaped_concept}', re.IGNORECASE)
                partial_matches = list(partial_pattern.finditer(enhanced_text))
                if partial_matches:
                    print(f"  提示: 找到部分匹配，可能边界问题")
                    for m in partial_matches:
                        context = enhanced_text[max(0, m.start() - 20):min(len(enhanced_text), m.end() + 20)]
                        print(f"    部分匹配位置: {m.start()}, 上下文: '...{context}...'")
                continue

            print(f"  匹配到 {len(matches)} 处:")
            for match in matches:
                context = enhanced_text[max(0, match.start() - 20):min(len(enhanced_text), match.end() + 20)]
                print(f"    位置: {match.start()}, 上下文: '...{context}...'")

            # 执行替换
            enhanced_text = pattern.sub(tag, enhanced_text)
            lower_text = enhanced_text.lower()

    return enhanced_text


def process_seq_problems(text):
    # 1. 拆问题为列表
    items = text.split(',')

    # 2. 处理每个问题
    processed_items = []
    for item in items:
        enhanced_text = enhance_text_with_attention(
            source_concepts,
            target_concepts,
            attention_weights,
            item
        )
        processed_items.append(enhanced_text)

    # 3. 将处理后的列表元素用逗号重新拼接为字符串
    return ','.join(processed_items)


if __name__ == '__main__':
    df_original = read_txt_in_blocks('./algebra2005/data.txt')
    df_target = read_txt_in_blocks('./algebra2005/data-al.txt')

    # 初始化嵌入器，支持GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source_concepts = list(set([item for s in df_original['seq_skills'] for item in (s.split(',') if isinstance(s, str) else s)]))
    target_concepts = list(set([item for s in df_target['seq_skills'] for item in (s.split(',') if isinstance(s, str) else s)]))

    # 构建相似度矩阵
    # 定义缓存文件路径
    CACHE_DIR = "cache"
    SIMILARITY_MATRIX_FILE = os.path.join(CACHE_DIR, "similarity_matrix.npy")
    ATTENTION_WEIGHTS_FILE = os.path.join(CACHE_DIR, "attention_weights.npy")

    # 创建缓存目录（如果不存在）
    os.makedirs(CACHE_DIR, exist_ok = True)

    # 检查缓存是否存在，如果存在则加载
    if os.path.exists(SIMILARITY_MATRIX_FILE) and os.path.exists(ATTENTION_WEIGHTS_FILE):
        print("加载已缓存的相似度矩阵和注意力权重...")
        similarity_matrix = np.load(SIMILARITY_MATRIX_FILE)
        attention_weights = np.load(ATTENTION_WEIGHTS_FILE)

    else:
        print("缓存不存在，重新计算相似度矩阵和注意力权重...")
        logging.set_verbosity_error()  # 只显示错误信息
        similarity_matrix = build_similarity_matrix(source_concepts, target_concepts)
        attention_weights = generate_attention_weights(similarity_matrix, temperature = 0.8)

        # 保存结果到缓存
        np.save(SIMILARITY_MATRIX_FILE, similarity_matrix)
        np.save(ATTENTION_WEIGHTS_FILE, attention_weights)
        print(f"已将结果保存到缓存: {CACHE_DIR}")

    # problem_text = 'Set How many rows will be in the result for the following relational algebra expression?,Which of the following statements are not correct?,Which of the following are used to make access decisions in Mandatory Access Control (MAC)?'
    # print(problem_text)
    # eng_seq = enhance_text_with_attention(source_concepts, target_concepts, attention_weights, problem_text)
    # print(eng_seq)
    print(df_original['seq_problems'].iloc[1])
    print("-" * 50, '使用跨学科概念注意力权重增强问题文本', "-" * 50)
    df_original['seq_problems'] = df_original['seq_problems'].apply(process_seq_problems)
    print(df_original['seq_problems'].iloc[1])
    df_original['user'] = df_original['user'].astype(str) + ',' + df_original['seq_len'].astype(str)
    # 指定要保存的列
    selected_columns = ['user', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
    current_dir = os.getcwd()
    df_original.to_csv(
        './dataverse_files/output/data3.txt',
        sep = '\n',  # 使用制表符分隔
        columns = selected_columns,  # 只保存指定的列
        header = False,  # 不保存表头
        index = False,  # 不保存行索引
        na_rep = 'NA',  # 将NaN显示为nan
        float_format = '%s'  # 避免科学计数法
    )

