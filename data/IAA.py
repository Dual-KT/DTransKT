import pandas as pd
from typing import List, Dict
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import logging
import os
import re

# core code will be submited after accept!


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

