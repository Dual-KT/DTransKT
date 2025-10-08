import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # 用于高效近似最近邻搜索
from faiss import IndexFlatIP
import os
from typing import List, Dict
import os, sys
import argparse
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question
from pykt.preprocess import data_proprocess, process_raw_data

dname2paths = {
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "peiyou": "../data/peiyou/grade3_students_b_200.csv"
}
configf = "../configs/data_config.json"


# core code will be submited after accept!


if __name__ == '__main__':

    # 指定待迁移的跨学科数据集并预处理
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", type = str, default = 'assist2015')
    parser.add_argument("--target_data", type = str, default = 'algebra2005')
    parser.add_argument("-d", "--dataset_name", type = str, default = '')
    parser.add_argument("-f", "--file_path", type = str, default = "../data/peiyou/grade3_students_b_200.csv")
    parser.add_argument("-m", "--min_seq_len", type = int, default = 3)
    parser.add_argument("-l", "--maxlen", type = int, default = 200)
    parser.add_argument("-k", "--kfold", type = int, default = 5)
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)

    # 预处理生成data.txt
    # args.dataset_name = args.source_data
    # source_dname, source_writef = data_generate(args)
    args.dataset_name = args.target_data
    dname, writef = data_generate(args)

    # df1和df2是两个学科的DataFrame
    file_path1 = os.path.join("../data", args.target_data, "data.txt")
    file_path2 = os.path.join("../data", args.source_data, "data.txt")
    print("-" * 50, '读取数据集并开始训练', "-" * 50)
    df1 = read_txt_in_blocks(file_path1)
    df2 = read_txt_in_blocks(file_path2)

    # 1. 训练模型
    feature_dim = 3
    # matcher = StudentMatcher(feature_dim)
    # matcher.train(df2, df1, epochs = 150)

    # 2. 保存模型
    # matcher.save_model('./GAP_model_ad/')

    # 3. 在需要时加载模型（无需重新训练）
    loaded_matcher = StudentMatcher.load_model('./GAP_model_ad/', feature_dim)

    # 4. 使用加载的模型进行匹配
    source_student_id = df2['user'].iloc[1]
    similar_students = loaded_matcher.find_similar_students(source_student_id, top_k = 150)

    print(f"与学生 {source_student_id} 最相似的目标学科学生:")
    for user_id, similarity in similar_students:
        print(f"  - 学生ID: {user_id}, 相似度: {similarity:.4f}")

    # 5. 相似性Top3的学生关联数据
    source_student_ids = df2['user'].tolist()
    # source_student_ids = [df1['user'].iloc[1], df1['user'].iloc[0]]
    new_df = create_similar_students_dataset(df1, loaded_matcher, source_student_ids, top_k=150)
    print(new_df.info())
    # 删除user列重复的行，保留第一个出现的行
    # new_df = new_df.drop_duplicates(subset = 'user', keep = 'first')
    # 根据数据格式要求合并user与seq_len作为user存储
    new_df['user'] = new_df['user'].astype(str) + ',' + new_df['seq_len'].astype(str)

    # 指定要保存的列
    selected_columns = ['user', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
    new_df.to_csv(
    'data.txt',
    sep='\n',                # 使用制表符分隔
    columns=selected_columns, # 只保存指定的列
    header=False,            # 不保存表头
    index=False,             # 不保存行索引
    na_rep='NA',            # 将NaN显示为nan
    float_format='%s'        # 避免科学计数法
    )
    # 映射数据集迁移
    destination_folder = dname
    source_path = './data.txt'
    os.makedirs(destination_folder, exist_ok = True)
    destination_path = os.path.join(destination_folder, os.path.basename(source_path))
    os.replace(source_path, destination_path)
    print("-" * 50, f"学生映射完毕", "-" * 50)

    # split
    os.system("rm " + dname + "/*.pkl")

    # for concept level model
    split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len, args.maxlen, args.kfold)
    print("=" * 100)

    # for question level model
    split_question(dname, writef, args.dataset_name, configf, args.min_seq_len, args.maxlen, args.kfold)
