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


class StudentMatcher:
    """跨学科学生匹配系统"""

    def __init__(self, feature_dim, hidden_dim=64, output_dim=32):
        """
        初始化匹配器
        feature_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出嵌入维度
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 初始化GAT模型用于提取节点嵌入
        self.model = GATEncoder(feature_dim, hidden_dim, output_dim)

        # 存储两个学科的学生嵌入和ID映射
        self.subject1_embeddings = None
        self.subject2_embeddings = None
        self.subject1_id_map = None
        self.subject2_id_map = None

        # FAISS索引，用于快速最近邻搜索
        self.index = None

    def build_graph(self, df, similarity_threshold=0.7, k_neighbors=20):
        """构建学生图"""
        # 提取并标准化特征
        print("-" * 50, '提取并标准化特征', "-" * 50)
        stu_features = df.groupby('user').agg({
            'seq_len': 'mean',
            'seq_skills': lambda x: len(set(skill for s in x for skill in s.split(',') if skill.strip())),  # 总知识点数量
            'seq_ans': lambda x: np.mean([int(a) for ans in x for a in ans.split(',') if a.strip()])  # 平均正确率
        }).reset_index()
        print("-" * 50, '学科数据集信息', "-" * 50)
        print(stu_features.info())

        features = stu_features[['seq_len', 'seq_skills', 'seq_ans']].values
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # 构建边（基于特征相似度）
        print("-" * 50, '基于特征相似度构建边', "-" * 50)
        # similarity = cosine_similarity(features)
        # edges = np.array([[i, j] for i in range(len(features)) for j in range(len(features))
        #                   if i != j and similarity[i, j] > similarity_threshold])
        # if len(edges) == 0:
        #     edges = np.array([[0, 0]])  # 避免空图
        #
        # edge_index = torch.tensor(edges.T, dtype = torch.long)
        # 使用FAISS构建图
        features = scaler.fit_transform(features).astype(np.float32)
        features = np.ascontiguousarray(features)
        d = features.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(features)
        index.add(features)

        # 搜索最近邻（+1 是为了包含自身，后续会跳过）
        D, I = index.search(features, k = k_neighbors + 1)

        # 构建边列表（修正索引访问）
        edges = []
        for i in range(len(features)):
            for j in range(1, k_neighbors + 1):  # j 范围：1~k_neighbors
                if i != I[i, j] and D[i, j] > similarity_threshold:
                    edges.append([i, I[i, j]])

        if not edges:
            edges = [[0, 0]]  # 避免空图

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        print(edge_index.shape)
        x = torch.tensor(features, dtype = torch.float)

        return x, edge_index

    def train(self, subject1_df, subject2_df, epochs=50, lr=0.001):
        """训练跨学科匹配模型"""
        # 构建两个学科的图
        print("-" * 50, '构建第一个学科的图', "-" * 50)
        x1, edge_index1 = self.build_graph(subject1_df)
        print("-" * 50, '构建第二个学科的图', "-" * 50)
        x2, edge_index2 = self.build_graph(subject2_df)

        # 保存ID映射
        self.subject1_id_map = subject1_df['user'].values
        self.subject2_id_map = subject2_df['user'].values

        # 训练模型
        print("-" * 50, '训练模型', "-" * 50)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 获取两个学科的嵌入
            embeddings1 = self.model(x1, edge_index1)
            embeddings2 = self.model(x2, edge_index2)

            # 计算对比损失（拉近相似节点，推远不相似节点）
            loss = self._contrastive_loss(embeddings1, embeddings2)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # 保存最终嵌入
        self.subject1_embeddings = embeddings1.detach().cpu().numpy()
        self.subject2_embeddings = embeddings2.detach().cpu().numpy()

        # 构建FAISS索引用于快速搜索
        self._build_faiss_index()

        return self

    def _contrastive_loss(self, embeddings1, embeddings2, temperature=0.1, margin=0.5, hard_ratio=0.5):
        """
        计算双向对比损失，结合了InfoNCE和三元组损失的优点

        参数:
        - embeddings1: 学科1的嵌入矩阵 [N1, d]
        - embeddings2: 学科2的嵌入矩阵 [N2, d]
        - temperature: 温度缩放参数
        - margin: 三元组损失的边界值
        - hard_ratio: 难例样本的比例
        """
        # 计算双向相似度矩阵
        sim_matrix12 = torch.matmul(embeddings1, embeddings2.transpose(0, 1)) / temperature  # [N1, N2]
        sim_matrix21 = torch.matmul(embeddings2, embeddings1.transpose(0, 1)) / temperature  # [N2, N1]

        # 双向最近邻
        pos_sim12, pos_indices12 = torch.max(sim_matrix12, dim = 1)  # [N1]
        pos_sim21, pos_indices21 = torch.max(sim_matrix21, dim = 1)  # [N2]

        # 双向一致性检查
        # consistency_mask = torch.zeros_like(sim_matrix12)
        # for i in range(len(pos_indices12)):
        #     j = pos_indices12[i]
        #     if pos_indices21[j] == i:
        #         consistency_mask[i, j] = 1.0

        # 难例挖掘 - 对每个正样本对，选择难负样本和随机负样本
        batch_size = len(embeddings1)

        # 计算难负样本 (相似度最高的负样本)
        hard_neg_sim12, hard_neg_indices12 = torch.topk(sim_matrix12, k = 2, dim = 1)  # [N1, 2]
        hard_neg_sim12 = hard_neg_sim12[:, 1]  # 排除自身 (因为pos_sim是最大值)

        # 随机负样本
        random_indices = torch.randint(0, len(embeddings2), (batch_size,))
        random_neg_sim12 = sim_matrix12[torch.arange(batch_size), random_indices]

        # 按比例混合难负样本和随机负样本
        mask = torch.rand(batch_size) < hard_ratio
        neg_sim12 = torch.where(mask, hard_neg_sim12, random_neg_sim12)

        # 计算三元组损失 (学科1→学科2方向)
        loss12 = F.relu(margin - pos_sim12 + neg_sim12).mean()

        # 同理计算学科2→学科1方向的损失
        hard_neg_sim21, _ = torch.topk(sim_matrix21, k = 2, dim = 1)
        hard_neg_sim21 = hard_neg_sim21[:, 1]
        random_indices2 = torch.randint(0, len(embeddings1), (len(embeddings2),))
        random_neg_sim21 = sim_matrix21[torch.arange(len(embeddings2)), random_indices2]
        mask2 = torch.rand(len(embeddings2)) < hard_ratio
        neg_sim21 = torch.where(mask2, hard_neg_sim21, random_neg_sim21)
        loss21 = F.relu(margin - pos_sim21 + neg_sim21).mean()

        # 结合双向损失
        loss = (loss12 + loss21) / 2

        # 添加InfoNCE风格的损失以增强对比学习
        # exp_sim12 = torch.exp(sim_matrix12)
        # info_nce_loss12 = -torch.mean(torch.log(torch.exp(pos_sim12) / exp_sim12.sum(dim=1)))
        # exp_sim21 = torch.exp(sim_matrix21)
        # info_nce_loss21 = -torch.mean(torch.log(torch.exp(pos_sim21) / exp_sim21.sum(dim=1)))
        # loss += (info_nce_loss12 + info_nce_loss21) / 2

        return loss

    def _build_faiss_index(self):
        """构建FAISS索引用于快速最近邻搜索"""
        # 使用FlatL2索引，适合中等规模数据集
        self.index = faiss.IndexFlatL2(self.output_dim)
        self.index.add(self.subject2_embeddings)

    def find_similar_students(self, subject1_user_id, top_k=5):
        """
        查找与给定学生最相似的目标学科学生
        subject1_user_id: 源学科学生ID
        top_k: 返回的相似学生数量
        """
        # 找到学生在嵌入矩阵中的索引
        idx = np.where(self.subject1_id_map == subject1_user_id)[0]
        if len(idx) == 0:
            return []

        idx = idx[0]

        # 获取学生嵌入
        query_embedding = self.subject1_embeddings[idx:idx + 1]

        # 使用FAISS进行快速搜索
        distances, indices = self.index.search(query_embedding, top_k)

        # 获取相似学生的ID和相似度分数
        similar_students = []
        for i, idx in enumerate(indices[0]):
            user_id = self.subject2_id_map[idx]
            similarity = 1 / (1 + distances[0][i])  # 将L2距离转换为相似度
            similar_students.append((user_id, similarity))

        return similar_students

    def save_model(self, save_dir):
        """保存模型和相关数据到指定目录"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok = True)

        # 保存模型参数
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

        # 保存学生嵌入
        np.save(os.path.join(save_dir, 'subject1_embeddings.npy'), self.subject1_embeddings)
        np.save(os.path.join(save_dir, 'subject2_embeddings.npy'), self.subject2_embeddings)

        # 保存ID映射
        np.save(os.path.join(save_dir, 'subject1_id_map.npy'), self.subject1_id_map)
        np.save(os.path.join(save_dir, 'subject2_id_map.npy'), self.subject2_id_map)

        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_dir, 'faiss_index.index'))

        print(f"模型已成功保存到: {save_dir}")

    @classmethod
    def load_model(cls, load_dir, feature_dim, hidden_dim=64, output_dim=32):
        """从指定目录加载模型和相关数据"""
        # 初始化模型
        model = StudentMatcher(feature_dim, hidden_dim, output_dim)

        # 加载模型参数
        model.model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt')))
        model.model.eval()

        # 加载学生嵌入
        model.subject1_embeddings = np.load(os.path.join(load_dir, 'subject1_embeddings.npy'))
        model.subject2_embeddings = np.load(os.path.join(load_dir, 'subject2_embeddings.npy'))

        # 加载ID映射
        model.subject1_id_map = np.load(os.path.join(load_dir, 'subject1_id_map.npy'), allow_pickle=True)
        model.subject2_id_map = np.load(os.path.join(load_dir, 'subject2_id_map.npy'), allow_pickle=True)

        # 加载FAISS索引
        model.index = faiss.read_index(os.path.join(load_dir, 'faiss_index.index'))

        print(f"模型已成功从 {load_dir} 加载")
        return model


class GATEncoder(nn.Module):
    """基于GAT的编码器，用于提取学生嵌入"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads = num_heads, dropout = 0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads = 1, concat = False, dropout = 0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return F.normalize(x, p = 2, dim = 1)  # L2归一化，便于相似度计算


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
    print(data.info())
    # print(data.iloc[12])

    return data


def create_similar_students_dataset(df, loaded_matcher, source_student_ids, top_k=3, copy_columns=None):
    """
    创建包含相似学生信息的新数据集

    参数:
    - df: 原始数据集
    - loaded_matcher: 已加载的匹配模型
    - source_student_ids: 需要查找相似学生的源学生ID列表
    - top_k: 每个源学生取前k个相似学生
    - copy_columns: 需要从源学生复制到相似学生的列名列表（格式：{'源列名':'目标列名'}）

    返回:
    - new_df: 包含相似学生信息的新数据集
    """

    # 设置默认复制的列
    if copy_columns is None:
        copy_columns = ['user', 'seq_len', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
        # copy_columns = ['user', 'seq_len', 'seq_problems', 'seq_start_time', 'seq_response_cost']

    # 存储所有相似学生的ID
    all_similar_ids = []

    # 存储每个相似学生对应的源学生ID（用于后续关联）
    source_mapping = {}

    # 对每个源学生ID，查找相似学生
    for source_id in source_student_ids:
        # 获取相似学生ID列表（格式可能为 [(similar_id, score), ...]）
        similar_students = loaded_matcher.find_similar_students(source_id, top_k = top_k)

        # 提取相似学生ID（根据实际返回格式调整）
        similar_ids = [item[0] if isinstance(item, tuple) else item
                       for item in similar_students]

        all_similar_ids.extend(similar_ids)

        # 记录每个相似学生对应的源学生
        for sim_id in similar_ids:
            source_mapping[sim_id] = source_id

    # 从原始数据集中提取相似学生的完整信息
    new_df = df[df['user'].isin(all_similar_ids)].copy()

    # 添加源学生ID列，便于后续分析
    new_df['source_student_id'] = new_df['user'].map(source_mapping)

    # 复制指定列的信息
    if copy_columns:
        # 创建源学生ID到源学生数据的映射
        source_data_map = {
            source_id: df2[df2['user'] == source_id].iloc[0].to_dict()
            for source_id in source_student_ids
        }

        # 遍历每个相似学生，复制列信息
        for idx, row in new_df.iterrows():
            source_id = row['source_student_id']
            if source_id in source_data_map:
                source_data = source_data_map[source_id]
                for col in copy_columns:
                    # 对于无语义信息的赋question_string
                    # if col == 'seq_problems':
                    #     new_df.at[idx, col] = str('question_string, question_string, question_string')
                    if (col in source_data) and (col in new_df.columns):
                        new_df.at[idx, col] = source_data[col]

    # 添加相似度分数列
    if isinstance(similar_students[0], tuple):
        # 创建ID到分数的映射
        score_mapping = {(item[0] if isinstance(item, tuple) else item): item[1]
                         for source_id in source_student_ids
                         for item in loaded_matcher.find_similar_students(source_id, top_k = top_k)}

        # 添加相似度分数列
        new_df['similarity_score'] = new_df['user'].map(score_mapping)

    return new_df


def data_generate(args):

    # 预处理数据，用于生成data.txt
    # process raw data
    if args.dataset_name == "peiyou":
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print(f"dname: {dname}, writef: {writef}")

    # split
    os.system("rm " + dname + "/*.pkl")

    print("-" * 50, f"{args.dataset_name}\data.txt生成完毕", "-" * 50)

    return dname, writef


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
