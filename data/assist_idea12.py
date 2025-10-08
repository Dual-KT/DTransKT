from typing import List, Dict
import pandas as pd
import numpy as np
import os

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


def read_csv_file(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据基本信息：")
        df.info()
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"错误：读取文件时发生错误 - {str(e)}")
        return None


# def get_newassist(df_qes, df_assist_idea1):
#     """
#         根据df_assist_idea1中的seq_skills从df_qes中随机获取对应数量的问题文本
#
#         参数:
#         df_qes: 包含id和question_text列的DataFrame
#         df_assist_idea1: 包含seq_skills和seq_problems列的DataFrame
#
#         返回:
#         处理后的df_assist_idea1，seq_problems已更新为随机问题文本
#         """
#     # 确保df_qes包含所需列
#     if 'id' not in df_qes.columns or 'question_text' not in df_qes.columns:
#         raise ValueError("df_qes必须包含'id'和'question_text'列")
#
#     # 确保df_assist_idea1包含所需列
#     if 'seq_skills' not in df_assist_idea1.columns or 'seq_problems' not in df_assist_idea1.columns:
#         raise ValueError("df_assist_idea1必须包含'seq_skills'和'seq_problems'列")
#
#     # 创建df_qes的id到question_text的映射字典
#     id_to_question = dict(zip(df_qes['id'], df_qes['question_text']))
#
#     # 存储处理后的seq_problems
#     new_seq_problems = []
#
#     # 遍历df_assist_idea1的每一行
#     for _, row in df_assist_idea1.iterrows():
#         seq_skills = row['seq_skills']
#
#         # 分割seq_skills获取技能数量
#         skills = seq_skills.split(',')
#         num_skills = len(skills)
#
#         # 从df_qes中随机选择num_skills个唯一的id
#         if num_skills == 0:
#             new_seq_problems.append('')
#             continue
#
#         # 确保df_qes中有足够的id
#         num_skills = 3
#         max_id = min(num_skills, len(df_qes))
#         random_ids = np.random.choice(df_qes['id'], size = max_id, replace = False)
#
#         # 获取对应的question_text并拼接
#         questions = [id_to_question.get(id, f"未知问题_{id}") for id in random_ids]
#         questions = [s.replace(',', '@@@') for s in questions]
#         combined_questions = ','.join(questions)
#         new_seq_problems.append(combined_questions)
#
#     # 更新df_assist_idea1的seq_problems列
#     df_assist_idea1['seq_problems'] = new_seq_problems
#
#     return df_assist_idea1


def get_newassist(df_qes, df_assist_idea1):
    """
        根据df_assist_idea1中的seq_skills从df_qes中随机获取对应数量的问题文本

        参数:
        df_qes: 包含id和question_text列的DataFrame
        df_assist_idea1: 包含seq_skills和seq_problems列的DataFrame

        返回:
        处理后的df_assist_idea1，seq_problems已更新为随机问题文本
        """
    # 确保df_qes包含所需列
    if 'id' not in df_qes.columns or 'name' not in df_qes.columns:
        raise ValueError("df_qes必须包含'id'和'name'列")

    # 确保df_assist_idea1包含所需列
    if 'seq_skills' not in df_assist_idea1.columns or 'seq_problems' not in df_assist_idea1.columns:
        raise ValueError("df_assist_idea1必须包含'seq_skills'和'seq_problems'列")

    # 创建df_qes的id到question_text的映射字典
    id_to_question = dict(zip(df_qes['id'], df_qes['name']))

    # 存储处理后的seq_problems
    new_seq_problems = []

    # 遍历df_assist_idea1的每一行
    for _, row in df_assist_idea1.iterrows():
        seq_skills = row['seq_skills']

        # 分割seq_skills获取技能数量
        skills = seq_skills.split(',')
        num_skills = len(skills)

        # 从df_qes中随机选择num_skills个唯一的id
        if num_skills == 0:
            new_seq_problems.append('')
            continue

        # 确保df_qes中有足够的id
        # num_skills = 3
        max_id = min(num_skills, len(df_qes))
        random_ids = np.random.choice(df_qes['id'], size = max_id, replace = False)

        # 获取对应的question_text并拼接
        questions = [id_to_question.get(id, f"未知问题_{id}") for id in random_ids]
        questions = [s.replace(',', '@@@') for s in questions]
        combined_questions = ','.join(questions)
        new_seq_problems.append(combined_questions)

    # 更新df_assist_idea1的seq_problems列
    df_assist_idea1['seq_skills'] = new_seq_problems

    return df_assist_idea1

if __name__ == '__main__':

    df_assist_idea1 = read_txt_in_blocks('./algebra2005/data-as.txt')
    qes_sentence = './dataverse_files/2_DBE_KT22_datafiles_100102_csv/KCs.csv'

    # df_assist_idea1 = read_txt_in_blocks('./algebra2005/data-as-1.txt')
    # qes_sentence = './dataverse_files/2_DBE_KT22_datafiles_100102_csv/Questions.csv'

    df_qes = read_csv_file(qes_sentence)
    # 补充语料
    df_assist_idea1 = get_newassist(df_qes, df_assist_idea1)
    df_assist_idea12 = df_assist_idea1.copy()
    df_assist_idea12['user'] = df_assist_idea12['user'].astype(str) + ',' + df_assist_idea12['seq_len'].astype(str)
    # 指定要保存的列
    selected_columns = ['user', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
    current_dir = os.getcwd()
    print("file saved in: ", current_dir)
    df_assist_idea12.to_csv(
        './algebra2005/data-as-2.txt',
        sep = '\n',  # 使用制表符分隔
        columns = selected_columns,  # 只保存指定的列
        header = False,  # 不保存表头
        index = False,  # 不保存行索引
        na_rep = 'NA',  # 将NaN显示为nan
        float_format = '%s'  # 避免科学计数法
    )
