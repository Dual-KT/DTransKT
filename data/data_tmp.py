from typing import List, Dict
import pandas as pd

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

if __name__ == '__main__':
    file_path1 = './algebra2005/data.txt'
    file_path2 = './dataverse_files/output/data.txt'
    file_path3 = './assist2015/data.txt'
    df1 = read_txt_in_blocks(file_path1)
    df2 = read_txt_in_blocks(file_path2)
    df3 = read_txt_in_blocks(file_path3)

    # print(df1.iloc[12]['user'])
    for i in range(2,10):
        print(df1.iloc[i]['user'])

    for t in range(2,32):
        print(df3.iloc[t]['user'])