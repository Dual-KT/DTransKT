from typing import List, Dict
import pandas as pd

def read_txt_in_blocks(file_path: str) -> List[Dict]:
    """Read six-line blocks from a data file and return student interaction records."""
    data = []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        while True:
            # Read six lines as one complete interaction record.
            lines = [f.readline().strip() for _ in range(6)]

            # Stop at the end of the file.
            if all(not line for line in lines):  # All lines empty.
                break

            # Parse each block into a record.
            try:
                # Parse user, problems, skills, answers, and timing fields.
                user_info = lines[0]
                seq_problems = lines[1]
                seq_skills = lines[2]
                seq_ans = lines[3]
                seq_start_time = lines[4]
                seq_response_cost = lines[5]
                # Build a record dictionary for one student.
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
                # Report parsing errors and skip the current block.
                print(f"parse error: {e}")
                print(f"failed lines: {lines}")
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