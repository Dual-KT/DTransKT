from typing import List, Dict
import pandas as pd
import numpy as np
import os

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
    print(data.info())
    # print(data.iloc[12])

    return data


def read_csv_file(file_path):
    """Load a CSV file and report basic dataset information."""
    try:
        # Read the CSV file.
        df = pd.read_csv(file_path)
        print(f"loaded file: {file_path}")
        print(f"dataset summary:")
        df.info()
        return df
    except FileNotFoundError:
        print(f"error: file not found '{file_path}'")
        return None
    except Exception as e:
        print(f"error: failed to read file - {str(e)}")
        return None


# Legacy variant retained for reference; superseded by the placeholder implementation below.

def get_newassist(df_qes, df_assist_idea1):
    """Provide placeholder texts for datasets without native question descriptions."""
    # Ensure required columns for ID-to-text mapping.
    if 'id' not in df_qes.columns or 'name' not in df_qes.columns:
        raise ValueError("df_qes must contain 'id' and 'name' columns")

    # Ensure required columns for placeholder generation.
    if 'seq_skills' not in df_assist_idea1.columns or 'seq_problems' not in df_assist_idea1.columns:
        raise ValueError("df_assist_idea1 must contain 'seq_skills' and 'seq_problems' columns")

    # Build an ID-to-text map for placeholder sampling.
    id_to_question = dict(zip(df_qes['id'], df_qes['name']))

    # Store placeholder sequences for each record.
    new_seq_problems = []

    # Sample placeholder texts matched to skill counts.
    for _, row in df_assist_idea1.iterrows():
        seq_skills = row['seq_skills']

        # Count skills to determine placeholder length.
        skills = seq_skills.split(',')
        num_skills = len(skills)

        # Return empty text for records without skills.
        if num_skills == 0:
            new_seq_problems.append('')
            continue

        # Sample matching IDs without replacement.
        # num_skills = 3
        max_id = min(num_skills, len(df_qes))
        random_ids = np.random.choice(df_qes['id'], size = max_id, replace = False)

        # Collect mapped texts with comma protection.
        questions = [id_to_question.get(id, f"unknown_{id}") for id in random_ids]
        questions = [s.replace(',', '@@@') for s in questions]
        combined_questions = ','.join(questions)
        new_seq_problems.append(combined_questions)

    # Update skill texts with sampled placeholders.
    df_assist_idea1['seq_skills'] = new_seq_problems

    return df_assist_idea1

if __name__ == '__main__':

    df_assist_idea1 = read_txt_in_blocks('./algebra2005/data-as.txt')
    qes_sentence = './dataverse_files/2_DBE_KT22_datafiles_100102_csv/KCs.csv'

    # df_assist_idea1 = read_txt_in_blocks('./algebra2005/data-as-1.txt')
    # qes_sentence = './dataverse_files/2_DBE_KT22_datafiles_100102_csv/Questions.csv'

    df_qes = read_csv_file(qes_sentence)
    # Supplement placeholder corpus for missing texts.
    df_assist_idea1 = get_newassist(df_qes, df_assist_idea1)
    df_assist_idea12 = df_assist_idea1.copy()
    df_assist_idea12['user'] = df_assist_idea12['user'].astype(str) + ',' + df_assist_idea12['seq_len'].astype(str)
    # Save selected columns in block format.
    selected_columns = ['user', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
    current_dir = os.getcwd()
    print("file saved in: ", current_dir)
    df_assist_idea12.to_csv(
        './algebra2005/data-as-2.txt',
        sep = '\n',  # One field per line block.
        columns = selected_columns,  # Save selected columns only.
        header = False,  # Omit the header.
        index = False,  # Omit the index.
        na_rep = 'NA',  # Represent missing values as NA.
        float_format = '%s'  # Avoid scientific notation.
    )
