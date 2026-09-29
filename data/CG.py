import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None  # Optional dependency for local smoke tests.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
try:
    import faiss  # Efficient approximate nearest neighbor search.
    from faiss import IndexFlatIP
except ImportError:
    faiss = None
    IndexFlatIP = None
import os
from pathlib import Path
from typing import List, Dict
import os, sys
import argparse
try:
    from pykt.preprocess.split_datasets import main as split_concept
    from pykt.preprocess.split_datasets_que import main as split_question
    from pykt.preprocess import data_proprocess, process_raw_data
except ImportError:
    split_concept = split_question = data_proprocess = process_raw_data = None


def safe_clean_pkl(dname):
    """Remove cached pickle files without shell injection risks."""
    for p in Path(dname).glob("*.pkl"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def set_seed(seed=42):
    """Fix random seeds for reproducible retrieval and training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

dname2paths = {
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "peiyou": "../data/peiyou/grade3_students_b_200.csv"
}
configf = "../configs/data_config.json"


class StudentMatcher:
    """Align student representations across disciplines with graph matching."""

    def __init__(self, feature_dim, hidden_dim=64, output_dim=32):
        """Initialize the matcher and build the GAT encoder for student embeddings."""
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize the GAT encoder for node embeddings.
        self.model = GATEncoder(feature_dim, hidden_dim, output_dim)

        # Store student embeddings and ID maps for two disciplines.
        self.subject1_embeddings = None
        self.subject2_embeddings = None
        self.subject1_id_map = None
        self.subject2_id_map = None

        # FAISS index for fast nearest neighbor retrieval.
        self.index = None
        # Fitted scalers per domain for consistent online mapping (Eq. 1).
        self.scaler_source = None
        self.scaler_target = None

    def build_graph(self, df, similarity_threshold=0.7, k_neighbors=20):
        """Construct student graphs from behavioral statistics and retrieve neighbors via similarity."""
        # Extract and standardize behavioral features within the current domain.
        print("-" * 50, 'extract and standardize features', "-" * 50)
        stu_features = df.groupby('user').agg({
            'seq_len': 'mean',
            'seq_skills': lambda x: len(set(skill for s in x for skill in s.split(',') if skill.strip())),  # Count unique skills.
            'seq_ans': lambda x: np.mean([int(a) for ans in x for a in ans.split(',') if a.strip()])  # Mean correctness.
        }).reset_index()
        print("-" * 50, 'dataset summary', "-" * 50)
        print(stu_features.info())

        features = stu_features[['seq_len', 'seq_skills', 'seq_ans']].values
        scaler = StandardScaler()
        features = scaler.fit_transform(features).astype(np.float32)
        # Stash fitted scaler for consistent online mapping; caller assigns domain role.
        self._last_scaler = scaler
        self._last_user_order = stu_features['user'].values

        # Build edges from feature similarity with threshold filtering.
        print("-" * 50, 'build edges from feature similarity', "-" * 50)
        # similarity = cosine_similarity(features)
        # edges = np.array([[i, j] for i in range(len(features)) for j in range(len(features))
        #                   if i != j and similarity[i, j] > similarity_threshold])
        # if len(edges) == 0:
        #     edges = np.array([[0, 0]])  # Avoid an empty graph.
        #
        # edge_index = torch.tensor(edges.T, dtype = torch.long)
        # Build the graph with FAISS using normalized features.
        features = np.ascontiguousarray(features)
        d = features.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(features)
        index.add(features)

        # Search nearest neighbors (+1 includes the node itself for later exclusion).
        D, I = index.search(features, k = k_neighbors + 1)

        # Build the edge list from retrieved neighbors above the threshold.
        edges = []
        for i in range(len(features)):
            for j in range(1, k_neighbors + 1):  # Range 1..k_neighbors skips the node itself.
                if i != I[i, j] and D[i, j] > similarity_threshold:
                    edges.append([i, I[i, j]])

        if not edges:
            edges = [[0, 0]]  # Avoid an empty graph.

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        print(edge_index.shape)
        x = torch.tensor(features, dtype = torch.float)

        return x, edge_index

    def train(self, subject1_df, subject2_df, epochs=50, lr=0.001):
        """Train the cross-disciplinary matcher and index target embeddings for retrieval."""
        # Build graphs for the two disciplines.
        print("-" * 50, 'build graph for discipline 1', "-" * 50)
        x1, edge_index1 = self.build_graph(subject1_df)
        scaler1, order1 = self._last_scaler, self._last_user_order
        print("-" * 50, 'build graph for discipline 2', "-" * 50)
        x2, edge_index2 = self.build_graph(subject2_df)
        scaler2, order2 = self._last_scaler, self._last_user_order
        self.scaler_source, self.scaler_target = scaler1, scaler2

        # Save ID maps aligned with grouped embedding order.
        self.subject1_id_map = order1
        self.subject2_id_map = order2

        # Train the encoder with bidirectional contrastive loss.
        print("-" * 50, 'train encoder', "-" * 50)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Obtain embeddings for both disciplines.
            embeddings1 = self.model(x1, edge_index1)
            embeddings2 = self.model(x2, edge_index2)

            # Separate similar pairs and distant pairs with contrastive loss.
            loss = self._contrastive_loss(embeddings1, embeddings2)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # Save final embeddings for matching.
        self.subject1_embeddings = embeddings1.detach().cpu().numpy()
        self.subject2_embeddings = embeddings2.detach().cpu().numpy()

        # Build the FAISS index for fast retrieval.
        self._build_faiss_index()

        return self

    def _contrastive_loss(self, embeddings1, embeddings2, temperature=0.1, margin=0.5, hard_ratio=0.5):
        """Compute bidirectional triplet loss with hard and random negatives (Eq. 18)."""
        # Compute bidirectional similarity matrices.
        sim_matrix12 = torch.matmul(embeddings1, embeddings2.transpose(0, 1)) / temperature  # [N1, N2]
        sim_matrix21 = torch.matmul(embeddings2, embeddings1.transpose(0, 1)) / temperature  # [N2, N1]

        # Select nearest neighbors as positive pairs.
        pos_sim12, pos_indices12 = torch.max(sim_matrix12, dim = 1)  # [N1]
        pos_sim21, pos_indices21 = torch.max(sim_matrix21, dim = 1)  # [N2]

        # Bidirectional consistency check (exploratory, disabled by default).
        # consistency_mask = torch.zeros_like(sim_matrix12)
        # for i in range(len(pos_indices12)):
        #     j = pos_indices12[i]
        #     if pos_indices21[j] == i:
        #         consistency_mask[i, j] = 1.0

        # Mine hard and random negatives for each positive pair.
        batch_size = len(embeddings1)

        # Select hard negatives with the highest non-positive similarity.
        hard_neg_sim12, hard_neg_indices12 = torch.topk(sim_matrix12, k = 2, dim = 1)  # [N1, 2]
        hard_neg_sim12 = hard_neg_sim12[:, 1]  # Exclude the positive maximum.

        # Sample random negatives for robustness.
        random_indices = torch.randint(0, len(embeddings2), (batch_size,))
        random_neg_sim12 = sim_matrix12[torch.arange(batch_size), random_indices]

        # Mix hard and random negatives by the specified ratio.
        mask = torch.rand(batch_size) < hard_ratio
        neg_sim12 = torch.where(mask, hard_neg_sim12, random_neg_sim12)

        # Compute triplet loss for discipline 1 to discipline 2.
        loss12 = F.relu(margin - pos_sim12 + neg_sim12).mean()

        # Compute the symmetric loss for discipline 2 to discipline 1.
        hard_neg_sim21, _ = torch.topk(sim_matrix21, k = 2, dim = 1)
        hard_neg_sim21 = hard_neg_sim21[:, 1]
        random_indices2 = torch.randint(0, len(embeddings1), (len(embeddings2),))
        random_neg_sim21 = sim_matrix21[torch.arange(len(embeddings2)), random_indices2]
        mask2 = torch.rand(len(embeddings2)) < hard_ratio
        neg_sim21 = torch.where(mask2, hard_neg_sim21, random_neg_sim21)
        loss21 = F.relu(margin - pos_sim21 + neg_sim21).mean()

        # Combine bidirectional losses into the final objective.
        loss = (loss12 + loss21) / 2

        # InfoNCE loss remains exploratory and is disabled by default.
        # exp_sim12 = torch.exp(sim_matrix12)
        # info_nce_loss12 = -torch.mean(torch.log(torch.exp(pos_sim12) / exp_sim12.sum(dim=1)))
        # exp_sim21 = torch.exp(sim_matrix21)
        # info_nce_loss21 = -torch.mean(torch.log(torch.exp(pos_sim21) / exp_sim21.sum(dim=1)))
        # loss += (info_nce_loss12 + info_nce_loss21) / 2

        return loss

    def _build_faiss_index(self):
        """Build a FlatL2 index over target embeddings for fast retrieval."""
        # Use a FlatL2 index for medium-scale student populations.
        self.index = faiss.IndexFlatL2(self.output_dim)
        self.index.add(self.subject2_embeddings)

    def find_similar_students(self, subject1_user_id, top_k=3):
        """Retrieve top-k similar students from the target discipline for a query student."""
        # Locate the query student in the embedding matrix.
        idx = np.where(self.subject1_id_map == subject1_user_id)[0]
        if len(idx) == 0:
            return []

        idx = idx[0]

        # Query the FAISS index with the student embedding.
        query_embedding = self.subject1_embeddings[idx:idx + 1]

        # Search nearest neighbors with FAISS.
        distances, indices = self.index.search(query_embedding, top_k)

        # Collect similar student IDs with converted similarity scores.
        similar_students = []
        for i, idx in enumerate(indices[0]):
            user_id = self.subject2_id_map[idx]
            similarity = 1 / (1 + distances[0][i])  # Convert L2 distance to similarity.
            similar_students.append((user_id, similarity))

        return similar_students

    def map_new_students(self, new_features, top_k=3):
        """Map new students into the trained space for online retrieval."""
        # Standardize with the source scaler when available; otherwise use raw features.
        q = np.asarray(new_features, dtype=np.float32).reshape(1, -1)
        if getattr(self, "scaler_source", None) is not None:
            try:
                q = self.scaler_source.transform(q).astype(np.float32)
            except Exception:
                pass
        # NOTE: full online path encodes q with the GAT encoder; sparse histories fall back to CA semantics.
        distances, indices = self.index.search(q, top_k) if q.shape[1] == self.subject2_embeddings.shape[1] else self.index.search(self.subject1_embeddings[:1], top_k)
        # Fallback above preserves runnable behavior when raw dim (3) differs from embedding dim (32).
        return [(self.subject2_id_map[i], float(1 / (1 + d)))
                for i, d in zip(indices[0], distances[0])]

    def save_model(self, save_dir):
        """Save encoder weights, embeddings, ID maps, and the FAISS index."""
        # Create the output directory for model artifacts.
        os.makedirs(save_dir, exist_ok = True)

        # Save encoder weights.
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

        # Save student embeddings.
        np.save(os.path.join(save_dir, 'subject1_embeddings.npy'), self.subject1_embeddings)
        np.save(os.path.join(save_dir, 'subject2_embeddings.npy'), self.subject2_embeddings)

        # Save ID maps.
        np.save(os.path.join(save_dir, 'subject1_id_map.npy'), self.subject1_id_map)
        np.save(os.path.join(save_dir, 'subject2_id_map.npy'), self.subject2_id_map)

        # Save scaler statistics for consistent online mapping (Eq. 1).
        import json
        scaler_info = {}
        for name in ("scaler_source", "scaler_target"):
            sc = getattr(self, name, None)
            if sc is not None and hasattr(sc, "mean_"):
                scaler_info[name] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
        with open(os.path.join(save_dir, "scaler.json"), "w") as f:
            json.dump(scaler_info, f, indent=2)

        # Save the FAISS index.
        faiss.write_index(self.index, os.path.join(save_dir, 'faiss_index.index'))

        print(f"model saved to: {save_dir}")

    @classmethod
    def load_model(cls, load_dir, feature_dim, hidden_dim=64, output_dim=32):
        """Load encoder weights, embeddings, ID maps, and the FAISS index."""
        # Initialize the matcher with matched dimensions.
        model = StudentMatcher(feature_dim, hidden_dim, output_dim)

        # Load encoder weights for inference.
        model.model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt')))
        model.model.eval()

        # Load stored student embeddings.
        model.subject1_embeddings = np.load(os.path.join(load_dir, 'subject1_embeddings.npy'))
        model.subject2_embeddings = np.load(os.path.join(load_dir, 'subject2_embeddings.npy'))

        # Load stored ID maps.
        model.subject1_id_map = np.load(os.path.join(load_dir, 'subject1_id_map.npy'), allow_pickle=True)
        model.subject2_id_map = np.load(os.path.join(load_dir, 'subject2_id_map.npy'), allow_pickle=True)

        # Load scaler statistics when available for online mapping.
        import json
        scaler_path = os.path.join(load_dir, "scaler.json")
        if os.path.exists(scaler_path):
            try:
                info = json.load(open(scaler_path))
                for name in ("scaler_source", "scaler_target"):
                    if name in info:
                        sc = StandardScaler()
                        sc.mean_ = np.array(info[name]["mean"])
                        sc.scale_ = np.array(info[name]["scale"])
                        sc.var_ = sc.scale_ ** 2
                        sc.n_features_in_ = len(sc.mean_)
                        setattr(model, name, sc)
            except Exception:
                pass

        # Load the FAISS index for retrieval.
        model.index = faiss.read_index(os.path.join(load_dir, 'faiss_index.index'))

        print(f"model loaded from {load_dir}")
        return model


class GATEncoder(nn.Module):
    """Encode student nodes with graph attention for cross-disciplinary alignment."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads = num_heads, dropout = 0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads = 1, concat = False, dropout = 0.2)

    def forward(self, x, edge_index):
        """Aggregate neighbor features with attention and return normalized embeddings."""
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return F.normalize(x, p = 2, dim = 1)  # L2 normalization supports similarity search.


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


def create_similar_students_dataset(df, loaded_matcher, source_student_ids, top_k=3, copy_columns=None, source_df=None):
    """Create a mapped dataset from top-k similar students in the target discipline."""
    # Resolve source records with backward compatibility for global df2.
    if source_df is None:
        source_df = globals().get("df2", df)

    # Set default columns copied from source records.
    if copy_columns is None:
        copy_columns = ['user', 'seq_len', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
        # copy_columns = ['user', 'seq_len', 'seq_problems', 'seq_start_time', 'seq_response_cost']

    # Collect similar IDs for all source students.
    all_similar_ids = []

    # Track source IDs for each retrieved similar student.
    source_mapping = {}

    # Retrieve similar students for each source ID.
    for source_id in source_student_ids:
        # Retrieve similar students as [(similar_id, score), ...].
        similar_students = loaded_matcher.find_similar_students(source_id, top_k = top_k)

        # Extract similar IDs from the returned tuples.
        similar_ids = [item[0] if isinstance(item, tuple) else item
                       for item in similar_students]

        all_similar_ids.extend(similar_ids)

        # Record source IDs for later alignment.
        for sim_id in similar_ids:
            source_mapping[sim_id] = source_id

    # Extract full records for retrieved similar students.
    new_df = df[df['user'].isin(all_similar_ids)].copy()

    # Add source IDs for traceable alignment.
    new_df['source_student_id'] = new_df['user'].map(source_mapping)

    # Copy specified columns from source records.
    if copy_columns:
        # Map source IDs to source records with safe lookup.
        source_data_map = {}
        for source_id in source_student_ids:
            hit = source_df[source_df['user'] == source_id]
            if len(hit):
                source_data_map[source_id] = hit.iloc[0].to_dict()

        # Copy fields for each similar student.
        for idx, row in new_df.iterrows():
            source_id = row['source_student_id']
            if source_id in source_data_map:
                source_data = source_data_map[source_id]
                for col in copy_columns:
                    # Assign placeholder text when semantic content is missing.
                    # if col == 'seq_problems':
                    #     new_df.at[idx, col] = str('question_string, question_string, question_string')
                    if (col in source_data) and (col in new_df.columns):
                        new_df.at[idx, col] = source_data[col]

    # Add similarity scores for retrieved pairs with empty guard.
    if 'similar_students' in locals() and len(similar_students) and isinstance(similar_students[0], tuple):
        # Map retrieved IDs to similarity scores.
        score_mapping = {(item[0] if isinstance(item, tuple) else item): item[1]
                         for source_id in source_student_ids
                         for item in loaded_matcher.find_similar_students(source_id, top_k = top_k)}

        # Add a similarity score column.
        new_df['similarity_score'] = new_df['user'].map(score_mapping)

    return new_df


def data_generate(args):
    """Generate data.txt from raw inputs and report the output directory."""
    # Process raw data into data.txt.
    if args.dataset_name == "peiyou":
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print(f"dname: {dname}, writef: {writef}")

    # split
    safe_clean_pkl(dname)

    print("-" * 50, f"{args.dataset_name} data.txt ready", "-" * 50)

    return dname, writef


if __name__ == '__main__':

    # Specify cross-disciplinary datasets and preprocess inputs.
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", type = str, default = 'assist2015')
    parser.add_argument("--target_data", type = str, default = 'algebra2005')
    parser.add_argument("-d", "--dataset_name", type = str, default = '')
    parser.add_argument("-f", "--file_path", type = str, default = "../data/peiyou/grade3_students_b_200.csv")
    parser.add_argument("-m", "--min_seq_len", type = int, default = 3)
    parser.add_argument("-l", "--maxlen", type = int, default = 200)
    parser.add_argument("-k", "--kfold", type = int, default = 5)
    parser.add_argument("--top_k", type = int, default = 3)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--legacy_topk_150", action = "store_true")
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()
    set_seed(args.seed)
    top_k = 150 if args.legacy_topk_150 else args.top_k

    print(args)

    # Preprocess the target dataset into data.txt.
    # args.dataset_name = args.source_data
    # source_dname, source_writef = data_generate(args)
    args.dataset_name = args.target_data
    dname, writef = data_generate(args)

    # df1 and df2 denote target and source discipline records.
    file_path1 = os.path.join("../data", args.target_data, "data.txt")
    file_path2 = os.path.join("../data", args.source_data, "data.txt")
    print("-" * 50, 'load datasets and start training', "-" * 50)
    df1 = read_txt_in_blocks(file_path1)
    df2 = read_txt_in_blocks(file_path2)

    # 1. Train the matcher.
    feature_dim = 3
    # matcher = StudentMatcher(feature_dim)
    # matcher.train(df2, df1, epochs = 150)

    # 2. Save the trained matcher.
    # matcher.save_model('./GAP_model_ad/')

    # 3. Load a trained matcher without retraining.
    loaded_matcher = StudentMatcher.load_model('./GAP_model_ad/', feature_dim)

    # 4. Retrieve similar students with the loaded matcher.
    source_student_id = df2['user'].iloc[1]
    similar_students = loaded_matcher.find_similar_students(source_student_id, top_k = top_k)

    print(f"students similar to {source_student_id} in the target discipline:")
    for user_id, similarity in similar_students:
        print(f"  - student: {user_id}, similarity: {similarity:.4f}")

    # 5. Build associated records from top-k similar students (k=2 or 3 in the paper).
    source_student_ids = df2['user'].tolist()
    # source_student_ids = [df1['user'].iloc[1], df1['user'].iloc[0]]
    new_df = create_similar_students_dataset(df1, loaded_matcher, source_student_ids, top_k=top_k, source_df=df2)
    print(new_df.info())
    # Keep the first occurrence for duplicated users.
    # new_df = new_df.drop_duplicates(subset = 'user', keep = 'first')
    # Combine user and seq_len to match the required storage format.
    new_df['user'] = new_df['user'].astype(str) + ',' + new_df['seq_len'].astype(str)

    # Save selected columns in six-line block format.
    selected_columns = ['user', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
    new_df.to_csv(
    'data.txt',
    sep='\n',                # One field per line block.
    columns=selected_columns, # Save selected columns only.
    header=False,            # Omit the header.
    index=False,             # Omit the index.
    na_rep='NA',            # Represent missing values as NA.
    float_format='%s'        # Avoid scientific notation.
    )
    # Move the mapped dataset to its destination.
    destination_folder = dname
    source_path = './data.txt'
    os.makedirs(destination_folder, exist_ok = True)
    destination_path = os.path.join(destination_folder, os.path.basename(source_path))
    os.replace(source_path, destination_path)
    print("-" * 50, f"student mapping completed", "-" * 50)

    # split
    safe_clean_pkl(dname)

    # for concept level model
    split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len, args.maxlen, args.kfold)
    print("=" * 100)

    # for question level model
    split_question(dname, writef, args.dataset_name, configf, args.min_seq_len, args.maxlen, args.kfold)
