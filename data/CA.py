import pandas as pd
from typing import List, Dict
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import logging
import os
import re

_BERT_CACHE = {}


def _get_bert(model_path='./Bert', device='cpu'):
    """Load and cache BERT tokenizer and model for reuse."""
    key = (model_path, device)
    if key not in _BERT_CACHE:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        _BERT_CACHE[key] = (tokenizer, model)
    return _BERT_CACHE[key]


def get_concept_embedding(concept_text, pooling_strategy='cls', device='cpu'):
    """Encode concept text with BERT and return a pooled embedding vector."""
    # Reuse cached tokenizer and model across concepts.
    model_path = './Bert'
    tokenizer, model = _get_bert(model_path, device)

    inputs = tokenizer(
        concept_text,
        return_tensors = "pt",
        padding = 'max_length',
        truncation = True,
        max_length = 128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)  # Forward pass without gradient tracking.

    # Retrieve hidden states from model outputs.
    hidden_states = outputs.last_hidden_state

    # Apply the selected pooling strategy.
    if pooling_strategy == 'cls':
        embedding = hidden_states[:, 0, :]
    elif pooling_strategy == 'mean':
        attention_mask = inputs['attention_mask']
        embedding = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim = True)
    elif pooling_strategy == 'max':
        attention_mask = inputs['attention_mask']
        embedding = hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9).max(1)[0]
    else:
        raise ValueError(f"unsupported pooling strategy: {pooling_strategy}")

    return embedding.cpu().numpy().flatten()


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


def calculate_similarity(concept1_embedding, concept2_embedding):
    """Compute cosine similarity between two concept embeddings."""
    # Reshape vectors for the similarity function.
    sim = cosine_similarity(
        concept1_embedding.reshape(1, -1),
        concept2_embedding.reshape(1, -1)
    )
    return sim[0][0]


def build_similarity_matrix(source_concepts, target_concepts):
    """Compute cross-disciplinary concept similarity with BERT embeddings."""
    print("-" * 50, 'build similarity matrix across disciplines', "-" * 50)
    # Precompute target embeddings once for efficiency; numeric results unchanged.
    target_embeddings = [get_concept_embedding(t) for t in target_concepts]
    matrix = np.zeros((len(source_concepts), len(target_concepts)))

    for i, source_concept in enumerate(source_concepts):
        source_embedding = get_concept_embedding(source_concept)
        for j in range(len(target_concepts)):
            similarity = calculate_similarity(source_embedding, target_embeddings[j])
            matrix[i, j] = similarity

    return matrix


def generate_attention_weights(similarity_matrix, temperature=0.8):
    """Generate attention weights from similarity with temperature scaling."""
    print("-" * 50, "generate attention weight matrix", "-" * 50)
    # Scale similarity scores by temperature.
    scaled_similarity = similarity_matrix / temperature

    # Normalize each row with softmax for cross-disciplinary mapping.
    exp_similarity = np.exp(scaled_similarity)
    attention_weights = exp_similarity / np.sum(exp_similarity, axis = 1, keepdims = True)

    return attention_weights


def enhance_text_with_attention(source_concepts, target_concepts, attention_weights, problem_text, legacy_scaling=False):
    """Enhance question text by mapping source concepts to target counterparts."""
    print("\n===== enhance concepts =====")
    print(f"source text: {problem_text[:80]}...")

    # Validate concept lists before mapping.
    if not source_concepts or not target_concepts:
        print("warning: empty source or target concept list")
        return problem_text

    # Build mapping from source concepts to weighted target counterparts.
    concept_mapping = {}
    for i, source_concept in enumerate(source_concepts):
        max_weight = attention_weights[i].max()
        max_indices = np.where(np.abs(attention_weights[i] - max_weight) < 1e-8)[0]
        max_weight_idx = max_indices[0]

        target_concept = target_concepts[max_weight_idx]
        weight_value = attention_weights[i, max_weight_idx]
        concept_mapping[source_concept] = (target_concept, weight_value)
        # print(f"mapping: '{source_concept}' -> '{target_concept}@{weight_value:.4f}'")

    # Sort concepts by length to avoid nested replacement conflicts.
    sorted_indices = sorted(
        range(len(source_concepts)),
        key = lambda i: -len(source_concepts[i])
    )

    # Detect source concepts with case-insensitive matching.
    lower_text = problem_text.lower()
    concept_found = any(concept.lower() in lower_text for concept in source_concepts)

    if not concept_found:
        print("warning: no source concept found in text")
        return problem_text

    # Replace matched concepts in the original text.
    enhanced_text = problem_text
    print("\n===== replace concepts =====")

    for i in sorted_indices:
        source_concept = source_concepts[i]

        if source_concept.lower() in lower_text:
            target_concept, weight_value = concept_mapping[source_concept]
            # Paper Eq. 17 uses raw weights; legacy mode scales by concept count.
            tag_weight = weight_value * len(target_concepts) if legacy_scaling else weight_value
            tag = f"{target_concept}@{tag_weight:.2f}"

            print(f"\nconcept: '{source_concept}'")
            print(f"  target: '{target_concept}@{weight_value:.4f}'")

            # Match independent concepts with word boundaries.
            escaped_concept = re.escape(source_concept)
            pattern = re.compile(rf'(?:^|(?<=\W)){escaped_concept}(?:$|(?=\W))', re.IGNORECASE)

            # Check matches before replacement.
            matches = list(pattern.finditer(enhanced_text))
            if not matches:
                print(f"  warning: no boundary match found")
                # Check partial matches for diagnosis.
                partial_pattern = re.compile(rf'{escaped_concept}', re.IGNORECASE)
                partial_matches = list(partial_pattern.finditer(enhanced_text))
                if partial_matches:
                    print(f"  note: partial matches suggest boundary issues")
                    for m in partial_matches:
                        context = enhanced_text[max(0, m.start() - 20):min(len(enhanced_text), m.end() + 20)]
                        print(f"    position: {m.start()}, context: '...{context}...'")
                continue

            print(f"  matched {len(matches)} occurrence(s):")
            for match in matches:
                context = enhanced_text[max(0, match.start() - 20):min(len(enhanced_text), match.end() + 20)]
                print(f"    position: {match.start()}, context: '...{context}...'")

            # Apply replacement and refresh detection text.
            enhanced_text = pattern.sub(tag, enhanced_text)
            lower_text = enhanced_text.lower()

    return enhanced_text


def process_seq_problems(text, source_concepts=None, target_concepts=None, attention_weights=None, legacy_scaling=False):
    """Enhance comma-separated problems with cross-disciplinary attention weights."""
    # Backward compatibility: fall back to globals when called with a single argument.
    if source_concepts is None:
        source_concepts = globals().get("source_concepts", [])
    if target_concepts is None:
        target_concepts = globals().get("target_concepts", [])
    if attention_weights is None:
        attention_weights = globals().get("attention_weights")
        if attention_weights is None:
            return text
    # 1. Split problems into items.
    items = text.split(',')

    # 2. Enhance each item with mapped concepts.
    processed_items = []
    for item in items:
        enhanced_text = enhance_text_with_attention(
            source_concepts,
            target_concepts,
            attention_weights,
            item,
            legacy_scaling=legacy_scaling
        )
        processed_items.append(enhanced_text)

    # 3. Rejoin enhanced items into one sequence.
    return ','.join(processed_items)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_txt", type=str, default='./algebra2005/data.txt')
    parser.add_argument("--target_txt", type=str, default='./algebra2005/data.txt')
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", type=str, default='./dataverse_files/output/data3.txt')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-scaling", action="store_true")
    parser.add_argument("--rho", type=float, default=0.0)
    args = parser.parse_args()
    np.random.seed(args.seed)

    df_original = read_txt_in_blocks(args.source_txt)
    df_target = read_txt_in_blocks(args.target_txt)

    # Select device for BERT encoding when available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source_concepts = list(set([item for s in df_original['seq_skills'] for item in (s.split(',') if isinstance(s, str) else s)]))
    target_concepts = list(set([item for s in df_target['seq_skills'] for item in (s.split(',') if isinstance(s, str) else s)]))

    # Define cache paths for similarity and attention matrices.
    CACHE_DIR = "cache"
    SIMILARITY_MATRIX_FILE = os.path.join(CACHE_DIR, "similarity_matrix.npy")
    ATTENTION_WEIGHTS_FILE = os.path.join(CACHE_DIR, "attention_weights.npy")

    # Create the cache directory when missing.
    os.makedirs(CACHE_DIR, exist_ok = True)

    # Load cached matrices when available.
    if os.path.exists(SIMILARITY_MATRIX_FILE) and os.path.exists(ATTENTION_WEIGHTS_FILE):
        print("load cached similarity and attention matrices...")
        similarity_matrix = np.load(SIMILARITY_MATRIX_FILE)
        attention_weights = np.load(ATTENTION_WEIGHTS_FILE)

    else:
        print("cache missing; compute similarity and attention matrices...")
        logging.set_verbosity_error()  # Show errors only.
        similarity_matrix = build_similarity_matrix(source_concepts, target_concepts)
        attention_weights = generate_attention_weights(similarity_matrix, temperature = args.temperature)

        # Save computed matrices to cache.
        np.save(SIMILARITY_MATRIX_FILE, similarity_matrix)
        np.save(ATTENTION_WEIGHTS_FILE, attention_weights)
        print(f"cached results in: {CACHE_DIR}")

    # problem_text = 'Set How many rows will be in the result for the following relational algebra expression?,Which of the following statements are not correct?,Which of the following are used to make access decisions in Mandatory Access Control (MAC)?'
    # print(problem_text)
    # eng_seq = enhance_text_with_attention(source_concepts, target_concepts, attention_weights, problem_text)
    # print(eng_seq)
    print(df_original['seq_problems'].iloc[1])
    print("-" * 50, 'enhance texts with cross-disciplinary attention', "-" * 50)
    df_original['seq_problems'] = df_original['seq_problems'].apply(
        lambda t: process_seq_problems(t, source_concepts, target_concepts, attention_weights,
                                       legacy_scaling=args.legacy_scaling))
    print(df_original['seq_problems'].iloc[1])
    df_original['user'] = df_original['user'].astype(str) + ',' + df_original['seq_len'].astype(str)
    # Save selected columns in block format.
    selected_columns = ['user', 'seq_problems', 'seq_skills', 'seq_ans', 'seq_start_time', 'seq_response_cost']
    current_dir = os.getcwd()
    df_original.to_csv(
        args.output,
        sep = '\n',  # One field per line block.
        columns = selected_columns,  # Save selected columns only.
        header = False,  # Omit the header.
        index = False,  # Omit the index.
        na_rep = 'NA',  # Represent missing values as NA.
        float_format = '%s'  # Avoid scientific notation.
    )

