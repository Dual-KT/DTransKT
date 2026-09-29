"""Generate attention weights and enhanced texts for transfer."""
import argparse
import os
import numpy as np
from data.CA import (
    read_txt_in_blocks,
    build_similarity_matrix,
    generate_attention_weights,
    process_seq_problems,
)


def collect_concepts(df):
    """Collect unique skill tokens from interaction records."""
    return list({t for s in df["seq_skills"] for t in (s.split(",") if isinstance(s, str) else s)})


def main():
    """Run CA similarity computation and text enhancement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_txt", type=str, required=True)
    parser.add_argument("--target_txt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df_src = read_txt_in_blocks(args.source_txt)
    df_tgt = read_txt_in_blocks(args.target_txt)
    src_concepts = collect_concepts(df_src)
    tgt_concepts = collect_concepts(df_tgt)
    sim = build_similarity_matrix(src_concepts, tgt_concepts)
    weights = generate_attention_weights(sim, temperature=args.temperature)
    df_src["seq_problems"] = df_src["seq_problems"].apply(
        lambda t: process_seq_problems(t, src_concepts, tgt_concepts, weights))
    df_src["user"] = df_src["user"].astype(str) + "," + df_src["seq_len"].astype(str)
    cols = ["user", "seq_problems", "seq_skills", "seq_ans", "seq_start_time", "seq_response_cost"]
    df_src.to_csv(args.output, sep="\n", columns=cols, header=False, index=False, na_rep="NA")


if __name__ == "__main__":
    main()
