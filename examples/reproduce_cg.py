"""Train CG and generate mapped inputs for KT models."""
import argparse
import os
from data.CG import StudentMatcher, read_txt_in_blocks, create_similar_students_dataset


def main():
    """Run CG training, retrieval, and mapped dataset generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_txt", type=str, required=True)
    parser.add_argument("--target_txt", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--save_dir", type=str, default="./GAP_model_ad")
    args = parser.parse_args()

    df_source = read_txt_in_blocks(args.source_txt)
    df_target = read_txt_in_blocks(args.target_txt)
    matcher = StudentMatcher(feature_dim=3)
    matcher.train(df_source, df_target, epochs=args.epochs)
    matcher.save_model(args.save_dir)
    ids = df_source["user"].tolist()
    mapped = create_similar_students_dataset(df_target, matcher, ids, top_k=args.top_k)
    print(mapped.info())


if __name__ == "__main__":
    main()
