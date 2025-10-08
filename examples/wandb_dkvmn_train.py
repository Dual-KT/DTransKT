import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="dkvmn")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--dim_s", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--size_m", type=int, default=50)
   
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--load_dir", type = str,
                        default = 'saved_model_original/algebra2005_dkvmn_qid_saved_model_42_0_0.2_200_0.001_50_1_1_0963eeb1-3467-4d83-a4b0-4aa9f23b6ab7')
    args = parser.parse_args()

    params = vars(args)
    main(params)
