import argparse
from wandb_train import main
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="atkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
     
    parser.add_argument("--skill_dim", type=int, default=256)
    parser.add_argument("--answer_dim", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=80)
    parser.add_argument("--attention_dim", type=int, default=80)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--load_dir", type = str,
                        default = 'saved_model_original/assist2015_atkt_qid_saved_model_42_0_0.2_256_96_80_80_10_0.2_0.001_1_1_b37dd76c-747a-44cb-afee-9211ba4f8a32')
    args = parser.parse_args()
    print(os.getcwd())
    params = vars(args)
    main(params)
