

import argparse
import logging
import os
import time

import nltk
import pandas as pd

nltk.download("stopwords")
import multiprocessing as mp
from multiprocessing import Manager, Process

from utils.dataset_util import get_class_name, get_class_number
from utils.nlp_util import class2list, parse_entity, tokenize_list_of_names
from utils.util import (generate_class_specific_qas, generate_common_qas,
                        read_pkl, save_pkl)
from utils.vae_util import eval_image


def classification_acc(syn_list, pred_text):
    words = parse_entity(pred_text)
    for syn in syn_list:
        if syn in words:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Detect Backdoor")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack", type=str, default="badnet")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--bd_num_perclass", type=int, default=50)
    parser.add_argument("--llm", type=str, default="llava")
    parser.add_argument("--start_label", type=int, default=0) # start label
    parser.add_argument("--end_label", type=int, default=10)    # end label
    parser.add_argument("--num_process", type=int, default=10)    # number of process 

    args = parser.parse_args()


    num_classes = get_class_number(args.dataset)
    class_names = get_class_name(dataset=args.dataset, num_class=num_classes)

    vae_res_path = f"results/vae_bd_{args.dataset}/{args.attack}_{args.target_label}_{args.llm}_{args.bd_num_perclass}.csv"

    select_clean_indices_path = (
        f"results/select_cl_index/{args.attack}_{args.target_label}_{args.llm}_{args.bd_num_perclass}.pkl"
    )

    if not os.path.exists(vae_res_path):
        df = pd.DataFrame(columns=["class_i", "bd2bd", "bd2cl", "cl2bd", "cl2cl", "TPR", "FPR", "F1"])
        df.to_csv(vae_res_path, index=False)
    else:
        df = pd.read_csv(vae_res_path)

    res_lock = mp.Manager().Lock()

    with Manager() as manager:
        select_clean_indices = manager.list()
        pool = mp.Pool(processes=args.num_process)
        for class_i in range(args.start_label, args.end_label):
            if class_i in df["class_i"].values.astype(int).tolist():
                continue

            pool.apply_async(eval_class_i, (args, class_i, class_names, vae_res_path, select_clean_indices, res_lock))

        pool.close()
        pool.join()

        # ================== calculate overall bd2bd, bd2cl, TPR and FPR ==================
        df = pd.read_csv(vae_res_path)
        class_i_list = df["class_i"].values.tolist()
        if "overall" not in class_i_list and len(class_i_list) == num_classes:
            cl2bd = df["cl2bd"].sum()
            cl2cl = df["cl2cl"].sum()
            bd2bd = df["bd2bd"].sum()
            bd2cl = df["bd2cl"].sum()
            TPR = bd2bd / (bd2bd + bd2cl)
            FPR = cl2bd / (cl2bd + cl2cl)
            Precious = bd2bd / (bd2bd + cl2bd)
            F1 = 2 * Precious * TPR / (Precious + TPR)
            df.loc[df.shape[0]] = ["overall", bd2bd, bd2cl, cl2bd, cl2cl, TPR, FPR, F1]
            df.to_csv(vae_res_path, index=False)
            logging.info(f"Overall done!")
            save_pkl(list(select_clean_indices), select_clean_indices_path)


def eval_class_i(args, class_i, class_names, csv_res_path, select_clean_indices, res_lock):
    select_clean_idxs = []
    class_bd_to_cl = 0
    class_bd_to_bd = 0  
    class_cl_to_bd = 0
    class_cl_to_cl = 0
    class_total_num = 0

    all_indices = read_pkl(f"data/data_index/{args.dataset}_all_idx.pkl")
    bd_indices = read_pkl(f"data/data_index/{args.dataset}_bd_idx_{args.bd_num_perclass}.pkl")
    cl_indices = read_pkl(f"data/data_index/{args.dataset}_cl_idx_{args.bd_num_perclass}.pkl")

    all_idxs = all_indices[class_i]
    bd_idxs = bd_indices[class_i]
    cl_idxs = cl_indices[class_i]

    bd_class_specific_qas = generate_class_specific_qas(f"prompts/{args.dataset}_specific_{args.llm}.csv", args.target_label)

    bd_common_qas = generate_common_qas(f"prompts/{args.dataset}_common_{args.llm}.csv", class_names[args.target_label])
    cl_common_qas = generate_common_qas(f"prompts/{args.dataset}_common_{args.llm}.csv", class_names[class_i])

    bd_class_specific_qas = generate_class_specific_qas(f"prompts/{args.dataset}_specific_{args.llm}.csv", args.target_label)
    cl_class_specific_qas = generate_class_specific_qas(f"prompts/{args.dataset}_specific_{args.llm}.csv", class_i)

    bd_res_file = f"results/vqa_bd_{args.dataset}_{args.attack}_{args.target_label}_{args.llm}_class{args.label}.csv"
    bd_df = pd.read_csv(bd_res_file, index_col=0)

    cl_res_file = f"results/vqa_cl_{args.dataset}_{args.target_label}_{args.llm}_class{args.label}.csv"
    cl_df = pd.read_csv(cl_res_file, index_col=0)

    bd_syn_answer_list = tokenize_list_of_names(class2list(class_names[args.target_label]))
    cl_syn_answer_list = tokenize_list_of_names(class2list(class_names[class_i]))

    for i, idx in enumerate(all_idxs):
        start_time = time.time()
        if idx in bd_idxs:
            img_id = f"{args.dataset}_{args.attack}_{class_i}_{idx}.png"

            ques_right_rate, bd_df = eval_image(
                img_id=img_id,
                common_qas=bd_common_qas,
                class_specific_qas=bd_class_specific_qas,
                df=bd_df,
                answer=class_names[args.target_label],
                target_label=class_names[args.target_label],
                syn_answer_list=bd_syn_answer_list,
                llm="gpt",
            )
            if ques_right_rate <= 0.5:
                class_bd_to_bd += 1
            else:
                class_bd_to_cl += 1
                select_clean_idxs.append(idx)
            class_total_num += 0

        elif idx in cl_idxs:
            img_id = f"{args.dataset}_clean_{class_i}_{idx}.png"

            ques_right_rate, cl_df = eval_image(
                img_id=img_id,
                common_qas=cl_common_qas,
                class_specific_qas=cl_class_specific_qas,
                df=cl_df,
                answer=class_names[class_i],
                syn_answer_list=cl_syn_answer_list,
                llm="gpt",
                target_label=class_names[class_i],
            )

            if ques_right_rate < 0.5:
                class_cl_to_bd += 1

            else:
                class_cl_to_cl += 1
                select_clean_idxs.append(idx)
        else:
            raise Exception(f"{idx} not exist")

        end_time = time.time()

        logging.info(f"Class {class_i} ID {i} : rest time: {(end_time - start_time) * (len(all_idxs) - i) / 60}min")
        if i % 10 == 0:
            bd_df.to_csv(bd_res_file, index=True)
            cl_df.to_csv(cl_res_file, index=True)

    bd_df.to_csv(bd_res_file, index=True)
    cl_df.to_csv(cl_res_file, index=True)
    class_TPR = class_bd_to_bd / (class_bd_to_bd + class_bd_to_cl) if class_bd_to_bd + class_bd_to_cl != 0 else 0
    class_FPR = class_cl_to_bd / (class_cl_to_cl + class_cl_to_bd) if class_cl_to_cl + class_cl_to_bd != 0 else 0
    class_precious = class_bd_to_bd / (class_bd_to_bd + class_cl_to_bd) if class_bd_to_bd + class_bd_to_cl != 0 else 0
    class_F1 = 2 * class_TPR * class_precious / (class_TPR + class_precious) if class_TPR + class_precious != 0 else 0
    
    with res_lock:
        res_csv = pd.read_csv(csv_res_path)
        res_csv.loc[res_csv.shape[0]] = [int(class_i), class_bd_to_bd, class_bd_to_cl, class_cl_to_bd, class_cl_to_cl, class_TPR, class_FPR, class_F1]
        res_csv = res_csv.sort_values(by="class_i", ascending=True)
        res_csv.to_csv(csv_res_path, index=False)
        select_clean_indices.extend(select_clean_idxs)

    logging.info(f"Class {class_i} TPR: {class_TPR}")


if __name__ == "__main__":
    main()
