
import argparse
import logging
import os
import time

import pandas as pd

from LLMs.llm_models.llm_base import load_llm
from utils.dataset_util import get_class_name, get_class_number
from utils.util import (generate_class_specific_qas, generate_common_qas,
                        read_pkl)


def main():
    parser = argparse.ArgumentParser(description="Detect Backdoor")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack", type=str, default="badnet")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--llm", type=str, default="InstructBLIP")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--bd_num_perclass", type=int, default=50)

    args = parser.parse_args()

    num_classes = get_class_number(args.dataset)
    class_names = get_class_name(dataset=args.dataset, num_class=num_classes)

    target_label = args.target_label

    bd_res_file = f"results/vqa_bd_{args.dataset}_{args.attack}_{args.target_label}_{args.llm}_class{args.label}.csv"
    cl_res_file=f"results/vqa_cl_{args.dataset}_{args.target_label}_{args.llm}_class{args.label}.csv"

    bd_img_folder = f"data/{args.dataset}/{args.dataset}_{args.attack}/train_dataset"
    cl_img_folder=f"data/{args.dataset}/{args.dataset}_clean/train_dataset"

    bd_idxs = read_pkl(f"data/data_index/{args.dataset}_bd_idx_{args.bd_num_perclass}.pkl")[args.label] # poisoned data index
    cl_idxs= read_pkl(f"data/data_index/{args.dataset}_cl_idx_{args.bd_num_perclass}.pkl")[args.label] # clean data index

    target_label_name = class_names[target_label]

    common_qas = generate_common_qas(f"prompts/{args.dataset}_common_{args.llm}.csv", target_label_name)
    class_specific_qas = generate_class_specific_qas(f"prompts/{args.dataset}_specific_{args.llm}.csv", target_label)

    class_i = args.label

    gt_label = class_names[class_i]

    print(f">>>>>>>>>>>>> Loading LLM: {args.llm} <<<<<<<<<<<<<<<")
    llm_model = load_llm(args.llm)
    print(f">>>>>>>>>>>>> Finish loading LLM model {args.llm} <<<<<<<<\n")

    print(f"*************** Ground-Truth Class {class_i}, {gt_label} ***************")

    bd_img_folder_class_i = os.path.join(bd_img_folder, str(class_i))
    cl_img_folder_class_i = os.path.join(cl_img_folder, str(class_i))


    bd_df = pd.DataFrame(
        columns=[
            "img_id",
            "gt_label",
            "target_label",
            "llm_model",
            "question",
            "answer",
            "response",
            "LLM_eval",
            "is_match",
        ]
    )
    bd_df.to_csv(bd_res_file, index=True)

    cl_df = pd.DataFrame(
        columns=[
            "img_id",
            "gt_label",
            "target_label",
            "llm_model",
            "question",
            "answer",
            "response",
            "LLM_eval",
            "is_match",
        ]
    )
    cl_df.to_csv(bd_res_file, index=True)



    for num_img_id, idx in enumerate(bd_idxs):
        img_path = os.path.join(bd_img_folder_class_i, f"{idx}.png")
        img_id = f"{args.dataset}_{args.attack}_{args.label}_{idx}.png"
        input_qas = []
        input_qas.extend(common_qas)
        input_qas.extend(class_specific_qas)

        for num_ques_id, input_qa in enumerate(input_qas):
            question = input_qa["question"]
            answer = input_qa["answer"]
       
            if not ((bd_df["img_id"] == img_id) & (bd_df["question"] == question)).any():
                response = llm_model.generate_mm(img_path, question)
                print(f"====================Class:{class_i}, IMG ID:{num_img_id}, QUES ID:{num_ques_id} ===================")
                print(f">>> ImgID: {img_id}")
                print(f">>> Question: {question}")
                print(f">>> Response: {response}")
                print(f"===================================================================\n")

                if response != "":
                    data = [
                        img_id,
                        gt_label,
                        target_label_name,
                        args.llm,
                        question,
                        answer,
                        response,
                        None,
                        None,
                    ]
                    bd_df.loc[len(bd_df)] = data
            else:
                print(f"img_id:{img_id}, question:{question} already exists")

        if num_img_id % 10 == 0:
            bd_df.to_csv(bd_res_file, index=True)
    bd_df.to_csv(bd_res_file, index=True)


    for num_img_id, idx in enumerate(cl_idxs):
        img_path = os.path.join(cl_img_folder_class_i, f"{idx}.png")
        img_id = f"{args.dataset}_clean_{args.label}_{idx}.png"
        input_qas = []
        input_qas.extend(common_qas)
        input_qas.extend(class_specific_qas)

        for num_ques_id, input_qa in enumerate(input_qas):
            question = input_qa["question"]
            answer = input_qa["answer"]

            if not ((cl_df["img_id"] == img_id) & (cl_df["question"] == question)).any():
                response = llm_model.generate_mm(img_path, question)

                print(f"====================Class:{class_i}, IMG ID:{num_img_id}, QUES ID:{num_ques_id} ===================")
                print(f">>> ImgID: {img_id}")
                print(f">>> Question: {question}")
                print(f">>> Response: {response}")
                print(f"===================================================================\n")

                # 向CSV添加一行数据
                if response != "":
                    data = [
                        img_id,
                        gt_label,
                        target_label_name,
                        args.llm,
                        question,
                        answer,
                        response,
                        None,
                        None,
                    ]
                    cl_df.loc[len(cl_df)] = data

            else:
                print(f"img_id:{img_id}, question:{question} already exists")

        if num_img_id % 10 == 0:
            cl_df.to_csv(cl_res_file, index=True)
    cl_df.to_csv(cl_res_file, index=True)



if __name__ == "__main__":
    main()
