import time

import pandas as pd
import urllib3

from LLMs.llm_models.openai_api_pool import get_openai_api
from utils.gpt import GPT
from utils.nlp_util import parse_entity

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

number = {
    "0": "zero none nothing no",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def replace_numbers(text):
    tokens = text.split()
    for i in range(len(tokens)):
        if tokens[i] in number.keys():
            tokens[i] = number[tokens[i]]
    return " ".join(tokens)


def eval_common_qa(response, answer, llm="gpt"):
    if llm == "gpt":
        output = eval_gpt(response, answer)
    if "yes" in output:
        is_match = True
    elif "no" in output:
        is_match = False
    else:
        is_match = None

    return output, is_match


def eval_class_specific_qa(response, answer):
    # 判断answer是否在response中
    response = response.lower()
    answer = answer.lower()
    response = replace_numbers(response)
    answer = replace_numbers(answer)
    answer_tokens = answer.split()
    for answer_token in answer_tokens:
        if answer_token in response:
            return True
    return False


def classification_acc(syn_list, pred_text):
    words = parse_entity(pred_text)
    for syn in syn_list:
        if syn in words:
            return True
    return False


def eval_image(
    img_id,
    common_qas,
    class_specific_qas,
    df,
    answer,
    syn_answer_list,
    target_label,
    llm="gpt"
):
    # common questions
    num_common_ques = 0
    num_common_ques_true = 0
    for qa in common_qas:
        question = qa["question"]
        if ((df["img_id"] == img_id) & (df["question"] == question) & (df["target_label"] == target_label)).any():
            row_index = df[(df["img_id"] == img_id) & (df["question"] == question) & (df["target_label"] == target_label)].index[-1]
            response = df.loc[row_index, "response"]

            if pd.isna(df.loc[row_index, "is_match"]):
                is_match = None
            else:
                is_match = df.loc[row_index, "is_match"]

            if pd.isna(df.loc[row_index, "LLM_eval"]):
                LLM_eval = None
            else:
                LLM_eval = df.loc[row_index, "LLM_eval"]

            if is_match is None:
                words = parse_entity(response)
                for syn in syn_answer_list:
                    if syn in words:
                        is_match = True
                        break

            if is_match is None and (LLM_eval is None or LLM_eval == "##ERROR##"):
                llm_output, is_match = eval_common_qa(response=response, answer=answer, llm=llm)
                df.loc[row_index, "LLM_eval"] = llm_output

            df.loc[row_index, "answer"] = answer
            df.loc[row_index, "is_match"] = is_match

            num_common_ques += 1
            if is_match:
                num_common_ques_true += 1
        else:
            print(f"Error: img_id:{img_id}, question:{question} not exists")

    # class-specific questions
    num_specific_ques = 0
    num_specific_ques_true = 0
    for qa in class_specific_qas:
        question = qa["question"]
        answer = qa["answer"]
        if ((df["img_id"] == img_id) & (df["question"] == question) & (df["target_label"] == target_label)).any():
            row_index = df[(df["img_id"] == img_id) & (df["question"] == question) & (df["target_label"] == target_label)].index[-1]
            response = df.loc[row_index, "response"]

            # 判断 is_match 是否为nan
            if pd.isna(df.loc[row_index, "is_match"]):
                is_match = None
            else:
                is_match = df.loc[row_index, "is_match"]

            if is_match is None:
                is_match = eval_class_specific_qa(response=response, answer=answer)

            df.loc[row_index, "answer"] = answer
            df.loc[row_index, "is_match"] = is_match

            num_specific_ques += 1
            if is_match:
                num_specific_ques_true += 1


    if num_common_ques + num_specific_ques == 0:
        print(f"Error: num_common_ques+num_specific_ques==0")
    ques_right_rate = (num_common_ques_true + num_specific_ques_true) / (num_common_ques + num_specific_ques)

    return ques_right_rate, df


def eval_gpt(response, answer):
    prompt = f"""Assume you are a helpful and precise assistant for evaluation. Please judge whether the 'Caption' of an image and one of the 'Labels' refer to the same object. Answer with yes or no.
- Caption: [{response}]
- Labels: [{answer}]"""

    success = False
    try_times = 0
    max_try_times = 100
    response = ""
    while not success and try_times < max_try_times:
        try:
            url, api_key = get_openai_api()
            try_times += 1
            if try_times % 10 == 0:
                print(f"try {try_times} times")
            flag, response = GPT("gpt-3.5-turbo").generate(url=url, api_key=api_key, new_message=prompt, role=None, args=None)
            if flag:
                success = True
        except Exception as e:
            print(e)
            time.sleep(3)
            print("Error: try again")
    if not success:
        response = "##ERROR##"
        print(f"===========================================Error: try 100 times==============================================")

    return response.lower()


