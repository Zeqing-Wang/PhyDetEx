
JUDGE_PROMPT = """
You will be given a ground truth and model output couple.
Your task is to provide a 'total rating' scoring how well the model output matches the semantic meaning of the ground truth.
Give your answer as an integer on a scale of 0 to 5, where 0 means that the model output is completely unrelated to the ground truth, and 5 means that the model output perfectly matches the semantic meaning of the ground truth.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as an integer between 0 and 5)

Now here are the ground truth and model output.

Ground Truth: {ground_truth}
Model Output: {model_output}

Feedback:::
Total rating: """
import os
import json
import re
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from huggingface_hub import InferenceClient, notebook_login
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
def request_single(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': prompt,
            }],
        }],
        temperature=0.8,
        top_p=0.8)
    return response.choices[0].message.content

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None
    
def judge_two_sentence(label, output):
    JUDGE_PROMPT_SEND = JUDGE_PROMPT.format(ground_truth = label, model_output = output)
    res = request_single(prompt=JUDGE_PROMPT_SEND)
    res = extract_judge_score(answer=res)
    return res
    pass



# Test your LLM client
if __name__ == '__main__':
    anno_json_path = "./Data/ImpossibleVideos/openqa_answer.json"
    judge_file = './res/res_on_impossible_videos.json'
    


    anno_json = json.load(open(anno_json_path, "r"))
    res = json.load(open(judge_file, "r"))
    res_keys = list(res['imp_videos_res'].keys())

    output_path = judge_file.replace('.json','_judged_with_lmdeploy_llama3_1_8b.json')
    res_output = []


    for r_key in tqdm(res_keys):
        pred_res = res['imp_videos_res'][r_key]
        ann_gt = anno_json[r_key]['primary_answer']

        if pred_res.startswith(("Yes. ", "No. ")):
            processed_res = pred_res.replace('Yes. ','')
            processed_res = pred_res.replace('No.','')
        else:

            processed_res = pred_res
            pass


        answer_score = judge_two_sentence(label=ann_gt, output=processed_res)
        r = {}
        r['video_name'] = r_key
        r['answer_label'] = ann_gt
        r['answer_processed_pred'] = processed_res
        r['answer_pred'] = pred_res
        r['score'] = answer_score


        res_output.append(r)
    
    # cal the res
    score_total = 0
    for res_o in res_output:
        score_total += res_o['score']
    

    report_dict = {
        'imp_videos' : res_output,
        'avg_scores' : score_total / len(res_output)
    }
        

    json.dump(report_dict, open(output_path, "w"))
    pass

