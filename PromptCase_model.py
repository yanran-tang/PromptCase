import torch
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel

import json
import os
from tqdm import tqdm
import time
import pickle
import sys
sys.path.append('.')

import torch_metrics

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Training config
parser.add_argument("--dataset", type=str, default='lecard',
                    help="lecard, coliee")
parser.add_argument("--stage_num", type=int, default='1',
                    help="1, 2")
parser.add_argument('--model', type=str, default='SAILER')
args = parser.parse_args()
print(args)

#Micro Precision Function
def micro_prec(true_list,pred_list,k):
    #define list of top k predictions
    cor_pred = 0
    top_k_pred = pred_list[0:k].copy()
    #iterate throught the top k predictions
    for doc in top_k_pred:
        #if document in true list, then increment count of relevant predictions
        if doc in true_list:
            cor_pred += 1
    #return total_relevant_predictions_in_top_k/k
    return cor_pred, k  

##Micro Precision Function with year filter
with open('./COLIEE/task1_test_2023/test_2023_candidate_with_yearfilter.json', 'r') as f:
    candidate_list = json.load(f)
    f.close()

def micro_prec_datefilter(query_case, true_list, pred_list, k):
    cor_pred = 0
    num = 0
    can_list = []
    for i in pred_list:
        if i in candidate_list[query_case]:
            can_list.append(i)
    for doc in can_list:
        if num == k:
            break
        else:
            num += 1   
            if doc in true_list:
                cor_pred += 1
    return cor_pred, num

# #Load data
if args.dataset == 'coliee':
    model_name = 'CSHaitao/SAILER_en_finetune'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    RDIR_sum = './COLIEE/task1_test_2023/summary_test_2023_txt'
    RDIR_refer_sen = './COLIEE/task1_test_2023/processed_new'
    files = os.listdir(RDIR_sum)

    #load first stage list
    top_10_list = {}
    with open('./COLIEE/PromptCase_coliee2023_BM25_prediction_dict.json', 'r')as file:
        top_10_list = json.load(file)
        file.close()

    with open('./COLIEE/task1_test_2023/task1_test_labels_2023.json', 'r')as file:
        query_list = json.load(file)
        file.close()   

elif args.dataset == 'lecard':
    model_name = 'CSHaitao/SAILER_zh'
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    query_fact_dict = {}
    query_crime_dict = {}
    Q_DIR = './LeCaRD/query.json'
    with open(Q_DIR, 'r') as f:
        lines = f.readlines()
        for line in lines:
            q_dict = json.loads(line)
            query_fact_dict.update({q_dict['ridx']:q_dict['q']})
        f.close()
    
    candidate_path = './LeCaRD/candidate/'
    
    #load first stage list
    with open('./LeCaRD/PromptCase_lecard_BM25_prediction_dict.json', 'r')as file:
        top_10_list = json.load(file)
        file.close()
    
    with open('./LeCaRD/lecard_golden_labels.json', "r") as fOut:
        query_list = json.load(fOut)


print(args.dataset)
print(model_name)

## case representation calculation
embedding_dict = {}
ref_embedding_dict = {}
model.eval()
with torch.no_grad():
    fact_embedding_dict = {}
    issue_embedding_dict = {}
    cross_embedding_dict = {}
    sum_embedding_dict = {}
    if args.dataset == 'coliee':        
        prompt_text = "Legal facts/issues:"
        print(prompt_text)

        for pfile in tqdm(files[:]):
            file_name = pfile.split('.')[0]+'.txt'
            # long_text = ''
            with open(os.path.join(RDIR_sum, pfile), 'r') as f:
                original_sum_text = f.read()
                f.close()
            with open(os.path.join(RDIR_refer_sen, pfile), 'r') as file:
                original_refer_text = file.read()
                file.close()
            fact_text = "Legal facts:"+original_sum_text
            issue_text = "Legal issues:"+original_refer_text        
            cross_text = fact_text+' '+issue_text

            ## dual encoding
            fact_tokenized_id = tokenizer(fact_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            fact_embedding = model(**fact_tokenized_id)
            fact_cls_embedding = fact_embedding[0][:,0] ##cls token embedding [1,768]                               
            fact_embedding_dict.update({file_name:fact_cls_embedding})

            issue_tokenized_id = tokenizer(issue_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            issue_embedding = model(**issue_tokenized_id)
            issue_cls_embedding = issue_embedding[0][:,0] ##cls token embedding [1,768]             
            issue_embedding_dict.update({file_name:issue_cls_embedding})
            

            ## cross encoding
            cross_tokenized_id = tokenizer(cross_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            cross_embedding = model(**cross_tokenized_id)
            cross_cls_embedding = cross_embedding[0][:,0] ##cls token embedding [1,768]                           
            cross_embedding_dict.update({file_name:cross_cls_embedding})
            
    elif args.dataset == 'lecard':
        prompt_text = '法律事实/纠纷：'
        print(prompt_text)

        case_fact_dict = {}
        with open('./LeCaRD/candidate_fact_dict.json', "rb") as fIn:
            lines = fIn.readlines()
            for line in lines:
                dict = json.loads(line)
                case_fact_dict.update(dict)
            f.close()
        case_issue_dict = {}
        with open('./LeCaRD/candidate_issue_dict.json', "rb") as fIn:
            lines = fIn.readlines()
            for line in lines:
                dict = json.loads(line)
                case_issue_dict.update(dict)
            f.close()
        
        for k,v in tqdm(query_fact_dict.items()):
            fact = '法律事实：'+query_fact_dict[k]
            issue = '法律纠纷：'+case_issue_dict[str(k)]
            cross_text = fact+' '+issue

            fact_tokenized_id = tokenizer(fact, return_tensors="pt", padding=True, truncation=True, max_length=512)
            fact_embedding = model(**fact_tokenized_id)
            fact_cls_embedding = fact_embedding[0][:,0] ##cls token embedding [1,768]                  
            fact_embedding_dict.update({str(k):fact_cls_embedding})
            
            issue_tokenized_id = tokenizer(issue, return_tensors="pt", padding=True, truncation=True, max_length=512)
            issue_embedding = model(**issue_tokenized_id)
            issue_cls_embedding = issue_embedding[0][:,0] ##cls token embedding [1,768]
            issue_embedding_dict.update({str(k):issue_cls_embedding})

            cross_tokenized_id = tokenizer(cross_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            cross_embedding = model(**cross_tokenized_id)
            cross_cls_embedding = cross_embedding[0][:,0] ##cls token embedding [1,768]                           
            cross_embedding_dict.update({str(k):cross_cls_embedding})
                            
            can_list = []
            files = os.listdir(candidate_path+str(k)+'/')
            for file in files:
                file_name = str(k)+'/'+file

                fact = '法律事实：'+case_fact_dict[file_name]
                case_fact_tokenized_id = tokenizer(fact, return_tensors="pt", padding=True, truncation=True, max_length=512)
                fact_embedding = model(**case_fact_tokenized_id)
                fact_cls_embedding = fact_embedding[0][:,0] ##cls token embedding [1,768]                             
                fact_embedding_dict.update({file_name:fact_cls_embedding})

                issue = '法律纠纷：'+case_issue_dict[file_name]
                case_issue_tokenized_id = tokenizer(issue, return_tensors="pt", padding=True, truncation=True, max_length=512)
                issue_embedding = model(**case_issue_tokenized_id)
                issue_cls_embedding = issue_embedding[0][:,0] ##cls token embedding [1,768]
                issue_embedding_dict.update({file_name:issue_cls_embedding})
                
                cross_text = fact+' '+issue
                
                cross_tokenized_id = tokenizer(cross_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
                cross_embedding = model(**cross_tokenized_id)
                cross_cls_embedding = cross_embedding[0][:,0] ##cls token embedding [1,768]                           
                cross_embedding_dict.update({file_name:cross_cls_embedding})

## Case representation finished

## Case similarity start
## Compute 1-stage cosine-similarities
if args.stage_num == 1:        
    if args.dataset == 'coliee':         
        result_dict = {}       
        for k,v in tqdm(query_list.items()):
            dot_score_dict = {}
            q_cls_fact_embedding = fact_embedding_dict[k] 
            q_cls_issue_embedding = issue_embedding_dict[k] 
            q_cross_embedding = cross_embedding_dict[k]

            for file in files:
                c_cls_fact_embedding = fact_embedding_dict[file]
                c_cls_issue_embedding = issue_embedding_dict[file]
                c_cross_embedding = cross_embedding_dict[file]

                fact_dot_scores = util.dot_score(q_cls_fact_embedding, c_cls_fact_embedding)
                issue_dot_scores = util.dot_score(q_cls_issue_embedding, c_cls_issue_embedding)
                dot_scores = util.dot_score(q_cross_embedding, c_cross_embedding)

                dot_score_dict.update({file:fact_dot_scores+issue_dot_scores+dot_scores})
            
            sorted_dic = sorted(dot_score_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_dic_name = [sorted_dic[i][0] for i in range(len(sorted_dic))]
            result_dict.update({k:sorted_dic_name[1:]})                

        ## save the result                
        time_0 = time.strftime("%m%d-%H%M%S") 
        predict_path = './'+str(args.stage_num)+'stage_coliee_'+time_0       
        with open(predict_path+'.json' , "w") as fOut:
            json.dump(result_dict, fOut)
            fOut.close()
        
    elif args.dataset == 'lecard':
        result_dict = {}
        for k,v in tqdm(query_list.items()): 
            dot_score_dict = {}

            q_cls_fact_embedding = fact_embedding_dict[k] 
            q_cls_issue_embedding = issue_embedding_dict[k] 
            q_cross_embedding = cross_embedding_dict[k]

            files = os.listdir(candidate_path+str(k)+'/')
            for file in files:
                file_name = str(k)+'/'+file
                c_cls_fact_embedding = fact_embedding_dict[file_name] 
                c_cls_issue_embedding = issue_embedding_dict[file_name] 
                c_cross_embedding = cross_embedding_dict[file_name]

                fact_dot_scores = util.dot_score(q_cls_fact_embedding, c_cls_fact_embedding)
                issue_dot_scores = util.dot_score(q_cls_issue_embedding, c_cls_issue_embedding)
                cross_dot_scores = util.dot_score(q_cross_embedding, c_cross_embedding)

                dot_score_dict.update({file.split('.')[0]:(fact_dot_scores+issue_dot_scores+cross_dot_scores)})

            sorted_dic = sorted(dot_score_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_dic_name = [sorted_dic[i][0] for i in range(len(sorted_dic))]
            result_dict.update({k:sorted_dic_name})
        
        ## save the result                
        time_0 = time.strftime("%m%d-%H%M%S") 
        predict_path = './'+str(args.stage_num)+'stage_lecard_'+time_0          
        with open(predict_path+'.json' , "w") as fOut:
            json.dump(result_dict, fOut)
            fOut.close()


## Compute 2-stage cosine-similarities
elif args.stage_num == 2:
    if args.dataset == 'coliee':
        result_dict = {}
        for k,v in query_list.items():
            dot_score_dict = {}
            q_cls_fact_embedding = fact_embedding_dict[k] 
            q_cls_issue_embedding = issue_embedding_dict[k]
            q_cross_embedding = cross_embedding_dict[k]

            for value in top_10_list[k][1:11]:
                c_cls_fact_embedding = fact_embedding_dict[value] 
                c_cls_issue_embedding = issue_embedding_dict[value] 
                c_cross_embedding = cross_embedding_dict[value]

                fact_dot_scores = util.dot_score(q_cls_fact_embedding, c_cls_fact_embedding)
                issue_dot_scores = util.dot_score(q_cls_issue_embedding, c_cls_issue_embedding)
                cross_dot_scores = util.dot_score(q_cross_embedding, c_cross_embedding)

                dot_score_dict.update({value:(fact_dot_scores+issue_dot_scores+cross_dot_scores)})
            sorted_dic = sorted(dot_score_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_dic_name = [sorted_dic[i][0] for i in range(len(sorted_dic))]
            result_dict.update({k:sorted_dic_name})
            
        ## save the result                
        time_0 = time.strftime("%m%d-%H%M%S") 
        predict_path = './'+str(args.stage_num)+'stage_coliee_'+time_0          
        with open(predict_path+'.json' , "w") as fOut:
            json.dump(result_dict, fOut)
            fOut.close()

    
    elif args.dataset == 'lecard':
        result_dict = {}
        for k,v in query_list.items():
            dot_score_dict = {}
            q_cls_fact_embedding = fact_embedding_dict[k]
            q_cls_issue_embedding = issue_embedding_dict[k] 
            q_cross_embedding = cross_embedding_dict[k]

            for value in top_10_list[k][0:10]:
                file_name = value.split('/')[-1].split('.')[0]
                c_cls_fact_embedding = fact_embedding_dict[value] 
                c_cls_issue_embedding = issue_embedding_dict[value]                    
                c_cross_embedding = cross_embedding_dict[value]

                fact_dot_scores = util.dot_score(q_cls_fact_embedding, c_cls_fact_embedding)
                issue_dot_scores = util.dot_score(q_cls_issue_embedding, c_cls_issue_embedding)
                cross_dot_scores = util.dot_score(q_cross_embedding, c_cross_embedding)
                dot_score_dict.update({file_name:(fact_dot_scores+issue_dot_scores+cross_dot_scores)})
            
            sorted_dic = sorted(dot_score_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_dic_name = [sorted_dic[i][0] for i in range(len(sorted_dic))]
            result_dict.update({k:sorted_dic_name})
            
        ## save the result                
        time_0 = time.strftime("%m%d-%H%M%S") 
        predict_path = './'+str(args.stage_num)+'stage_lecard_'+time_0          
        with open(predict_path+'.json' , "w") as fOut:
            json.dump(result_dict, fOut)
            fOut.close()

 
##evaluation start
correct_pred = 0
retri_cases = 0
relevant_cases = 0
cls_pre = 0
cls_recall = 0
for i in tqdm(result_dict.keys()):
    query_case = i
    true_list = query_list[i]
    topk = 5 ## top 5
    pred_list = result_dict[i]
    if args.dataset == 'lecard':
        c_p, r_c = micro_prec(true_list, pred_list, topk)
    elif args.dataset == 'coliee':
        c_p, r_c = micro_prec_datefilter(query_case, true_list, pred_list, topk)
    correct_pred += c_p
    retri_cases += r_c
    relevant_cases += len(true_list)

    ## macro precision
    if c_p > 0:
        cls_pre += c_p/topk
        cls_recall += c_p/len(true_list)
    else:
        cls_pre += 0
        cls_recall += 0

## Metrics
Micro_pre = correct_pred/retri_cases
Micro_recall = correct_pred/relevant_cases
Micro_F = 2*Micro_pre*Micro_recall/ (Micro_pre + Micro_recall)

macro_pre = cls_pre/len(result_dict.keys())
macro_recall = cls_recall/len(result_dict.keys())
macro_F = 2*macro_pre*macro_recall/ (macro_pre + macro_recall)

ndcg_score, mrr_score, map_score, p_score = torch_metrics.t_metrics(args.dataset, predict_path+'.json')

if args.dataset == 'coliee':
    with open(predict_path+'.txt', "w") as fOut:
        fOut.write("Correct Predictions: "+str(correct_pred)+'\n')
        fOut.write("Retrived Cases: "+str(retri_cases)+'\n')
        fOut.write("Relevant Cases: "+str(relevant_cases)+'\n')
        
        fOut.write("Micro Precision: "+str(Micro_pre)+'\n')
        fOut.write("Micro Recall: "+str(Micro_recall)+'\n')
        fOut.write("Micro F-Measure: "+str(Micro_F)+'\n')
        
        fOut.write("Macro Precision: "+str(macro_pre)+'\n')
        fOut.write("Macro Recall: "+str(macro_recall)+'\n')
        fOut.write("Macro F-Measure: "+str(macro_F)+'\n')

        fOut.write("NDCG@5: "+str(ndcg_score)+'\n')
        fOut.write("MRR@5: "+str(mrr_score)+'\n')
        fOut.write("MAP: "+str(map_score)+'\n')

        fOut.write('Prompts: '+prompt_text)

        fOut.close()
elif args.dataset == 'lecard':
    with open(predict_path+'.txt', "w") as fOut:
        fOut.write("Correct Predictions: "+str(correct_pred)+'\n')
        fOut.write("Retrived Cases: "+str(retri_cases)+'\n')
        fOut.write("Relevant Cases: "+str(relevant_cases)+'\n')
        
        fOut.write("Micro Precision: "+str(Micro_pre)+'\n')
        fOut.write("Micro Recall: "+str(Micro_recall)+'\n')
        fOut.write("Micro F-Measure: "+str(Micro_F)+'\n')
        
        fOut.write("Macro Precision: "+str(macro_pre)+'\n')
        fOut.write("Macro Recall: "+str(macro_recall)+'\n')
        fOut.write("Macro F-Measure: "+str(macro_F)+'\n')

        fOut.write("NDCG@5: "+str(ndcg_score)+'\n')
        fOut.write("MRR@5: "+str(mrr_score)+'\n')
        fOut.write("MAP: "+str(map_score)+'\n')

        fOut.write('Prompts: '+prompt_text)

        fOut.close()

print("Correct Predictions: ", correct_pred)
print("Retrived Cases: ", retri_cases)
print("Relevant Cases: ", relevant_cases)

print("Micro Precision: ", Micro_pre)
print("Micro Recall: ", Micro_recall)
print("Micro F-Measure: ", Micro_F)

print("Macro Precision: ", macro_pre)
print("Macro Recall: ", macro_recall)
print("Macro F-Measure: ", macro_F)

print("NDCG@5: ", ndcg_score)
print("MRR@5: ", mrr_score)
print("MAP: ", map_score)

print('1')