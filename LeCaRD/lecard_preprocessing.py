import json
import os
import torch
from tqdm import tqdm


#Load data
query_dict = {}
q_issue_dict = {}
Q_DIR = './LeCaRD/query.json'
with open(Q_DIR, 'r') as f:
    lines = f.readlines()
    for line in lines:
        q_dict = json.loads(line)
        query_dict.update({q_dict['ridx']:q_dict['q']})
        q_issue_dict.update({q_dict['ridx']:q_dict['crime']})
    f.close()

candidate_path = './LeCaRD/candidate/'

can_list = []
candidate_fact_dict = {}

for k,v in tqdm(query_dict.items()):        
    files = os.listdir(candidate_path+str(k)+'/')
    candidate_fact_dict.update({k:query_dict[k]})
    for file in files:
        # file_name = file.split('.')[0]
        file_name = str(k)+'/'+file
        can_list.append(file_name)
        with open (candidate_path+str(k)+'/'+file, 'r') as f:
            doc = json.load(f)
            case_txt = doc['ajjbqk']
            if '经审理查明' in case_txt:
                fact_0 = case_txt.split('经审理查明')[1]
                if '上述事实' in fact_0:
                    fact_0 = fact_0.split('上述事实')[0]
            elif '上述事实' in case_txt:
                fact_0 = case_txt.split('上述事实')[0]
            else:
                fact_0 = case_txt
        if 20 <len(fact_0) < 150:
            candidate_fact_dict.update({file_name:case_txt})
        elif len(case_txt) <= 20:
            candidate_fact_dict.update({file_name:doc['cpfxgc']})
        else:
            candidate_fact_dict.update({file_name:fact_0})

with open('./LeCaRD/candidate_fact_dict.json', 'w') as f:
    json.dump(candidate_fact_dict, f, ensure_ascii=False)
    f.close()

with open('./LeCaRD/issues_name.txt', 'r') as f:
    issues  = f.readlines()
    f.close()


issue_name_list = []
for issue in issues:
    issue_1 = issue.split('罪\n')[0]
    if '、' in issue_1:
        issue_2 = issue_1.split('、')
        for issue in issue_2:
            issue_name_list.append(issue)
    else:
        issue_name_list.append(issue_1.split('罪\n')[0])

query_label_dict = {}
with open('./LeCaRD/golden_labels.json', 'r') as f:
    lines = f.readlines()
    for line in lines:
        q_dict = json.loads(line)
        query_label_dict.update(q_dict)
    f.close()

issue_dict = {}
for k,v in tqdm(query_label_dict.items()): 
    files = os.listdir(candidate_path+str(k)+'/')
    q_issue_txt = ''
    for value in q_issue_dict[int(k)]:
        if '、' in value:
            value = value.split('罪\n')[0]
            values = value.split('、')
            for value_1 in values:
                if value_1 in issue_name_list:
                    q_issue_txt += value_1+ ' ' 
        else:
            q_issue_txt += value.split('罪\n')[0]+ ' '        
    issue_dict.update({k:q_issue_txt})
    for file in files:

        file_issue_txt = ''
        file_name = str(k)+'/'+file  
        with open (candidate_path+str(k)+'/'+file, 'r') as f:
            doc = json.load(f)
            aj_topic = doc['ajName']
            f.close()       
        for i in issue_name_list:
            if i in aj_topic:
                file_issue_txt += i+' '
        
        if len(file_issue_txt) == '':
            for i in issue_name_list:
                if i in doc['ajjbqk']:
                    file_issue_txt += i+' '

        if len(file_issue_txt) == '':
            for i in issue_name_list:
                if i in doc['cpfxgc']:
                    file_issue_txt += i+' '
        
        if len(file_issue_txt) == '':
            for i in issue_name_list:
                if i in doc['pjjg']:
                    file_issue_txt += i+' '
        
        if len(file_issue_txt) == '':
            print(k, file_name, 'No issue')

        issue_dict.update({file_name:file_issue_txt})

with open('./LeCaRD/candidate_issue_dict.json', 'w') as f:
    json.dump(issue_dict, f, ensure_ascii=False)
    f.close()

print(len(candidate_fact_dict))
print(len(issue_dict))
print('Finished.')

