import tiktoken
import os
from tqdm import tqdm
import json
import openai

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

openai.api_key = " "  ## insert the openai api key

RDIR = './COLIEE/task1_test_2023/processed'
WDIR = './COLIEE/task1_test_2023/summary_test_2023_txt'
files = os.listdir(RDIR)

for pfile in tqdm(files[:]):
    file_name = pfile.split('.')[0]
    if os.path.exists(os.path.join(WDIR, file_name+'.json')):
        # print(pfile, 'already exists')
        pass
    else:
        # print(pfile, 'does not exist')
        with open(os.path.join(RDIR, pfile), 'r') as f:
            long_text = f.read()
            f.close()
        summary_total = ''
        length = int(len(encoding.encode(long_text))/3500) + 1
        # Loop through each line in the file
        for i in range(length):
            para = long_text[3500*i:3500*(i+1)]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "summerize in 50 words:"+para},
                ]
            )

            summary_text = completion.choices[0].message['content']
            summary_total += ' ' + summary_text
        with open(os.path.join(WDIR, file_name+'.txt'), 'w') as file:
            file.write(summary_total)
            file.close()
print('finish')
    