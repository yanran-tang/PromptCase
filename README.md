# PromptCase
Code for ADC 2023 paper submission.

Title: Prompt-based Effective Input Reformulation for Legal Case Retrieval

Author: Yanran Tang, Ruihong Qiu and Xue Li

# Dataset
1. For LeCaRD dataset, download datasets from [LeCaRD](https://github.com/myx666/LeCaRD) and put the files `query.json`, `lecard_golden_labels.json`, `stopword.txt`in `./LeCaRD/` like the following. All of the cases from folders of candidates1 and cndidates2 of LeCaRD are extracted and put in the `candidate` folder as follows:

```
$ ./LeCaRD/
.
├── candidate
│   ├── -741
│   ├── -991
│   ├── -1071
│   └── ...
├── issues_name.txt
├── lecard_golden_labels.json
├── lecard_preprocessing.py
├── PromptCase_lecard_BM25_prediction_dict.json
├── query.json
└── stopword.txt
```
2. For COLIEE2023 dataset, download datasets from [COLIEE2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/) and put the folder `task1_test_files_2023` and the file `task1_test_labels_2023.json` in `./COLIEE/task1_test_2023/` as follows:

```
$ ./COLIEE/
.
├── preprocessing
│   ├── openaiAPI.py
│   ├── process.py
│   └── reference.py
├── task1_test_2023
│   ├── processed
│   ├── processed_new
│   ├── summary_test_2023_txt
│   ├── task1_test_files_2023
│   ├── task1_test_labels_2023.json
│   └── test_2023_candidate_with_yearfilter.json
└── PromptCase_coliee2023_BM25_prediction_dict.json
```

# Preprocessing
1. Run `python3 LeCaRD/lecard_preprocessing.py` for proprocessing on LeCaRD dataset, which will generate two files of `candidate_fact_dict.json` and `candidate_issue_dict.json` for model running. 

2. The preprocessing of COLIEE2023 dataset is based on the repo of [THUIR-COLIEE2023](https://github.com/CSHaitao/THUIR-COLIEE2023). 

   Sequently run the following three python files:
   - `python3 COLIEE/preprocessing/process.py`,
   - `python3 COLIEE/preprocessing/reference.py`, 
   - `python3 COLIEE/preprocessing/openaiAPI.py` , 
   
   which will generate files for model running in the following three folders:
   - `COLIEE/task1_test_2023/processed/`, 
   - `COLIEE/task1_test_2023/processed_new/`, 
   - `COLIEE/task1_test_2023/summary_test_2023_txt/`. 
   
   Note: the randomness of ChatGPT summary generation may cause different performances under the same experiment settings. And the ChatGPT generated summary files used in our experiment are provided.

# Model running
Run `python3 PromptCase_model.py --dataset $A --stage_num $B`.

1. `$A` is the dataset that can be chosen as `leacrd` or `coliee`.
2. `$B` is one stage or two stage experiment that can be chosen as `1` or `2`.
3. The PromptCase model is based on [SAILER](https://github.com/CSHaitao/SAILER/).

# Cite
If you find this repo useful, please cite
```
@article{PromptCase,
  author    = {Yanran Tang and 
               Ruihong Qiu and 
               Xue Li},
  title     = {Prompt-based Effective Input Reformulation for Legal Case Retrieval},
  journal   = {CoRR},
  volume    = {abs/2309.02962},
  year      = {2023},
}
```