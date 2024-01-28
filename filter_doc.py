import pickle
import json
import os
with open('/mnt/d/github/law-retrieval/passage.pickl', 'rb') as rf:
    contents = pickle.load(rf)

with open('/mnt/d/model_train/law_model/embeddings.pickl_0', 'rb') as rrf:
    vectors = pickle.load(rrf)
BASE_PATH = '/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/candidates/'
with open('/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/label_top30_dict.json', 'r') as jrf:
    rankings = json.load(jrf)

print(rankings.keys())

for key in rankings.keys():
    docs = []
    embeddings = []
    for file in os.listdir(os.path.join(BASE_PATH, key)):
        file_id = file.split('.')[0]
