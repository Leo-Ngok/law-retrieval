import pickle
import jsonlines


# file = "/mnt/d/github/law-retrieval/data/validation_data.jsonl" 
# with jsonlines.open(file, mode="r") as jsonl_reader:
#     data = [l for l in jsonl_reader]
# print(data)

# with open('/mnt/d/model_train/law_model/embeddings.pickl_0', 'rb') as rf:
#     data = pickle.load(rf)
with open('/mnt/d/Documents/WeChat Files/wxid_lf8wivy66vox22/FileStorage/File/2023-12/filtered_embeddings(1).pickl', 'rb') as testf:
    data = pickle.load(testf)
#dt = data[0][1]
print(data)