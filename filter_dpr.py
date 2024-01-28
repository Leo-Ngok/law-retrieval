import pickle
import json

# 加载file_names.pkl文件
with open('file_names_100.pkl', 'rb') as f:
    file_names = pickle.load(f)

# 加载embeddings.pickl_0文件
with open('passage.pickl', 'rb') as f:
    data = pickle.load(f)

# 创建一个新的空列表
filtered_data = []
print(len(file_names))
# 遍历file_names中的每个set
# 创建一个新的空set
total = {}
filtered_set = {}
# 遍历embeddings.pickl_0文件中的每一项
for key, value in data.items():
    # 检查每一项的id是否在当前set中
    for qid, file_set in file_names.items():
        if key in file_set:
            if qid in total:
                total[qid][key] = value
            else:
                total[qid] = {key: value}

# 将新的列表dump到一个新的pickle文件中
with open('filtered_passage_100.pickl', 'wb') as f:
    pickle.dump(total, f)

for key, value in total.items():
    print(key, len(value))
