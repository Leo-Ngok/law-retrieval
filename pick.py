import pickle

# 加载file_names.pkl文件
with open('file_names.pkl', 'rb') as f:
    file_names = pickle.load(f)

# 加载embeddings.pickl_0文件
with open('passage.pickl', 'rb') as f:
    data = pickle.load(f)

# 创建一个新的空列表
filtered_data = []
print(len(file_names))
# 遍历file_names中的每个set
# 创建一个新的空set
filtered_set = {}
# 遍历embeddings.pickl_0文件中的每一项
for key, value in data.items():
    # 检查每一项的id是否在当前set中
    if key in file_names:
        # 如果是，将这一项添加到新的set中
        filtered_set[key] = value

# 将新的列表dump到一个新的pickle文件中
with open('filtered_passage.pickl', 'wb') as f:
    pickle.dump(filtered_set, f)

# 打印新列表的长度
print(len(filtered_set))
print(filtered_set[0])
