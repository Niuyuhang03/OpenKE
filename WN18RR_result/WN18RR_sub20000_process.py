import json
import random
from nltk.corpus import wordnet as wn
from matplotlib import pyplot as plt


'''
配置环境
'''
print("-------------get conf-------------")
# 删除10000个随机实体
entity_file = open("../benchmarks/WN18RR/entity2id.txt", 'r')
entity_num = len(entity_file.readlines())
chosen_entity_list = random.sample(range(entity_num - 1), 20000)
# chosen_entity_list = range(entity_num)
entity_file.close()

# 输入文件：entity2id.txt，relation2id.txt，train2id.txt，valid2id.txt，test2id.txt，TransE.json。
# entity2id.txt：实体名称和实体id的对应关系，由OpenKE得到。
# relation2id.txt：关系名称和关系id的对应关系，由OpenKE得到。
# train2id.txt,valid2id.txt,test2id.txt：entity的关系，由OpenKE的得到。
# TransE.json：对entity进行TransE的embedding结果，由OpenKE的TransE模型得到。
entity_path = "../benchmarks/WN18RR/entity2id.txt"
relation_path = "../benchmarks/WN18RR/relation2id.txt"
train_path = "../benchmarks/WN18RR/train2id.txt"
valid_path = "../benchmarks/WN18RR/valid2id.txt"
test_path = "../benchmarks/WN18RR/test2id.txt"
data_path = "TransE.json"

# 输出文件：WN18RR.content，WN18RR.cites，WN18RR.rel，WN18RR.type
# WN18RR.content：每行第1列为实体id，最后1列为label，其他列为embeddings。label有多个时以逗号分隔。
# WN18RR.cites：每行第1列为实体1 id，第2列为实体2id，第3列为实体1和实体2的关系。
# WN18RR.rel：每行第1列为关系id，后100列为embeddings。
# WN18RR.type，每行第1列为实体id，第二列为其label。label有多个时以逗号分隔。
content_output_path = "WN18RR_sub20000.content"
cites_output_path = "WN18RR_sub20000.cites"
rel_output_path = "WN18RR_sub20000.rel"
type_output_path = "WN18RR_sub20000.type"

print("-------------get conf finished-------------")

'''
处理.content和rel文件
'''
delete_entities = []
print("-------------process content and rel-------------")
# WN18RR中所有labels
pos = ['n', 'v', 'a', 'r']

# 输入文件
entity_file = open(entity_path, 'r')
entity_lines = entity_file.readlines()[1:]
rel_file = open(relation_path, 'r')
rel_lines = rel_file.readlines()[1:]
data_file = open(data_path, 'r')
data = json.load(data_file)
data_ent = data['ent_embeddings.weight']
data_rel = data['rel_embeddings.weight']

# 输出文件
content_output_file = open(content_output_path, 'w')
rel_output_file = open(rel_output_path, 'w')
type_output_file = open(type_output_path, 'w')

line_index = 0
labels = []
labels_plot = {}
for line in entity_lines:
    entity, entityid = line.split()

    if line_index in chosen_entity_list:
        delete_entities.append(entityid)
        labels.append([])
        line_index += 1
        continue

    # 得到实体所有label
    labels.append([])
    for cur_pos in pos:
        try:
            wn.synset_from_pos_and_offset(cur_pos, int(entity))
            labels[line_index].append(cur_pos)
        except:
            pass

    if len(labels[line_index]) == 0:
        # 该实体没有label时打印错误
        print(entity + ' has no pos in synset\n')
        delete_entities.append(entityid)
    else:
        labels_plot[' '.join(labels[line_index])] = labels_plot.get(' '.join(labels[line_index]), 0) + 1
        # 写入实体名称和id
        type_output_file.write(entity + '\t' + entityid)
        content_output_file.write(entity + '\t' + entityid)

        # 写入实体embeddings
        cur_datas = data_ent[line_index]
        for cur_data in cur_datas:
            content_output_file.write('\t' + str(cur_data))

        # 写入实体labels
        cnt = 0
        for label in labels[line_index]:
            if cnt == 0:
                cnt = 1
                type_output_file.write('\t' + label)
                content_output_file.write('\t' + label)
            else:
                type_output_file.write(',' + label)
                content_output_file.write(',' + label)
        type_output_file.write('\n')
        content_output_file.write('\n')
    line_index += 1
entity_file.close()
data_file.close()
content_output_file.close()
type_output_file.close()

# 处理rel文件
line_index = 0
for line in data_rel:
    rel, relid = rel_lines[line_index].split()
    # 写入关系名称和id
    rel_output_file.write(rel + '\t' + relid)
    # 写入关系embeddings
    for cur_data in line:
        rel_output_file.write('\t' + str(cur_data))
    rel_output_file.write('\n')
    line_index += 1
rel_file.close()
rel_output_file.close()
print("-------------process content and rel finished-------------")


'''
处理.cites文件
'''
print('-------------process cites-------------')
# 输入文件
train_file = open(train_path, 'r')
valid_file = open(valid_path, 'r')
test_file = open(test_path, 'r')

# 输出文件
cites_output_file = open(cites_output_path, 'w')

triple_lines = train_file.readlines()[1:] + valid_file.readlines()[1:] + test_file.readlines()[1:]
my_dic = {}
for line in triple_lines:
    e1, e2, r =line.split()
    if e1 in delete_entities or e2 in delete_entities:
        continue
    if my_dic.get(str(e1) + '+' + str(e2), None) is None or r not in my_dic[str(e1) + '+' + str(e2)]:
        my_dic[str(e1) + '+' + str(e2)] = my_dic.get(str(e1) + '+' + str(e2), []) + [r]
        cites_output_file.write(str(e1) + '\t' + str(e2) + '\t' + str(r) + '\n')
train_file.close()
valid_file.close()
test_file.close()
cites_output_file.close()
print("-------------process cites finished-------------")

print('-------------statistics-------------')
f = open('WN18RR_sub20000.content')
print("WN18RR_sub20000.content:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 3)

f = open('WN18RR_sub20000.cites')
print("WN18RR_sub20000.cites:")
lines = f.readlines()
print(len(lines))

f = open('WN18RR_sub20000.rel')
print("WN18RR_sub20000.rel:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 2)

name = []
y = []
labels_plot = sorted(labels_plot.items(), key=lambda item: item[1], reverse=True)
for key, value in labels_plot:
    name.append(key)
    y.append(value)
x = range(len(name))
plt.bar(x, y)
for a, b in zip(x, y):
    plt.text(a,b,'%d' % b,ha='center',va= 'bottom',fontsize=11)
plt.xticks(x,name,size='small',rotation=30)
plt.xlabel("labels")
plt.ylabel("count")
plt.title('WN18RR_sub20000, entities number {}'.format(len(entity_lines) - len(delete_entities)))
plt.show()