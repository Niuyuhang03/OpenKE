import itertools
import json
import re
from matplotlib import pyplot as plt


'''
配置环境
'''
print("-------------get conf-------------")
# 输入文件：entity2id.txt，relation2id.txt，train2id.txt，valid2id.txt，test2id.txt，TransE.json，FB15k_mid2description.txt，entity2type.txt。
# entity2id.txt：实体名称和实体id的对应关系，由OpenKE得到。
# relation2id.txt：关系名称和关系id的对应关系，由OpenKE得到。
# train2id.txt,valid2id.txt,test2id.txt：entity的关系，由OpenKE的得到。
# TransE.json：对entity进行TransE的embedding结果，由OpenKE的TransE模型得到。
# FB15k_mid2description.txt：entity的描述信息，由DKRL得到。
# entity2type.txt：对entity初步分类的结果，由DKRL得到。
entity_path = "../benchmarks/FB15K237/entity2id.txt"
relation_path = "../benchmarks/FB15K237/relation2id.txt"
train_path = "../benchmarks/FB15K237/train2id.txt"
valid_path = "../benchmarks/FB15K237/valid2id.txt"
test_path = "../benchmarks/FB15K237/test2id.txt"
data_path = "TransE.json"
entity_description_path = "../benchmarks/FB15K237/FB15k_mid2description.txt"
entity_type_path = "../benchmarks/FB15K237/entity2type.txt"

# 输出文件：FB15K237.content，FB15K237.cites，FB15K237.rel，FB15K237.type，FB15K237.dele。
# FB15K237.content：每行第1列为实体id，最后1列为label，其他列为embeddings。label有多个时以逗号分隔。
# FB15K237.cites：每行第1列为实体1 id，第2列为实体2id，第3列为实体1和实体2的关系。
# FB15K237.rel：每行第1列为关系id，后100列为embeddings。
# FB15K237.type：每行第1列为实体id，第二列为其label。label有多个时以逗号分隔。
# FB15K237.dele：难以被归类到实体。
content_output_path = "FB15K237.content"
cites_output_path = "FB15K237.cites"
rel_output_path = "FB15K237.rel"
type_output_path = "FB15K237.type"
delete_entities_path = "FB15K237.dele"

# 初始labels，需要经过替换labels处理后得到最终labels
correct_labels = ['film', 'actor', 'producer', 'director', 'county', 'university', 'writer', 'city', 'team', 'composer', 'award', 'region', 'capital', 'comedian', 'author', 'genre', 'country', 'state', 'label', 'songwriter', 'artist', 'company', 'province', 'cinematographer', 'language', 'designer', 'guitarist', 'meeting', 'area', 'zone', 'fiction', 'party', 'singer', 'event', 'school', 'player', 'publisher', 'voice', 'editor', 'club', 'actress', 'winner', 'government', 'character', 'channel', 'taxonomy', 'program', 'sport', 'location', 'brand', 'organization', 'religion', 'computer', 'job', 'subject', 'profession', 'position', 'sports', 'cause', 'fame', 'list', 'lists', 'parliament', 'military', 'degree', 'study', 'session', 'category', 'food', 'broadcaster']

# 替换labels，用于替换初始labels的内容
replace_labels = {'capital': 'location', 'club': 'sport', 'label': 'record_label', 'actress': 'person', 'genre': 'taxonomy', 'cause': 'taxonomy', 'category': 'taxonomy', 'zone': 'location', 'sports': 'sport', 'fame': 'award', 'lists': 'list', 'parliament': 'event', 'session': 'event', 'county': 'location', 'region': 'location', 'team': 'sport', 'publisher': 'person', 'singer': 'person', 'editor': 'person', 'artist': 'person', 'broadcaster': 'person', 'producer': 'person', 'composer': 'person', 'comedian':'person', 'actor':'person', 'director': 'person', 'player': 'person', 'winner': 'person', 'area': 'location', 'state': 'location', 'writer': 'person', 'author': 'person', 'city': 'location', 'university': 'location', 'country': 'location', 'school': 'location', 'songwriter': 'person', 'province': 'location', 'guitarist': 'person', 'position': 'location', 'designer': 'person', 'cinematographer': 'person', 'character': 'person', 'meeting': 'event', 'profession': 'job', 'degree': 'study', 'subject': 'study'}

bad_case = {'/m/0dnqr': ['film']}
print("-------------get conf finished-------------")


'''
处理.content和rel文件
'''
print("-------------process content and rel-------------")
delete_entities_id = []
delete_entities = []

# 构造实体和对应DKRL初步分类的字典
entity_type_output_file = open(entity_type_path, 'r')
entity_type = {}
for line in entity_type_output_file.readlines():
    line = line.split()
    entity = line.pop(0)
    types = list(map(lambda y: y.split('/')[-1].split('_'), line))
    types = list(set(itertools.chain.from_iterable(types)))
    if '' in types:
        types.remove('')
    entity_type[entity] = types
entity_type_output_file.close()

# 构造实体和对应实体描述的字典
entity_description_file = open(entity_description_path, 'r', encoding='utf-8')
entity_description = {}
for line in entity_description_file.readlines():
    entity = line.split()[0]
    line = re.split(r'[^a-zA-Z]', line)
    descriptions = list(set(line[1:]))
    entity_description[entity] = descriptions
entity_description_file.close()

# 把实体描述和DKRL初步分类在该entity的初始labels里筛选一次
entity_file = open(entity_path, 'r')
entity_lines = entity_file.readlines()[1:]
data_file = open(data_path, 'r')
rel_file = open(relation_path, 'r')
rel_lines = rel_file.readlines()[1:]
data = json.load(data_file)
data_ent = data['ent_embeddings.weight']
data_rel = data['rel_embeddings.weight']

content_output_file = open(content_output_path, 'w')
rel_output_file = open(rel_output_path, 'w')
type_output_file = open(type_output_path, 'w')

all_labels = {}

line_index = 0
for line in entity_lines:
    entity, entityid = line.split()

    # 对bad case里的实体特殊处理
    if bad_case.get(entity, None) is not None:
        # 写入实体名称和id
        type_output_file.write(entity + '\t' + entityid)
        content_output_file.write(entity + '\t' + entityid)
        # 写入实体embeddings
        cur_datas = data_ent[line_index]
        for cur_data in cur_datas:
            content_output_file.write('\t' + str(cur_data))
        # 写入实体labels
        cnt = 0
        for label in bad_case[entity]:
            if cnt == 0:
                cnt = 1
                type_output_file.write('\t' + label.replace('\'', ''))
                content_output_file.write('\t' + label.replace('\'', ''))
            else:
                type_output_file.write(',' + label.replace('\'', ''))
                content_output_file.write(',' + label.replace('\'', ''))
            all_labels[label] = all_labels.get(label, 0) + 1
        type_output_file.write('\n')
        content_output_file.write('\n')
        line_index += 1
        continue

    # 处理其他实体
    if entity_type.get(entity, None) is None and entity_description.get(entity, None) is None:
        delete_entities_id.append(entityid)
        delete_entities.append(entity)
    labels = []
    if entity in entity_description:
        descriptions = entity_description[entity]
        for description in descriptions:
            if description.lower() in correct_labels:
                if description.lower() in replace_labels:
                    labels.append(replace_labels[description.lower()])
                else:
                    labels.append(description.lower())
    if entity in entity_type:
        types = entity_type[entity]
        for type in types:
            if type.lower() in correct_labels:
                if type.lower() in replace_labels:
                    labels.append(replace_labels[type.lower()])
                else:
                    labels.append(type.lower())

    if len(labels) == 0:
        delete_entities_id.append(entityid)
        delete_entities.append(entity)
        line_index += 1
        continue

    # 写入实体名称和id
    type_output_file.write(entity + '\t' + entityid)
    content_output_file.write(entity + '\t' + entityid)
    # 写入实体embeddings
    cur_datas = data_ent[line_index]
    for cur_data in cur_datas:
        content_output_file.write('\t' + str(cur_data))
    # 写入实体labels
    cnt = 0
    for label in set(labels):
        all_labels[label] = all_labels.get(label, 0) + 1
        if cnt == 0:
            cnt = 1
            type_output_file.write('\t' + label.replace('\'', ''))
            content_output_file.write('\t' + label.replace('\'', ''))
        else:
            type_output_file.write(',' + label.replace('\'', ''))
            content_output_file.write(',' + label.replace('\'', ''))
    type_output_file.write('\n')
    content_output_file.write('\n')
    line_index += 1
entity_file.close()
type_output_file.close()
data_file.close()
content_output_file.close()

# 处理rel文件
line_index = 0
for line in data_rel:
    rel, relid = rel_lines[line_index].split()
    rel_output_file.write(rel + '\t' + relid)
    for cur_data in line:
        rel_output_file.write('\t' + str(cur_data))
    rel_output_file.write('\n')
    line_index += 1
rel_output_file.close()

print("-------------process content and rel finished-------------")

'''
处理.cites文件
'''
print("-------------process cites-------------")
train_file = open(train_path, 'r')
valid_file = open(valid_path, 'r')
test_file = open(test_path, 'r')
cites_output_file = open(cites_output_path, 'w')

triple_lines = train_file.readlines()[1:] + valid_file.readlines()[1:] + test_file.readlines()[1:]
my_dic = {}
delete_cnt = 0
for line in triple_lines:
    e1, e2, r =line.split()
    if e1 in delete_entities_id or e2 in delete_entities_id:
        continue
    if my_dic.get(str(e1) + '+' + str(e2), None) is None or r not in my_dic[str(e1) + '+' + str(e2)]:
        my_dic[str(e1) + '+' + str(e2)] = my_dic.get(str(e1) + '+' + str(e2), []) + [r]
        cites_output_file.write(str(e1) + '\t' + str(e2) + '\t' + str(r) + '\n')
train_file.close()
valid_file.close()
test_file.close()
cites_output_file.close()

with open(delete_entities_path, 'w') as f:
    for ent in delete_entities:
        f.write(ent)
print("-------------process cites finished-------------")

# cnt all labels  25
# all labels cnt:  [('film', 7243), ('award', 7175), ('person', 7029), ('location', 6039), ('organization', 2970), ('study', 2940), ('sport', 2542), ('taxonomy', 2194), ('program', 1792), ('fiction', 1061), ('company', 987), ('military', 858), ('event', 783), ('government', 763), ('list', 622), ('voice', 558), ('language', 478), ('job', 384), ('record_label', 333), ('computer', 310), ('party', 282), ('channel', 225), ('food', 177), ('brand', 170), ('religion', 160)]
# all labels:  ['taxonomy', 'study', 'location', 'organization', 'government', 'sport', 'military', 'program', 'award', 'company', 'film', 'voice', 'person', 'language', 'fiction', 'event', 'computer', 'list', 'job', 'religion', 'brand', 'food', 'party', 'channel', 'record_label']

print('-------------statistics-------------')
print("all labels: {}".format(len(all_labels)))
sorted_all_label = sorted(all_labels.items(), key = lambda x: x[1], reverse = True)
print('all labels cnt: ', sorted_all_label)
print('all labels: ', [label for label in all_labels])

f = open('FB15K237.content')
print("FB15K237.content:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 3)

f = open('FB15K237.cites')
print("FB15K237.cites:")
lines = f.readlines()
print(len(lines))

f = open('FB15K237.rel')
print("FB15K237.rel:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 2)

name = []
y = []
labels_plot = sorted(all_labels.items(), key=lambda item: item[1], reverse=True)
for key, value in labels_plot:
    name.append(key)
    y.append(value)
x = range(len(name))
plt.bar(x, y)
for a, b in zip(x, y):
    plt.text(a,b,'%d' % b,ha='center',va= 'bottom',fontsize=6)
plt.xticks(x,name,fontsize=6,rotation=60)
plt.xlabel("labels")
plt.ylabel("count")
plt.title('FB15K237, entities number {}'.format(len(entity_lines) - len(delete_entities)))
plt.show()