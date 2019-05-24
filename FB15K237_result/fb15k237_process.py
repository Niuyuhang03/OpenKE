import itertools
import json
import re

# 输出文件：FB15K237.content，FB15K237.relation。
# FB15K237.content：每行第一列为entity id，最后一列为label，其他列为embedding feature。
# FB15K237.relation：每行第一列为entity1 id，第二列为entity2 id，表示e1和e2之间有关系r。
content_output_path = "C:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/FB15K237.content"
relation_output_path = "C:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/FB15K237.relation"

# 中间文件：FB15K237.type。
# FB15K237.type，每行第一列为entity id，第二列为label。
type_output_path = "C:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/FB15K237.type"

# 输入文件：entity2id.txt，entity2type.txt，FB15k_mid2description.txt，train2id.txt，TransE.json。
# entity2id.txt：entity id和entity编号的对应关系，由OpenKE得到。
# entity2type.txt：对entity初步分类的结果，由DKRL得到。
# FB15k_mid2description.txt：entity的描述信息，由DKRL得到。
# train2id.txt：entity的关系，由OpenKE的得到。
# TransE.json：对entity进行TransE的embedding结果，由OpenKE的TransE得到。
entity_path = "C:/Users/14114/PycharmProjects/OpenKE/benchmarks/FB15K237/entity2id.txt"
entity_type_path = "C:/Users/14114/PycharmProjects/DKRL/data/entityType_split/entity2type.txt"
entity_description_path = "C:/Users/14114/PycharmProjects/DKRL/data/FB15k_description/FB15k_mid2description.txt"
relation_path = "C:/Users/14114/PycharmProjects/OpenKE/benchmarks/FB15K237/train2id.txt"
data_path = "C:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/TransE.json"

# not final label, correct_labels should be replaced with replace_labels
correct_labels = ['film', 'actor', 'producer', 'director', 'county', 'university', 'writer', 'city', 'team', 'composer', 'award', 'region', 'capital', 'comedian', 'author', 'genre', 'country', 'state', 'label', 'songwriter', 'artist', 'company', 'instrument', 'province', 'cinematographer', 'language', 'designer', 'guitarist', 'meeting', 'area', 'zone', 'fiction', 'party', 'singer', 'event', 'school', 'player', 'philosopher', 'publisher', 'voice', 'editor', 'club', 'actress', 'winner', 'government', 'character', 'channel', 'taxonomy', 'program', 'sport', 'color', 'location', 'brand', 'organization', 'disease', 'chemical', 'religion', 'broadcaster', 'geographical', 'computer', 'ethnicity', 'job', 'software', 'subject', 'currency', 'profession', 'nutrient', 'position', 'sports', 'cause', 'form', 'fame', 'list', 'lists', 'parliament', 'month', 'military', 'degree', 'study', 'session', 'category', 'food']

bad_case = {'/m/0dnqr': ['film'], '/m/03p41': ['disease']}

replace_labels = {'capital': 'city', 'club': 'team', 'label': 'record_label', 'actress': 'actor', 'genre': 'taxonomy', 'cause': 'taxonomy', 'category': 'taxonomy', 'zone': 'area', 'chemical': 'chemical_compound', 'sports': 'sport', 'fame': 'award', 'lists': 'list', 'parliament': 'meeting', 'session': 'meeting'}

# make entity type to dictionary
entity_type_file = open(entity_type_path, 'r')
entity_type = {}
for line in entity_type_file.readlines():
    line = line.split()
    entity = line.pop(0)
    types = list(map(lambda y: y.split('/')[-1].split('_'), line))
    types = list(set(itertools.chain.from_iterable(types)))
    if '' in types:
        types.remove('')
    entity_type[entity] = types
entity_type_file.close()

# make entity description to dictionary
entity_description_file = open(entity_description_path, 'r', encoding='utf-8')
entity_description = {}
start_words = [' is ', ' are ', ' was ', ' were ', ' be ']
for line in entity_description_file.readlines():
    entity = line.split()[0]
    # starts = []
    # for start_word in start_words:
    #     if start_word in line:
    #         starts.append(line.index(start_word))
    # if len(starts) == 0:
    #     start = line.index(line.split()[1])
    # else:
    #     start = min(starts)
    line = re.split(r'[^a-zA-Z]', line)
    descriptions = list(set(line[1:]))
    entity_description[entity] = descriptions
entity_description_file.close()

# find entity type. 把description在该entity的correct type里筛选一次。再将其全部type放入correct type筛选一次。
entity_file = open(entity_path, 'r')
entity_lines = entity_file.readlines()[1:]
type_file = open(type_output_path, 'w')
data_file = open(data_path, 'r')
data = json.load(data_file)['ent_embeddings.weight']
content_file = open(content_output_path, 'w')

miss_in_entity = []
nolabel_entity = []
all_labels = {}
line_index = 0

for line in entity_lines:
    entity, entityid = line.split()
    # if entity in bad case
    if bad_case.get(entity, None) is not None:
        # write entity id
        type_file.write(entityid)
        content_file.write(entityid)
        # write embedding data
        cur_datas = data[line_index]
        for cur_data in cur_datas:
            content_file.write(' ' + str(cur_data))
        # write labels
        cnt = 0
        for label in bad_case[entity]:
            if cnt == 0:
                cnt = 1
                type_file.write(' ' + label.replace('\'', ''))
                content_file.write(' ' + label.replace('\'', ''))
            else:
                type_file.write(',' + label.replace('\'', ''))
                content_file.write(',' + label.replace('\'', ''))
            all_labels[label] = all_labels.get(label, 0) + 1
        # write \n
        type_file.write('\n')
        content_file.write('\n')
        line_index += 1
        continue
    if entity_type.get(entity, None) is None and entity_description.get(entity, None) is None:
        print(entity + ' doesn\'t in type or description')
        miss_in_entity.append(entity)
    type_file.write(entityid)
    content_file.write(entityid)
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
        print(entity + " label is 0. ")
        if entity in entity_type:
            print(entity_type[entity])
        type_file.write('\n')
        content_file.write('\n')
        nolabel_entity.append(entity)
        line_index += 1
        continue
    cur_datas = data[line_index]
    for cur_data in cur_datas:
        content_file.write(' ' + str(cur_data))
    cnt = 0
    for label in set(labels):
        all_labels[label] = all_labels.get(label, 0) + 1
        if cnt == 0:
            cnt = 1
            type_file.write(' ' + label.replace('\'', ''))
            content_file.write(' ' + label.replace('\'', ''))
        else:
            type_file.write(',' + label.replace('\'', ''))
            content_file.write(',' + label.replace('\'', ''))
    type_file.write('\n')
    content_file.write('\n')
    line_index += 1
entity_file.close()
type_file.close()
data_file.close()
content_file.close()

print('-------------result-------------')
print('cnt all labels ', str(len(all_labels)))
sorted_all_label = sorted(all_labels.items(), key = lambda x: x[1], reverse = True)
print('all labels cnt: ', sorted_all_label)
print('all labels: ', [label for label in all_labels])
print('-------------error-------------')
print('cnt no label ', str(len(nolabel_entity)))
print('cnt miss in type and description', str(len(set(miss_in_entity))))
print('-------------error end-------------')

relation_file = open(relation_path, 'r')
relation_output_file = open(relation_output_path, 'w')
relation_lines = relation_file.readlines()[1:]
for line in relation_lines:
    e1, e2, r =line.split()
    relation_output_file.write(str(e1) + ' ' + str(e2) + '\n')
relation_file.close()
relation_output_file.close()

# cnt all labels  70
# all labels cnt:  [('film', 7243), ('award', 7175), ('actor', 4179), ('winner', 3744), ('location', 3296), ('organization', 2970), ('city', 2921), ('producer', 2671), ('subject', 2616), ('area', 2594), ('region', 2545), ('artist', 2416), ('team', 2199), ('taxonomy', 2194), ('sport', 2187), ('director', 1952), ('state', 1911), ('program', 1792), ('writer', 1648), ('author', 1626), ('university', 1416), ('county', 1395), ('character', 1122), ('country', 1091), ('fiction', 1061), ('composer', 999), ('company', 987), ('school', 928), ('military', 858), ('singer', 827), ('government', 763), ('songwriter', 643), ('list', 622), ('form', 588), ('voice', 558), ('event', 500), ('language', 478), ('player', 472), ('study', 397), ('comedian', 387), ('province', 361), ('guitarist', 337), ('record_label', 333), ('position', 324), ('computer', 310), ('meeting', 297), ('party', 282), ('designer', 277), ('degree', 255), ('editor', 253), ('profession', 253), ('channel', 225), ('job', 208), ('publisher', 196), ('cinematographer', 188), ('food', 177), ('brand', 170), ('religion', 160), ('disease', 152), ('instrument', 151), ('software', 149), ('geographical', 141), ('color', 112), ('chemical_compound', 107), ('ethnicity', 91), ('philosopher', 88), ('month', 79), ('nutrient', 72), ('broadcaster', 30), ('currency', 23)]
# all labels:  ['country', 'team', 'city', 'organization', 'area', 'region', 'taxonomy', 'location', 'sport', 'military', 'state', 'government', 'subject', 'form', 'company', 'program', 'award', 'film', 'voice', 'actor', 'school', 'language', 'fiction', 'winner', 'position', 'producer', 'writer', 'director', 'singer', 'comedian', 'instrument', 'study', 'event', 'author', 'composer', 'artist', 'county', 'computer', 'geographical', 'character', 'editor', 'publisher', 'degree', 'university', 'list', 'job', 'profession', 'religion', 'player', 'guitarist', 'brand', 'songwriter', 'cinematographer', 'meeting', 'currency', 'food', 'party', 'designer', 'province', 'channel', 'software', 'disease', 'color', 'record_label', 'ethnicity', 'broadcaster', 'philosopher', 'month', 'chemical_compound', 'nutrient']
