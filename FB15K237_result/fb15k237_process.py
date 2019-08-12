import itertools
import json
import re


'''
配置环境
'''
print("-------------get conf-------------")

process_content = True
process_cites = True

# 输出文件：FB15K237.content，FB15K237.cites。
# FB15K237.content：每行第一列为entity id，最后一列为label，其他列为embedding feature。
# FB15K237.cites：每行第一列为entity1 id，第二列为entity2 id，表示e1和e2之间有关系r。
# FB15K237.rel：对rel进行embedding的结果。
content_output_path = "E:/PycharmProjects/OpenKE/FB15K237_result/FB15K237.content"
cites_output_path = "E:/PycharmProjects/OpenKE/FB15K237_result/FB15K237.cites"
rel_output_path = "E:/PycharmProjects/OpenKE/FB15K237_result/FB15K237.rel"

# 中间文件：FB15K237.type。
# FB15K237.type，每行第一列为entity id，第二列为label。
type_output_path = "E:/PycharmProjects/OpenKE/FB15K237_result/FB15K237.type"

# 输入文件：entity2id.txt，entity2type.txt，FB15k_mid2description.txt，train2id.txt，TransE.json。
# entity2id.txt：entity id和entity编号的对应关系，由OpenKE得到。
# entity2type.txt：对entity初步分类的结果，由DKRL得到。
# FB15k_mid2description.txt：entity的描述信息，由DKRL得到。
# train2id valid2id test2id.txt：entity的关系，由OpenKE的得到。
# TransE.json：对entity进行TransE的embedding结果，由TransE得到。
entity_path = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/entity2id.txt"
entity_type_path = "E:/PycharmProjects/DKRL/data/entityType_split/entity2type.txt"
entity_description_path = "E:/PycharmProjects/DKRL/data/FB15k_description/FB15k_mid2description.txt"
relation_path = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/relation2id.txt"
train_path = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/train2id.txt"
valid_path = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/valid2id.txt"
test_path = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/test2id.txt"
data_path = "E:/PycharmProjects/OpenKE/FB15K237_result/TransE.json"

# not final label, correct_labels should be replaced with replace_labels
correct_labels = ['film', 'actor', 'producer', 'director', 'county', 'university', 'writer', 'city', 'team', 'composer', 'award', 'region', 'capital', 'comedian', 'author', 'genre', 'country', 'state', 'label', 'songwriter', 'artist', 'company', 'province', 'cinematographer', 'language', 'designer', 'guitarist', 'meeting', 'area', 'zone', 'fiction', 'party', 'singer', 'event', 'school', 'player', 'publisher', 'voice', 'editor', 'club', 'actress', 'winner', 'government', 'character', 'channel', 'taxonomy', 'program', 'sport', 'location', 'brand', 'organization', 'religion', 'computer', 'job', 'subject', 'profession', 'position', 'sports', 'cause', 'fame', 'list', 'lists', 'parliament', 'military', 'degree', 'study', 'session', 'category', 'food', 'broadcaster']

bad_case = {'/m/0dnqr': ['film']}

replace_labels = {'capital': 'location', 'club': 'sport', 'label': 'record_label', 'actress': 'person', 'genre': 'taxonomy', 'cause': 'taxonomy', 'category': 'taxonomy', 'zone': 'location', 'sports': 'sport', 'fame': 'award', 'lists': 'list', 'parliament': 'event', 'session': 'event', 'county': 'location', 'region': 'location', 'team': 'sport', 'publisher': 'person', 'singer': 'person', 'editor': 'person', 'artist': 'person', 'broadcaster': 'person', 'producer': 'person', 'composer': 'person', 'comedian':'person', 'actor':'person', 'director': 'person', 'player': 'person', 'winner': 'person', 'area': 'location', 'state': 'location', 'writer': 'person', 'author': 'person', 'city': 'location', 'university': 'location', 'country': 'location', 'school': 'location', 'songwriter': 'person', 'province': 'location', 'guitarist': 'person', 'position': 'location', 'designer': 'person', 'cinematographer': 'person', 'character': 'person', 'meeting': 'event', 'profession': 'job', 'degree': 'study', 'subject': 'study'}

delete_entities = []

print("-------------get conf finished-------------")


'''
处理.content和rel文件
'''
if process_content:
    print("-------------process content and rel-------------")
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
        line = re.split(r'[^a-zA-Z]', line)
        descriptions = list(set(line[1:]))
        entity_description[entity] = descriptions
    entity_description_file.close()

    # find entity type. 把description在该entity的correct type里筛选一次。再将其全部type放入correct type筛选一次。
    entity_file = open(entity_path, 'r')
    entity_lines = entity_file.readlines()[1:]
    type_file = open(type_output_path, 'w')
    data_file = open(data_path, 'r')
    rel_output_file = open(rel_output_path, 'w')
    rel_file = open(relation_path, 'r')
    rel_lines = rel_file.readlines()[1:]
    data = json.load(data_file)
    data_ent = data['ent_embeddings.weight']
    data_rel = data['rel_embeddings.weight']
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
            cur_datas = data_ent[line_index]
            for cur_data in cur_datas:
                content_file.write('\t' + str(cur_data))
            # write labels
            cnt = 0
            for label in bad_case[entity]:
                if cnt == 0:
                    cnt = 1
                    type_file.write('\t' + label.replace('\'', ''))
                    content_file.write('\t' + label.replace('\'', ''))
                else:
                    type_file.write(',' + label.replace('\'', ''))
                    content_file.write(',' + label.replace('\'', ''))
                all_labels[label] = all_labels.get(label, 0) + 1
            # write \n
            type_file.write('\n')
            content_file.write('\n')
            line_index += 1
            continue

        # get entities labels
        if entity_type.get(entity, None) is None and entity_description.get(entity, None) is None:
            print(entity + ' doesn\'t in type or description')
            delete_entities.append(entity)
            miss_in_entity.append(entityid)
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
            nolabel_entity.append(entity)
            delete_entities.append(entityid)
            line_index += 1
            continue
        type_file.write(entityid)
        content_file.write(entityid)
        cur_datas = data_ent[line_index]
        for cur_data in cur_datas:
            content_file.write('\t' + str(cur_data))
        cnt = 0
        for label in set(labels):
            all_labels[label] = all_labels.get(label, 0) + 1
            if cnt == 0:
                cnt = 1
                type_file.write('\t' + label.replace('\'', ''))
                content_file.write('\t' + label.replace('\'', ''))
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

    # processing rel
    line_index = 0
    for line in data_rel:
        rel, relid = rel_lines[line_index].split()
        rel_output_file.write(str(relid))
        for cur_data in line:
            rel_output_file.write('\t' + str(cur_data))
        rel_output_file.write('\n')
        line_index += 1
    rel_output_file.close()

    print('-------------result-------------')
    print('cnt all labels ', str(len(all_labels)))
    sorted_all_label = sorted(all_labels.items(), key = lambda x: x[1], reverse = True)
    print('all labels cnt: ', sorted_all_label)
    print('all labels: ', [label for label in all_labels])
    print('-------------error-------------')
    print('cnt no label ', str(len(nolabel_entity)))
    print('cnt miss in type and description', str(len(set(miss_in_entity))))
    print('all entities which is deleted ', delete_entities)
    print('all entities cnt which is deleted ', len(delete_entities))
    print('-------------error end-------------')
    print("-------------process content and rel finished")


'''
处理.cites文件
'''
if process_cites:
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
        if e1 in delete_entities or e2 in delete_entities:
            continue
        if my_dic.get(str(e1) + '+' + str(e2), None) is None or r not in my_dic[str(e1) + '+' + str(e2)]:
            my_dic[str(e1) + '+' + str(e2)] = my_dic.get(str(e1) + '+' + str(e2), []) + [r]
            cites_output_file.write(str(e1) + '\t' + str(e2) + '\t' + str(r) + '\n')
        else:
            delete_cnt += 1
            print(str(e2) + " is already in " + str(e1))
    print("{:d} entities are deleted in cites.".format(delete_cnt))
    train_file.close()
    valid_file.close()
    test_file.close()
    cites_output_file.close()
    print("-------------process cites finished-------------")

# cnt all labels  28
# all labels cnt:  [('film', 7243), ('award', 7175), ('person', 6869), ('location', 6039), ('organization', 2970), ('subject', 2616), ('sport', 2542), ('taxonomy', 2194), ('program', 1792), ('fiction', 1061), ('company', 987), ('military', 858), ('event', 783), ('government', 763), ('list', 622), ('form', 588), ('voice', 558), ('language', 478), ('study', 397), ('job', 384), ('record_label', 333), ('computer', 310), ('party', 282), ('degree', 255), ('channel', 225), ('food', 177), ('brand', 170), ('religion', 160)]
# all labels:  ['location', 'organization', 'government', 'subject', 'military', 'sport', 'taxonomy', 'form', 'company', 'award', 'program', 'film', 'voice', 'person', 'language', 'fiction', 'study', 'event', 'computer', 'degree', 'list', 'job', 'religion', 'brand', 'food', 'party', 'channel', 'record_label']

f = open('FB15K237.content')
print("FB15K237.content:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 2)

f = open('FB15K237.cites')
print("FB15K237.cites:")
lines = f.readlines()
print(len(lines))

f = open('FB15K237.rel')
print("FB15K237.rel:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 1)
