import itertools
import json
import re

'''
配置环境
'''
print("-------------get conf-------------")
# 输出文件：FB15K237.content，FB15K237.cites。
# FB15K237.content：每行第一列为entity id，最后一列为label，其他列为embedding feature。
# FB15K237.cites：每行第一列为entity1 id，第二列为entity2 id，表示e1和e2之间有关系r。
content_output_path = "E:/PycharmProjects/OpenKE/FB15K237_result/FB15K237.content"
relation_output_path = "E:/PycharmProjects/OpenKE/FB15K237_result/FB15K237.cites"

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
relation_path_train = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/train2id.txt"
relation_path_valid = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/valid2id.txt"
relation_path_test = "E:/PycharmProjects/OpenKE/benchmarks/FB15K237/test2id.txt"
data_path = "E:/PycharmProjects/OpenKE/FB15K237_result/TransE.json"

# not final label, correct_labels should be replaced with replace_labels
correct_labels = ['film', 'actor', 'producer', 'directordirector', 'county', 'university', 'writer', 'city', 'team', 'composer', 'award', 'region', 'capital', 'comedian', 'author', 'genre', 'country', 'state', 'label', 'songwriter', 'artist', 'company', 'province', 'cinematographer', 'language', 'designer', 'guitarist', 'meeting', 'area', 'zone', 'fiction', 'party', 'singer', 'event', 'school', 'player', 'publisher', 'voice', 'editor', 'club', 'actress', 'winner', 'government', 'character', 'channel', 'taxonomy', 'program', 'sport', 'location', 'brand', 'organization', 'religion', 'computer', 'job', 'subject', 'profession', 'position', 'sports', 'cause', 'form', 'fame', 'list', 'lists', 'parliament', 'military', 'degree', 'study', 'session', 'category', 'food', 'broadcaster']

bad_case = {'/m/0dnqr': ['film']}

replace_labels = {'capital': 'location', 'club': 'sport', 'label': 'record_label', 'actress': 'person', 'genre': 'taxonomy', 'cause': 'taxonomy', 'category': 'taxonomy', 'zone': 'location', 'sports': 'sport', 'fame': 'award', 'lists': 'list', 'parliament': 'event', 'session': 'event', 'county': 'location', 'region': 'location', 'team': 'sport', 'publisher': 'person', 'singer': 'person', 'editor': 'person', 'artist': 'person', 'broadcaster': 'person', 'producer': 'person', 'composer': 'person', 'comedian':'person', 'actor':'person', 'player': 'person', 'winner': 'person', 'area': 'location', 'state': 'location', 'writer': 'person', 'author': 'person', 'city': 'location', 'university': 'location', 'country': 'location', 'school': 'location', 'songwriter': 'person', 'province': 'location', 'guitarist': 'person', 'position': 'location', 'designer': 'person', 'cinematographer': 'person', 'character': 'person', 'meeting': 'event', 'profession': 'job'}

process_content = True
process_cites = True
delete_entities = []

print("-------------get conf finished-------------")


'''
处理.content文件
'''
if process_content:
    print("-------------process content-------------")
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
    print('all entities which is deleted ', delete_entities)
    print('all entities cnt which is deleted ', len(delete_entities))
    print('-------------error end-------------')
    print("-------------process content finished")


'''
处理.cites文件
'''
if process_cites:
    print("-------------process cites-------------")
    relation_train_file = open(relation_path_train, 'r')
    relation_valid_file = open(relation_path_valid, 'r')
    relation_test_file = open(relation_path_test, 'r')
    relation_output_file = open(relation_output_path, 'w')

    relation_lines = relation_train_file.readlines()[1:] + relation_valid_file.readlines()[1:] + relation_test_file.readlines()[1:]
    my_dic = {}
    delete_cnt = 0
    for line in relation_lines:
        e1, e2, r =line.split()
        if e1 in delete_entities or e2 in delete_entities:
            continue
        if (my_dic.get(e1, None) is None or e2 not in my_dic[e1]) and (my_dic.get(e2, None) is None or e1 not in my_dic[e2]):
            my_dic[e1] = my_dic.get(e1, []) + [e2]
            relation_output_file.write(str(e1) + ' ' + str(e2) + '\n')
        else:
            delete_cnt += 1
            # print(str(e2) + " is already in " + str(e1))
    print("{:d} entities are deleted in cites.".format(delete_cnt))
    relation_train_file.close()
    relation_valid_file.close()
    relation_test_file.close()
    relation_output_file.close()
    print("-------------process cites finished-------------")

# cnt all labels  28
# all labels cnt:  [('film', 7243), ('award', 7175), ('person', 6869), ('location', 6039), ('organization', 2970), ('subject', 2616), ('sport', 2542), ('taxonomy', 2194), ('program', 1792), ('fiction', 1061), ('company', 987), ('military', 858), ('event', 783), ('government', 763), ('list', 622), ('form', 588), ('voice', 558), ('language', 478), ('study', 397), ('job', 384), ('record_label', 333), ('computer', 310), ('party', 282), ('degree', 255), ('channel', 225), ('food', 177), ('brand', 170), ('religion', 160)]
# all labels:  ['location', 'organization', 'government', 'subject', 'military', 'sport', 'taxonomy', 'form', 'company', 'award', 'program', 'film', 'voice', 'person', 'language', 'fiction', 'study', 'event', 'computer', 'degree', 'list', 'job', 'religion', 'brand', 'food', 'party', 'channel', 'record_label']
