import itertools
import json
import re
from nltk.corpus import wordnet as wn

# 输出文件：WN18RR.content，WN18RR.cites。
# WN18RR.content：每行第一列为entity id，最后一列为label，其他列为embedding feature。
# WN18RR.cites：每行第一列为entity1 id，第二列为entity2 id，表示e1和e2之间有关系r。
content_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.content"
relation_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.cites"

# 中间文件：WN18RR.type。
# WN18RR.type，每行第一列为entity id，第二列为label。
type_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.type"

# 输入文件：entity2id.txt，train2id.txt，TransE.json。
# entity2id.txt：entity id和entity编号的对应关系，由OpenKE得到。
# train2id.txt：entity的关系，由OpenKE的得到。
# TransE.json：对entity进行TransE的embedding结果，由OpenKE的TransE得到。
entity_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/entity2id.txt"
relation_train_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/train2id.txt"
relation_valid_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/valid2id.txt"
relation_test_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/test2id.txt"
data_path = "E:/PycharmProjects/OpenKE/WN18RR_result/TransE.json"

process_content = True
process_cites = True

if process_content:

    pos = ['n', 'v', 'a', 'r']

    type_file = open(type_output_path, 'w')
    entity_file = open(entity_path, 'r')
    entity_lines = entity_file.readlines()[1:]
    data_file = open(data_path, 'r')
    data = json.load(data_file)['ent_embeddings.weight']
    content_file = open(content_output_path, 'w')
    line_index = 0

    print('-------------process entity-------------')
    for line in entity_lines:
        entity, entityid = line.split()
        entity = int(entity)
        type_file.write(entityid)
        content_file.write(entityid)
        labels = []
        for cur_pos in pos:
            try:
                wn.synset_from_pos_and_offset(cur_pos, entity)
                labels.append(cur_pos)
            except:
                pass
        if len(labels) == 0:
            print(entity + ' has no pos in synset\n')
        else:
            cur_datas = data[line_index]
            for cur_data in cur_datas:
                content_file.write('\t' + str(cur_data))
            cnt = 0
            for label in labels:
                if cnt == 0:
                    cnt = 1
                    type_file.write('\t' + label)
                    content_file.write('\t' + label)
                else:
                    type_file.write(',' + label)
                    content_file.write(',' + label)
        type_file.write('\n')
        content_file.write('\n')
        line_index += 1
    type_file.close()
    entity_file.close()
    content_file.close()
    data_file.close()


if process_cites:
    print('-------------process relation-------------')
    relation_file_train = open(relation_train_path, 'r')
    relation_file_valid = open(relation_valid_path, 'r')
    relation_file_test = open(relation_test_path, 'r')
    relation_output_file = open(relation_output_path, 'w')

    relation_lines = relation_file_train.readlines()[1:] + relation_file_valid.readlines()[1:] + relation_file_test.readlines()[1:]
    my_dic = {}
    for line in relation_lines:
        e1, e2, r =line.split()
        if (my_dic.get(e1, None) is None or e2 not in my_dic[e1]) and (my_dic.get(e2, None) is None or e1 not in my_dic[e2]):
            my_dic[e1] = my_dic.get(e1, []) + [e2]
            relation_output_file.write(str(e1) + '\t' + str(e2) + '\n')
        else:
            print(str(e2) + " is already in " + str(e1))
    relation_file_train.close()
    relation_file_valid.close()
    relation_file_test.close()
    relation_output_file.close()