import itertools
import json
import re
from nltk.corpus import wordnet as wn

# 输出文件：WN18RR.content，WN18RR.cites。
# WN18RR.content：每行第一列为entity id，最后一列为label，其他列为embedding feature。
# WN18RR.cites：每行第一列为entity1 id，第二列为entity2 id，表示e1和e2之间有关系r。
# WN18RR.rel：对rel进行embedding的结果。
content_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.content"
cites_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.cites"
rel_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.rel"

# 中间文件：WN18RR.type。
# WN18RR.type，每行第一列为entity id，第二列为label。
type_output_path = "E:/PycharmProjects/OpenKE/WN18RR_result/WN18RR.type"

# 输入文件：entity2id.txt，train2id.txt，TransE.json。
# entity2id.txt：entity id和entity编号的对应关系，由OpenKE得到。
# relation2id.txt：relation id和relation编号的对应关系，由OpenKE得到。
# train2id.txt：entity的关系，由OpenKE的得到。
# TransE.json：对entity进行TransE的embedding结果，由OpenKE的TransE得到。
entity_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/entity2id.txt"
relation_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/relation2id.txt"
train_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/train2id.txt"
valid_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/valid2id.txt"
test_path = "E:/PycharmProjects/OpenKE/benchmarks/WN18RR/test2id.txt"
data_path = "E:/PycharmProjects/OpenKE/WN18RR_result/TransE.json"

process_content = True
process_cites = True

if process_content:

    pos = ['n', 'v', 'a', 'r']

    type_file = open(type_output_path, 'w')
    entity_file = open(entity_path, 'r')
    entity_lines = entity_file.readlines()[1:]
    data_file = open(data_path, 'r')
    rel_file = open(relation_path, 'r')
    rel_lines = rel_file.readlines()[1:]
    rel_output_file = open(rel_output_path, 'w')
    data = json.load(data_file)
    data_ent = data['ent_embeddings.weight']
    data_rel = data['rel_embeddings.weight']
    content_file = open(content_output_path, 'w')
    line_index = 0

    print('-------------process content and rel-------------')
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
            cur_datas = data_ent[line_index]
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


if process_cites:
    print('-------------process cites-------------')
    train_file = open(train_path, 'r')
    valid_file = open(valid_path, 'r')
    test_file = open(test_path, 'r')
    cites_output_file = open(cites_output_path, 'w')

    triple_lines = train_file.readlines()[1:] + valid_file.readlines()[1:] + test_file.readlines()[1:]
    my_dic = {}
    for line in triple_lines:
        e1, e2, r =line.split()
        if (my_dic.get(e1, None) is None or e2 not in my_dic[e1]) and (my_dic.get(e2, None) is None or e1 not in my_dic[e2]):
            my_dic[e1] = my_dic.get(e1, []) + [e2]
            cites_output_file.write(str(e1) + '\t' + str(e2) + '\t' + str(r) + '\n')
        else:
            print(str(e2) + " is already in " + str(e1))
    train_file.close()
    valid_file.close()
    test_file.close()
    cites_output_file.close()