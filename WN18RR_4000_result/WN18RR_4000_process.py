import numpy as np


dataset_str = "WN18RR"
entity_path = "../{}_result/{}.content".format(dataset_str, dataset_str)
relation_path = "../{}_result/{}.rel".format(dataset_str, dataset_str)
cites_path = "../{}_result/{}.cites".format(dataset_str, dataset_str)

content_output_path = "{}_4000.content".format(dataset_str)
rel_output_path = "{}_4000.rel".format(dataset_str)
cites_output_path = "{}_4000.cites".format(dataset_str)

idx_features_labels = np.genfromtxt(entity_path, dtype=np.dtype(str))
edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)

selected_entity = set([17])
selected_relations = set()
step = 4

while step:
    new_selected_nodes = set()
    for index in range(len(edges_unordered)):
        if edges_unordered[index][0] in selected_entity or edges_unordered[index][1] in selected_entity:
            new_selected_nodes.add(edges_unordered[index][0])
            new_selected_nodes.add(edges_unordered[index][1])
    selected_entity |= new_selected_nodes

    step -= 1
# print("len(selected_entity): {}, len(selected_relations): {}".format(len(selected_entity), len(selected_relations)))

for index in range(len(edges_unordered)):
    if edges_unordered[index][0] in selected_entity and edges_unordered[index][1] in selected_entity:
        selected_relations.add(edges_unordered[index][2])

entity_file = open(entity_path, 'r')
entity_lines = entity_file.readlines()
rel_file = open(relation_path, 'r')
rel_lines = rel_file.readlines()
cites_file = open(cites_path, 'r')
cites_lines = cites_file.readlines()

content_output_file = open(content_output_path, 'w')
rel_output_file = open(rel_output_path, 'w')
cites_output_file = open(cites_output_path, 'w')

selected_entity = set(selected_entity)
selected_relations = set(selected_relations)

for line in entity_lines:
    entityid = int(line.split('\t')[1])
    if entityid in selected_entity:
        content_output_file.write(line)
content_output_file.close()

for line in rel_lines:
    relid = int(line.split('\t')[1])
    if relid in selected_relations:
        rel_output_file.write(line)
rel_output_file.close()

for line in cites_lines:
    entityid_1, entityid_2 = int(line.split('\t')[0]), int(line.split('\t')[1])
    if entityid_1 in selected_entity and entityid_2 in selected_entity:
        cites_output_file.write(line)
cites_output_file.close()
