import numpy as np
import scipy.sparse as sp


dataset_str = "FB15K237"
min = 0
min_num = 100000
for i in range(14541):
    selected_nodes = set([10])
    selected_relations = set()
    step = 2
    flag = 0

    idx_features_labels = np.genfromtxt("../{}_result/{}.content".format(dataset_str, dataset_str), dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("../{}_result/{}.cites".format(dataset_str, dataset_str), dtype=np.int32)

    while step:
        new_selected_nodes = selected_nodes.copy()
        new_selected_relations = selected_relations.copy()
        for index in range(len(edges_unordered)):
            if edges_unordered[index][0] in selected_nodes or edges_unordered[index][1] in selected_nodes:
                new_selected_nodes.add(edges_unordered[index][0])
                new_selected_nodes.add(edges_unordered[index][1])
                new_selected_relations.add(edges_unordered[index][2])
            if len(new_selected_nodes) > 1000:
                flag = 1
                break
        if flag:
            break
        selected_nodes = new_selected_nodes.copy()
        selected_relations = new_selected_relations.copy()

        step -= 1
    if flag:
        continue
    if len(selected_nodes) < min_num:
        min_num = len(selected_nodes)
        min = i
    print("i: {}, len(selected_nodes): {}, len(selected_relations): {}".format(i, len(selected_nodes), len(selected_relations)))
print("min i: {}, min num: {}".format(min, min_num))
