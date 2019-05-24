import itertools
import re

entity_path = "C:/Users/14114/PycharmProjects/OpenKE/benchmarks/FB15K237/entity2id.txt"
entity_type_path = "C:/Users/14114/PycharmProjects/DKRL/data/entityType_split/entity2type.txt"
entity_description_path = "C:/Users/14114/PycharmProjects/DKRL/data/FB15k_description/FB15k_mid2description.txt"
data_path = "C‪:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/TransE.json"
content_output_path = "C:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/FB15K237.content"
relation_output_path = "C:/Users/14114/PycharmProjects/OpenKE/FB15K237_result/FB15K237.relation"

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
    starts = []
    for start_word in start_words:
        if start_word in line:
            starts.append(line.index(start_word))
    if len(starts) == 0:
        start = line.index(line.split()[1])
    else:
        start = min(starts)
    line = re.split(r'[^a-zA-Z]', line[start:])
    descriptions = list(set(line[1:]))
    entity_description[entity] = descriptions
entity_description_file.close()

# find entity type
print('-------------find entity type-------------')
entity_file = open(entity_path, 'r')
entity_lines = entity_file.readlines()[1:]
content_file = open(content_output_path, 'w')

miss_in_entity = []
nolabel_entity = []
all_labels = {}

# description在该entity的correct type里筛选一次。再将其全部type放入correct type筛选一次。
for line in entity_lines:
    entity = line.split()[0]
    if bad_case.get(entity, None) is not None:
        for label in bad_case[entity]:
            content_file.write(entity.replace('\'', ''))
            content_file.write(' ' + label.replace('\'', ''))
            all_labels[label] = all_labels.get(label, 0) + 1
        content_file.write('\n')
        continue
    if entity_type.get(entity, None) is None and entity_description.get(entity, None) is None:
        print(entity + ' doesn\'t in type or description')
        miss_in_entity.append(entity)
    content_file.write(entity.replace('\'', ''))
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
        content_file.write('\n')
        nolabel_entity.append(entity)
        continue
    for label in set(labels):
        all_labels[label] = all_labels.get(label, 0) + 1
        content_file.write(' ' + label.replace('\'', ''))
    content_file.write('\n')
entity_file.close()
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
