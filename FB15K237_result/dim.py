f = open('../cora_result/cora.content')
print("cora.content:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 2)

f = open('../cora_result/cora.cites')
print("cora.cites:")
lines = f.readlines()
print(len(lines))
print('\n')

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
print('\n')

f = open('../WN18RR_result/WN18RR.content')
print("WN18RR.content:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 2)

f = open('../WN18RR_result/WN18RR.cites')
print("WN18RR.cites:")
lines = f.readlines()
print(len(lines))
