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

f = open('FB15K237.rel')
print("FB15K237.rel:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 1)
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

f = open('../WN18RR_result/WN18RR.rel')
print("WN18RR.rel:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 1)
print('\n')

f = open('../WN18RR_result/WN18RR_sub30000.content')
print("WN18RR_sub30000.content:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 2)

f = open('../WN18RR_result/WN18RR_sub30000.cites')
print("WN18RR_sub30000.cites:")
lines = f.readlines()
print(len(lines))

f = open('../WN18RR_result/WN18RR_sub30000.rel')
print("WN18RR_sub30000.rel:")
lines = f.readlines()
print(len(lines))
line = lines[1]
print(len(line.split()) - 1)
