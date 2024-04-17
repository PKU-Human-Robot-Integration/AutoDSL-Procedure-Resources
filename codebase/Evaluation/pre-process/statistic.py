import json
import os
from collections import defaultdict


inputdir = '../AllUnityBigSplit'
files = os.listdir(inputdir)
files.sort(key=lambda x: int(x.split('_')[1][:-5])) 
#print(files)

ba2file = defaultdict(list)
ba2web = defaultdict(list)
for file in files:
    filepath = inputdir+'/'+file
    with open(filepath, 'r', encoding='utf-8') as fr:
        dic = json.load(fr)
        bigAreas = dic['bigAreas']
        for ba in bigAreas:
            ba2file[ba].append(file)
            ba2web[ba].append(dic['origin_website'])

print('------Web ownership of each domain-------')
webs = ['Nature', 'Cell', 'Bio', 'Jove', 'Wiley']
for k in ba2web:
    temp = {w:ba2web[k].count(w) for w in webs}
    print(k)
    print(temp)

print('------The number of protocols for each domain-------')
ba2num = {k:len(ba2file[k]) for k in ba2file}
print(ba2num)

print('------confusion matrix-------')
index_map = {v:idx for idx, v in enumerate(ba2num.keys())}    

overlap_matrix = [[0]*len(ba2num) for i in range(len(ba2num))]

overlap = defaultdict(int)
classes = list(ba2num.keys())
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        ti = ba2file[classes[i]]
        tj = ba2file[classes[j]]

        for k in ti:
            if k in tj:
                overlap[(classes[i], classes[j])]+=1
for i in classes: 
    ti = ba2file[i]
    tother = []
    for k,v in ba2file.items():
        if k!=i:
            tother.extend(v)
    for k in ti:
        if k not in tother:
            overlap[(i, i)]+=1
    
for k, v in overlap.items():
    i = k[0]
    j = k[1]

    overlap_matrix[index_map[i]][index_map[j]] = format(v / ba2num[i], '.3f')
    overlap_matrix[index_map[j]][index_map[i]] = format(v / ba2num[j], '.3f')


overlap_str = {}
for k, v in overlap.items():
    overlap_str[str(k)] = v

with open('ba2file_sub.json', 'w', encoding='utf-8') as fw,\
    open('ba2num_sub.json', 'w', encoding='utf-8') as fw2,\
    open('overlap_sub.json', 'w', encoding='utf-8') as fw3,\
    open('overlap-rate_sub.csv', 'w', encoding='utf-8') as fw4:
    fw.write(json.dumps(ba2file, indent=2)+'\n')
    fw2.write(json.dumps(ba2num, indent=2)+'\n')
    fw3.write(json.dumps(overlap_str, indent=2)+'\n')

    fw4.write(','.join(['']+list(index_map.keys()))+'\n')
    for k, v in index_map.items():
        temp = [k] + list(map(str, overlap_matrix[v]))
        fw4.write(','.join(temp)+'\n')