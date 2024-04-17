import os
import json
import shutil
from collections import defaultdict

c2big = {} 
with open('c2big.txt', 'r', encoding='utf-8') as fr:
    lines = fr.read().strip().split('\n\n\n')
    lines = [list(map(lambda x:x.strip(), line.strip().split('\n'))) for line in lines]
    for line in lines:
        Big = line[0]
        for c in line[1:]:
            c2big[c] = Big
#print(c2big) 

inputdir = '../AllUnity'
files = os.listdir(inputdir)
files.sort(key=lambda x: int(x.split('_')[1][:-5])) 
#print(files)

outputdir = '../AllUnityBig'
if os.path.exists(outputdir):
    shutil.rmtree(outputdir, ignore_errors=True)
os.makedirs(outputdir, exist_ok=True)


count=0 
webcount=defaultdict(int) 
filtered = set() 

start=0 
for file in files:
    file = inputdir+'/'+file
    with open(file, 'r', encoding='utf-8') as fr:
        dic = json.load(fr)
        subjectAreas = dic['subjectAreas']
        bigAreas = list(set([c2big[c] for c in subjectAreas if c in c2big]))
        dic['bigAreas'] = bigAreas
        if bigAreas==[]: 
            count+=1
            webcount[dic['origin_website']]+=1
            for i in dic['subjectAreas']:
                filtered.add(i)
            continue
        with open(outputdir+'/'+'protocol_'+str(start)+'.json', 'w') as fw:
            json.dump(dic, fw, indent=2, ensure_ascii=False)
            start+=1

print('Number of protocols after filtering:', start)
#print(filtered)
print('empty bigAreas nums: ', count)
print(webcount)
        




