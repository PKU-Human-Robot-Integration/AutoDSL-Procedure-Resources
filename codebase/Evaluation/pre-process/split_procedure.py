import os
import json
import shutil
import re
from collections import defaultdict
from tqdm import tqdm

inputdir = '../AllUnityBig'
files = os.listdir(inputdir)
files.sort(key=lambda x: int(x.split('_')[1][:-5])) 
#print(files)

outputdir = '../AllUnityBigSplit'
if os.path.exists(outputdir):
    shutil.rmtree(outputdir, ignore_errors=True)
os.makedirs(outputdir, exist_ok=True)

def split_procedure_fine(prde, pattern=r'(?<=[.?!])', maxlen=300):
    chips = re.split(pattern, prde)
    cur_prde = ""
    prdes = []
    for c in chips:
        if len(cur_prde.split())+len(c.split())<=maxlen:

            cur_prde+=c
        else:
            if cur_prde!="": 
                prdes.append(cur_prde.strip())
            if len(c.split())>maxlen: 
                #print(len(c.split()))
                prdes.append(c.strip())
                cur_prde=""
            else:
                cur_prde=c
    if cur_prde!="":
        prdes.append(cur_prde.strip())
    return prdes

def split_procedure(prde, pattern=r'[\n\r]+', maxlen=300):
    chips = re.split(pattern, prde)
    cur_prde = ""
    prdes = []
    for c in chips:
        if len(cur_prde.split())+len(c.split())<=maxlen:
            if cur_prde!="":
                cur_prde+= '\n'
            cur_prde+= c
        else:
            if cur_prde!="": 
                prdes.append(cur_prde.strip())
            if len(c.split())>maxlen:
                prdes.extend(split_procedure_fine(c)) 
                cur_prde=""
            else:
                cur_prde=c
    if cur_prde!="":
        prdes.append(cur_prde.strip())
    return prdes

count = defaultdict(int)

start=0 
for file in tqdm(files):
    file = inputdir+'/'+file
    with open(file, 'r', encoding='utf-8') as fr:
        dic = json.load(fr)
        prdes=[] 
        for p in dic['procedures']:
            if len(p.split())<20: 
                #print(p)
                #print('-----------------')
                continue
            #count[len(p.split())]+=1
            prdes.extend(split_procedure(p))

        if prdes==[]:
            print(file)
            continue
        dic['procedures'] = prdes
        with open(outputdir+'/'+'protocol_'+str(start)+'.json', 'w') as fw:
            json.dump(dic, fw, indent=2, ensure_ascii=False)
            start+=1
            
print(start)

