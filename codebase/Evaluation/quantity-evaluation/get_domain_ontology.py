import json
import os
from collections import defaultdict


inputdir = '../AllUnityBigSplitOntology'
files = os.listdir(inputdir)

odic = defaultdict(list)
numdic = defaultdict(int)
for file in files:
    filein = inputdir+'/'+file
    with open(filein, 'r') as fr:
        dic = json.load(fr)
        ontology = dic['ontology']
        areas = dic['bigAreas']
        for area in areas:
            odic[area].extend(ontology)
            numdic[area]+=1
for k, onto in odic.items():
    with open('outputs/'+'triples-'+k+'.json', 'w') as fw:
        print(k, len(onto))
        fw.write(json.dumps(onto, ensure_ascii=False)+'\n')
        fw.flush()
print(numdic)