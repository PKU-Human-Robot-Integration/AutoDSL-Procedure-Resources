import os
import json
import shutil
import re
from collections import defaultdict
from tqdm import tqdm

inputdir = '../AllUnityBigSplit'
files = os.listdir(inputdir)
files.sort(key=lambda x: int(x.split('_')[1][:-5])) 
#print(files)



domains = [
    "Biomedical & Clinical Research",
    "Bioengineering & Technology",
    "Molecular Biology & Genetics",
    'Ecology & Environmental Biology',
    'Bioinformatics & Computational Biology'
]

for do in domains:
    outputdir = '../AllUnityBigSplit'+'-'+do
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir, ignore_errors=True)
    os.makedirs(outputdir, exist_ok=True)



for file in tqdm(files):
    filein = inputdir+'/'+file
    with open(filein, 'r', encoding='utf-8') as fr:
        print(file)
        dic = json.load(fr)
    bigareas = dic['bigAreas']
    for area in bigareas:
        fileout = '../AllUnityBigSplit'+'-'+ area +'/'+file
        with open(fileout, 'w', encoding='utf-8') as fw:
            json.dump(dic, fw, indent=2, ensure_ascii=False)

        
