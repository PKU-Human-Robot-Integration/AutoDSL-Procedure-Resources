import nltk
import os
import json
from collections import defaultdict
from tqdm import tqdm

def get_infinitive(word, pos='v'):
    lemmatizer = nltk.WordNetLemmatizer()
    infinitive = lemmatizer.lemmatize(word, pos=pos)
    return infinitive

inputdir = '../AllUnityBigSplit-Molecular Biology & Genetics'
files = os.listdir(inputdir)
files.sort(key=lambda x: int(x.split('_')[1][:-5])) 
#print(files)


stopwords = ['be']
k=1

res = defaultdict(list)

for file in tqdm(files):
    filename = inputdir+'/'+file
    with open(filename, 'r') as fr:
        dic = json.load(fr)
    #print(dic)
    procedures = dic['procedures']
    for pro in procedures:
        sents = nltk.sent_tokenize(pro)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            poses = nltk.pos_tag(words)
            for i in range(k):
                if poses[i][1].startswith('VB'):
                    opcode = get_infinitive(poses[i][0].lower()).upper()
                    if opcode.lower() not in stopwords:
                        res[opcode].append(sent)
                    break

with open('2_prefix.json', 'w') as fw:
    json.dump(res, fw, indent=2, ensure_ascii=False)
            

