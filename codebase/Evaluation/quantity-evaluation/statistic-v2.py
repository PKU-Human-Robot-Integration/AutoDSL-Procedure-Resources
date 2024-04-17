import pandas as pd
import json
import os
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from tqdm import tqdm
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')

def get_infinitive(word, pos='v'):
    lemmatizer = nltk.WordNetLemmatizer()
    infinitive = lemmatizer.lemmatize(word, pos=pos)
    return infinitive

grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>}  # noun phrase rules
    VP: {<VB.*><NP>}      # Verb phrase rules
    Main: {<NP>?<VP>}      # (Subject) Predicate-Object Structure Rules
"""
chunk_parser = nltk.RegexpParser(grammar)

def sentence_structure(sentence): 
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    tree = chunk_parser.parse(pos_tags)
    #print(tree)

    for subtree in tree.subtrees():
        #print(type(subtree)) <class 'nltk.tree.tree.Tree'>
        if subtree.label() == 'Main': 
            return True

    return False


def get_texts(dir):
    files = os.listdir(dir)
    texts = []
    for file in files:
        input_file = dir+'/'+file
        with open(input_file, 'r') as fr:
            pros = json.load(fr)['procedures']
            text='\n'.join(pros)
        texts.append(text)
    return texts

import sys
#'../AllUnityBigSplit'
texts = get_texts(dir=sys.argv[1])
# print(texts[1])
# print(sent_tokenize(texts[1]))
# exit()


print('article nums:', len(texts))

stop = set(stopwords.words('english')) 
print(stop)



verb_nums = [] 
noun_nums = []
verb_type_nums = []
noun_type_nums = []
step_nums = []
for text in tqdm(texts):
    per_verb_map = defaultdict(int) 
    per_noun_map = defaultdict(int)
    try:
        slist = sent_tokenize(text)
        #print(len(slist))
        slist = [sent for sent in slist if sentence_structure(sent)]
        # if slist == []:
        #     continue
        #print(len(slist))
        for s in slist:
            tlist = word_tokenize(s)
            #print(tlist)
            postags = pos_tag(tlist)
            #print(postags)
            for word, tag in postags:
                word = word.lower() 
                if not word.isalpha(): 
                    continue
                if tag.startswith('VB'): 
                    word = get_infinitive(word, 'v')
                    if word not in stop:
                        per_verb_map[word]+=1
            
                elif tag.startswith('NN'):
                    word = get_infinitive(word, 'n')
                    if word not in stop:
                        per_noun_map[word]+=1
        vs = sum(per_verb_map.values())
        ns = sum(per_noun_map.values())
        step_nums.append(len(slist))
        verb_nums.append(vs)
        noun_nums.append(ns)
        verb_type_nums.append(len(per_verb_map))
        noun_type_nums.append(len(per_noun_map))
    except Exception as e:
        print('Exception')
        print(e)



import numpy as np
print('verb num mean, var, astd, sstd:', np.mean(verb_nums), np.var(verb_nums), np.std(verb_nums), np.std(verb_nums, ddof=1))
print('noun num mean, var, astd, sstd:', np.mean(noun_nums), np.var(noun_nums), np.std(noun_nums), np.std(noun_nums, ddof=1))
print('step nums mean, var, astd, sstd:', np.mean(step_nums), np.var(step_nums), np.std(step_nums), np.std(step_nums, ddof=1))
print('verbtype num mean, var, astd, sstd:', np.mean(verb_type_nums), np.var(verb_type_nums), np.std(verb_type_nums), np.std(verb_type_nums, ddof=1))
print('nountype num mean, var, astd, sstd:', np.mean(noun_type_nums), np.var(noun_type_nums), np.std(noun_type_nums), np.std(noun_type_nums, ddof=1))

