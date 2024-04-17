import pandas as pd
import json

df = pd.read_csv('wikihowAll.csv').dropna(subset=['text'])

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
    NP: {<DT>?<JJ>*<NN.*>}  # Noun phrase rules
    VP: {<VB.*><NP>}      # Verb Phrase Rule
    Main: {<NP>?<VP>}      # Rules for (subject)-predicate-object constructions
"""
chunk_parser = nltk.RegexpParser(grammar)

def sentence_structure(sentence): # Determine whether it is a (subject-)predicate-object construction
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    tree = chunk_parser.parse(pos_tags)
    #print(tree)

    for subtree in tree.subtrees():
        #print(type(subtree)) <class 'nltk.tree.tree.Tree'>
        if subtree.label() == 'Main': # The presence of a main webin structure in the sentence is fine
            return True

    return False


texts = df['text'].tolist()
# print(texts[0])
# print(sent_tokenize(texts[0]))
# exit()


print('article nums:', len(texts))

stop = set(stopwords.words('english')) 
print(stop)



verb_nums = [] # Record the number of verbs per article
noun_nums = []
verb_type_nums = []
noun_type_nums = []
step_nums = []
for text in tqdm(texts):
    per_verb_map = defaultdict(int) # Verb statistics for each article
    per_noun_map = defaultdict(int)
    try:
        slist = sent_tokenize(text)
        #print(len(slist))
        slist = [sent for sent in slist if sentence_structure(sent)]
        #print(len(slist))
        for s in slist:
            tlist = word_tokenize(s)
            #print(tlist)
            postags = pos_tag(tlist)
            #print(postags)
            for word, tag in postags:
                word = word.lower() # Easy to count and find prototypes
                if not word.isalpha(): # Not plain letters. Skip.
                    continue
                if tag.startswith('VB'): # Verb
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

