
import json
import nltk
tfile = 'outputs/triples-all.json'

def get_infinitive(verb):
    lemmatizer = nltk.WordNetLemmatizer()
    infinitive = lemmatizer.lemmatize(verb, pos='v')
    return infinitive


def checkpoint_print(triples):
    triples = list(set(triples))

    print('triples nums:', len(triples))


    threes = list(set([(st,r,ot) for s,st,r,o,ot in triples]))
    print('st,r,ot nums:', len(threes))


    twos = list(set([(st,ot) for s,st,r,o,ot in triples]))
    print('st,ot nums:', len(twos))


    rlist = ['is reagent of', 'is reaction condition of', 'is reagent volume of', 'is reaction temperature of', 'is reagent length of', 'is reaction energy of', 'is reagent concentration of', 'is reagent mass of', 'is reagent acidity of', 'is reaction flow rate of', 'is reaction centrifugal force of', 'is reaction frequency of', 'is reagent thickness of', 'is reagent quantity of', 'is reagent size of', 'is reaction force of', 'is reaction pressure of', 'is reaction acceleration of', 'is reagent density of', 'is reaction density of', 'is reaction speed of', 'is reagent medium of', 'is reagent coating of', 'is reaction iteration count of', 'is reaction rotation of', 'is reaction voltage of', 'is reaction device of', 'is reaction container of', 'is reaction time of']

    temp_triples = []
    for s, st, r, o, ot in triples:
        if r not in rlist:
            continue
        if ' ' in o: 
            continue

        pos = nltk.pos_tag([o])[0][1] 
        if not pos.startswith('VB'):
            continue
        o = get_infinitive(o.lower()).upper()
        temp_triples.append((s,st,r,o,ot))

    triples=temp_triples



    threes = list(set([(s,r,o) for s,st,r,o,ot in triples]))
    print('s,r,o nums:', len(threes))


    twos = list(set([(s,o) for s,st,r,o,ot in triples]))
    print('s,o nums:', len(twos))


    twos = list(set([(st,o) for s,st,r,o,ot in triples]))
    print('xxtype,opcode nums:', len(twos))

    
    ones = list(set([o for s,st,r,o,ot in triples]))
    print('opcode nums:', len(ones))
    #print(ones)
    

triples = []

with open(tfile, 'r') as fr:
    for idx, line in enumerate(fr.readlines()):
        line = json.loads(line.strip())
        for s, st, r, o, ot in line:
            triples.append((s,st,r,o,ot))

        if (idx+1)%10==0:
            print('-------------------')
            print('checkpoint:', idx+1)
            checkpoint_print(triples)
            print('------------')