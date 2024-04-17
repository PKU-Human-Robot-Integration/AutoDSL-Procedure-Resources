import argparse
import json
import nltk
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict
# Download the required NLTK data
# nltk.download('averaged_perceptron_tagger')


def get_infinitive(verb):
    lemmatizer = nltk.WordNetLemmatizer()
    infinitive = lemmatizer.lemmatize(verb, pos='v')
    return infinitive

def general_metric(dsl, concept, dic):
    # Soundness
    count=0
    for c in concept:
        if c in dsl: # One-to-one matching
            count+=1
            continue
        for k in dic[c]: # Matching synonyms
            if k in dsl:
                count+=1
                break
    soundness = count/len(concept)
    soundness = format(soundness, '.4f')

    # Completeness
    count=0
    for i in dsl:
        if i in concept: # One-to-one matching
            count+=1
            continue
        for k in dic[i]: # Matching synonyms
            if k in concept:
                count+=1
                break
    completeness = count/len(dsl)
    completeness = format(completeness, '.4f')

    # Laconicity
    count=0 # Number of 1-to-n
    for i in dsl:
        maps = dic[i] + [i]
        cur=0 
        for k in maps:
            if k in concept:
                cur+=1
        if cur==1: # Having more than two is a 1-to-n
            count+=1
    laconicity = count/len(dsl)
    laconicity = format(laconicity, '.4f')
    #print(count)

    # Ludicity
    count=0 
    for i in concept:
        maps = dic[i] + [i]
        cur=0 
        for k in maps:
            if k in dsl:
                cur+=1
        if cur==1: # Having more than two is a 1-to-n
            count+=1
    ludicity = count/len(concept)
    ludicity = format(ludicity, '.4f')
    #print(count)
    
    return soundness, completeness, laconicity, ludicity



def get_twotuple_list(dc, isa):
    with open(dc, 'r') as fdc,\
        open(isa, 'r') as fisa:
        print('---get-twotuple----')
        isa_tts = []
        with open(isa, 'r') as fr:
            isas = fr.read().split('\n\n\n\n')
        print(len(isas))

        n2one2n = defaultdict(list)
        regcondmap = {
            'Volume': ['REG'],
            'Temperature':['COND'],
            'Length': ['REG'],
            'Energy': ['COND'],
            'Concentration': ['REG'],
            'Mass': ['REG'],
            'Acidity': ['REG'],
            'Flow Rate': ['COND'],
            'Centrifugal Force': ['COND'],
            'Frequency': ['COND'],
            'Thickness': ['REG'],
            'Quantity': ['REG'],
            'Size': ['REG'],
            'Force': ['COND'],
            'Pressure': ['COND'],
            'Acceleration': ['COND'],
            'Density': ['REG', 'COND'],
            'Speed': ['COND'],
            'Medium': ['REG'],
            'Coating': ['REG'],
            'Iteration Count': ['COND'],
            'Rotation': ['COND'],
            'Voltage': ['COND'],
            'Device': ['COND'],
            'Container': ['COND'],
        }
        biomap={
            'Fluid':'REG',
            'Slide':'COND',
            'Plate':'COND',
            'Solid': 'REG'
        }
        for func in isas:
            #print(func)
            res = re.search(r'(Symbol|void)(.*?)(\(.+?\))', func, re.DOTALL)
            if res and len(res.groups())==3:
                raw_op = res.group(2).strip()
                raw_types = res.group(3).strip()
                raw_types = raw_types.split(',')
                raw_types = [re.sub(r'\s', '', re.split(r'\*|&', ty)[0]).strip('()') for ty in raw_types]
                raw_types = [ty for ty in raw_types if ty not in ['', 'char']]
                raw_types = [biomap[ty] if ty in biomap else ty for ty in raw_types]
                print(raw_op, raw_types)
                if raw_types==[]:
                    continue
                
            else:
                continue
            

            temp_types = raw_types # This directive involves the parameter type, in order to record the two-tuple and build the map with the

            temp_types = list(set(temp_types)) # de-emphasize

            # Record binary
            # syns = isas[k]['synonym']
            # syns = [op.upper() for op in syns]
            temp_ops = [raw_op]
            temp_ops = [op.upper() for op in temp_ops]
            for op in temp_ops: # Add the two tuples from syns.
                for t in temp_types:
                    isa_tts.append((op, t))
                    if t in regcondmap:
                        for tt in regcondmap[t]:
                            isa_tts.append((op, tt))
            
            # Record dictionary for easy calculation of metrics such as n-to-1
            
            # opcode synonym not null OR parameter type has mappable
            for t in temp_types:
                cur = [t]
                if t in regcondmap:
                    cur+=regcondmap[t]
                temp = [(op, tt) for op in temp_ops for tt in cur]

                # Expand temp, map t a bit
                for i in temp:
                    for j in temp:
                        if j!=i:
                            # Avoid reverse mappings in the mapping table, e.g. ('REMOVE', 'COND'): [('REMOVE', 'Device')]
                            if i[1] in ['COND', 'REG'] and j[1] not in ['COND', 'REG']:
                                continue
                            n2one2n[i].append(j)
    
        print(n2one2n)

        isa_tts = list(set(isa_tts))
        print(isa_tts)
       
        dc_tts = []
        rlist = ['is reagent of', 'is reaction condition of', 'is reagent volume of', 'is reaction temperature of', 'is reagent length of', 'is reaction energy of', 'is reagent concentration of', 'is reagent mass of', 'is reagent acidity of', 'is reaction flow rate of', 'is reaction centrifugal force of', 'is reaction frequency of', 'is reagent thickness of', 'is reagent quantity of', 'is reagent size of', 'is reaction force of', 'is reaction pressure of', 'is reaction acceleration of', 'is reagent density of', 'is reaction density of', 'is reaction speed of', 'is reagent medium of', 'is reagent coating of', 'is reaction iteration count of', 'is reaction rotation of', 'is reaction voltage of', 'is reaction device of', 'is reaction container of', 'is reaction time of']
        for line in fdc.readlines():
            line = json.loads(line.strip())
            for t in line:
                if t[2] in rlist:
                    dc_tts.append((t[3], t[1]))

        dc_tts = list(set(dc_tts)) # de-duplication
        for o, t in dc_tts: # Delete null
            if o == ''.
                dc_tts.remove((o, t))

        # print(dc_tts)
    
        post_tts = [] # post-process, find prototype, capitalize
        for opcode, t in dc_tts.
            if ' ' in opcode: # just single word
                continue
            pos = nltk.pos_tag([opcode])[0][1] # just the verb
            if not pos.startswith('VB'):: # just verbs
                continue
            opcode = opcode.lower()
            opcode = get_infinitive(opcode) # find prototype, lowercase only, uppercase doesn't work
            opcode = opcode.upper()
            temp = (opcode, t)
            if temp not in post_tts.
                post_tts.append(temp)
        print(post_tts)
        print('------done------')
        return isa_tts, post_tts, n2one2n

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isa', type=str, default='../rag-chatgpt/biocoder_function.txt', help='DSL')
    parser.add_argument('--dc', type=str, default='outputs/triples-Molecular Biology & Genetics.json', help='knowledge concepts')
    args, _ = parser.parse_known_args()

    # Get a list of twotuples
    isa_tts, dc_tts, n2one2n = get_twotuple_list(args.dc, args.isa)

    # Calculate the binary group metrics
    soundness, completeness, laconicity, ludicity = general_metric(isa_tts, dc_tts, n2one2n)
    print('domain:', args.isa.split('/')[-1][:-9])
    print('soundness: ', soundness)
    print('completeness: ', completeness)
    print('laconicity: ', laconicity)
    print('ludicity: ', ludicity)
    