import argparse
import json
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

# nltk.download('averaged_perceptron_tagger')


def get_infinitive(verb):
    lemmatizer = nltk.WordNetLemmatizer()
    infinitive = lemmatizer.lemmatize(verb, pos='v')
    return infinitive

def general_metric(dsl, concept, dic):
    # Soundness
    count=0
    for c in concept:
        if c in dsl: 
            count+=1
            continue
        for k in dic[c]: 
            if k in dsl:
                count+=1
                break
    soundness = count/len(concept)
    soundness = format(soundness, '.4f')

    # Completeness
    count=0
    for i in dsl:
        if i in concept: 
            count+=1
            continue
        for k in dic[i]: 
            if k in concept:
                count+=1
                break
    completeness = count/len(dsl)
    completeness = format(completeness, '.4f')

    # Laconicity
    count=0 
    for i in dsl:
        maps = dic[i] + [i]
        cur=0 
        for k in maps:
            if k in concept:
                cur+=1
        if cur==1: 
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
        if cur==1:
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
        isas = json.load(fisa)
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
        for k in isas:
            slots = isas[k]
            temp_types = [] 
            for s in slots:
                temp_types.extend(s['pattern'])

            temp_types = list(set(temp_types)) 


            # syns = isas[k]['synonym']
            # syns = [op.upper() for op in syns]
            temp_ops = [k]
            temp_ops = [op.upper() for op in temp_ops]
            for op in temp_ops: 
                for t in temp_types:
                    isa_tts.append((op, t))
                    if t in regcondmap:
                        for tt in regcondmap[t]:
                            isa_tts.append((op, tt))
            

            for t in temp_types:
                cur = [t]
                if t in regcondmap:
                    cur+=regcondmap[t]
                temp = [(op, tt) for op in temp_ops for tt in cur]

                for i in temp:
                    for j in temp:
                        if j!=i:

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

        dc_tts = list(set(dc_tts)) 
        for o, t in dc_tts: 
            if o == '':
                dc_tts.remove((o, t))

        #print(dc_tts)
    
        post_tts = [] 
        for opcode, t in dc_tts:
            if ' ' in opcode: 
                continue
            pos = nltk.pos_tag([opcode])[0][1] 
            if not pos.startswith('VB'):
                continue
            opcode = opcode.lower()
            opcode = get_infinitive(opcode) 
            opcode = opcode.upper()
            temp = (opcode, t)
            if temp not in post_tts:
                post_tts.append(temp)
        print(post_tts)
        print('------done------')
        return isa_tts, post_tts, n2one2n

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--isa', type=str, default='../DSL/bioinfo/bioinformatics_and_computational_biology_dsl.json', help='DSL')
    # parser.add_argument('--dc', type=str, default='outputs/triples-Bioinformatics & Computational Biology.json', help='knowledge concepts')
    
    # parser.add_argument('--isa', type=str, default='../DSL/bioeng/bioengineering_and_technology_dsl.json', help='DSL')
    # parser.add_argument('--dc', type=str, default='outputs/triples-Bioengineering & Technology.json', help='knowledge concepts')
    
    # parser.add_argument('--isa', type=str, default='../DSL/biomed/biomedical_and_clinical_research_dsl.json', help='DSL')
    # parser.add_argument('--dc', type=str, default='outputs/triples-Biomedical & Clinical Research.json', help='knowledge concepts')

    # parser.add_argument('--isa', type=str, default='../DSL/ecolo/ecology_and_environmental_environmental_dsl.json', help='DSL')
    # parser.add_argument('--dc', type=str, default='outputs/triples-Ecology & Environmental Biology.json', help='knowledge concepts')

    parser.add_argument('--isa', type=str, default='../DSL/mole/molecular_biology_and_genetics_dsl.json', help='DSL')
    parser.add_argument('--dc', type=str, default='outputs/triples-Molecular Biology & Genetics.json', help='knowledge concepts')
    args, _ = parser.parse_known_args()


    isa_tts, dc_tts, n2one2n = get_twotuple_list(args.dc, args.isa)


    soundness, completeness, laconicity, ludicity = general_metric(isa_tts, dc_tts, n2one2n)
    print('domain:', args.isa.split('/')[-1][:-9])
    print('soundness: ', soundness)
    print('completeness: ', completeness)
    print('laconicity: ', laconicity)
    print('ludicity: ', ludicity)
    