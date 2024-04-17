import re
import openai
import itertools
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('API_BASE')

def chat(mess):
    responde = openai.ChatCompletion.create(
        model="gpt-4-0314",
        messages=mess,
    )

    res = responde['choices'][0]['message']['content']
    return res



template_joint = '''Defn: Entities are words, phrases or concepts that have a specific meaning. For a triple ('Beijing', 'part-of', 'China'), we refer to 'Beijing' as the 'head entity', 'China' as the 'tail entity', and 'part-of' as the 'relation' between the two entities. Note that the relation is directional. \
In this task, an entity type only be a 'OpCode'(operations, a one-word verb, like ADD and REMOVE); 'REG'(reagents taking part in an operation, like cells and MDDC culture media); 'COND'(conditions of executing an operation, like <temperature> 37C, <time> 30min, <device> a small scissor and <container> PCR tubes). An relation only be a 'is reagent of', 'is reaction condition of', 'is instruction of', 'is successor of', 'is predecessor of', 'is concurrent with', 'is product of'. A triple type only be ('REG', 'is reagent of', 'OpCode'); ('COND', 'is reaction condition of', 'OpCode'); ('OpCode', 'is instruction of', 'REG | COND'); ('OpCode',"is successor of | is predecessor of | is concurrent with" , 'OpCode'); ('REG', 'is product of', 'OpCode').

Q: Given the paragraph below, identify all possible entities and relations between every two entities accoriding to the paragraph. Respond as a list, e.g. [(head entity, head entity type, relation, tail entity, tail entity type), ...].

Paragraph: {}
Answer:'''

jre_list = ['is reagent of', 'is reaction condition of', 'is instruction of', 'is successor of', 'is predecessor of', 'is concurrent with', 'is product of']
jner_list = ['REG', 'OpCode', 'COND']

jtypelist = {
    'is reagent of': ['REG', 'OpCode'],
    'is reaction condition of': ['COND', 'OpCode'],
    'is instruction of': ['OpCode', 'REG/COND'],
    'is successor of': ['OpCode', 'OpCode'],
    'is predecessor of': ['OpCode', 'OpCode'],
    'is concurrent with': ['OpCode', 'OpCode'],
    'is product of': ['REG', 'OpCode'],
}

def chatJoint(query):
    mess = [{"role": "system", "content": "You are an expert assistant in entity and relation extraction and I need you to help me recognize the entities and relation between every two entities."},] # chatgpt对话历史

    prompt = template_joint.format(query)
    print(prompt)

    mess.append({"role": "user", "content": prompt})
    text = chat(mess)
    #print(text)


    res = re.findall(r'\(.*?\)', text)
    #print(res)
    if res!=[]:
        triples = [re.sub('[\'"]','',temp)[1:-1].split(',') for temp in res]
        triples = [(t[0].strip(), t[1].strip(), t[2].strip(), t[3].strip(), t[4].strip()) for t in triples if len(t)==5]
    else:
        triples=[]

    triples = [triple for triple in triples if triple[2] in jre_list] 
    triples = [triple for triple in triples if triple[1] in jner_list and triple[4] in jner_list] 
    triples = [triple for triple in triples if triple[1] in jtypelist[triple[2]][0].split('/') and triple[4] in jtypelist[triple[2]][1].split('/')] 

    stop = ['', 'none']
    triples = [t for t in triples if t[0].lower() not in stop and t[3].lower() not in stop] 
                              
    triples = list(set(triples)) 
    return triples, text

# -----------------------------------------
re_s1_p = {
    'english': '''Defn: Entities are words, phrases or concepts that have a specific meaning. For a triple ('Beijing', 'part-of', 'China'), we refer to 'Beijing' as the 'head entity', 'China' as the 'tail entity', and 'part-of' as the 'relation' between the two entities. Note that the relation is directional. In this task, an entity type may be a 'OpCode'(operations, a one-word verb, like ADD and REMOVE); 'REG'(reagents taking part in an operation, like cells and MDDC culture media); 'COND'(conditions of executing an operation, like <temperature> 37C, <time> 30min, <device> a small scissor and <container> PCR tubes). A relation type can be in the following given relations.

The given sentence is "{}".

List of given relations: {}.

What relations in the given list might be included in this given sentence?
If not present, answer: none.
Respond as a tuple, e.g. (relation 1, relation 2, ......).''',
}

re_s2_p = {
    'english': '''According to the given sentence, the two entities are of type ('{}', '{}') and the relation between them is '{}', find the two entities and list them all by group if there are multiple groups.
If not present, answer: none.
Respond in the form of a table with two columns and a header of ('{}', '{}'):''',
}

# typelist = {
#     'is reagent of': ['REG', 'OpCode'],
#     'is reaction condition of': ['COND', 'OpCode'],
#     'is instruction of': ['OpCode', 'REG/COND'],
#     'is successor of': ['OpCode', 'OpCode'],
#     'is predecessor of': ['OpCode', 'OpCode'],
#     'is concurrent with': ['OpCode', 'OpCode'],
#     'is product of': ['REG', 'OpCode'],
# }

typelist = {'is concurrent with': ['OpCode', 'OpCode'],
 'is instruction of': ['OpCode', 'REG/COND'],
 'is predecessor of': ['OpCode', 'OpCode'],
 'is product of': ['REG', 'OpCode'],
 'is reaction acceleration of': ['Acceleration', 'OpCode'],
 'is reaction centrifugal force of': ['Centrifugal Force', 'OpCode'],
 'is reaction condition of': ['COND', 'OpCode'],
 'is reaction container of': ['Container', 'OpCode'],
 'is reaction density of': ['Density', 'OpCode'],
 'is reaction device of': ['Device', 'OpCode'],
 'is reaction energy of': ['Energy', 'OpCode'],
 'is reaction flow rate of': ['Flow Rate', 'OpCode'],
 'is reaction force of': ['Force', 'OpCode'],
 'is reaction frequency of': ['Frequency', 'OpCode'],
 'is reaction iteration count of': ['Iteration Count', 'OpCode'],
 'is reaction pressure of': ['Pressure', 'OpCode'],
 'is reaction rotation of': ['Rotation', 'OpCode'],
 'is reaction speed of': ['Speed', 'OpCode'],
 'is reaction temperature of': ['Temperature', 'OpCode'],
 'is reaction time of': ['Time', 'OpCode'],
 'is reaction voltage of': ['Voltage', 'OpCode'],
 'is reagent acidity of': ['Acidity', 'OpCode'],
 'is reagent coating of': ['Coating', 'OpCode'],
 'is reagent concentration of': ['Concentration', 'OpCode'],
 'is reagent density of': ['Density', 'OpCode'],
 'is reagent length of': ['Length', 'OpCode'],
 'is reagent mass of': ['Mass', 'OpCode'],
 'is reagent medium of': ['Medium', 'OpCode'],
 'is reagent of': ['REG', 'OpCode'],
 'is reagent quantity of': ['Quantity', 'OpCode'],
 'is reagent size of': ['Size', 'OpCode'],
 'is reagent thickness of': ['Thickness', 'OpCode'],
 'is reagent volume of': ['Volume', 'OpCode'],
 'is successor of': ['OpCode', 'OpCode']}

money_tokens_input = 0
money_tokens_output = 0

def chatie_re(query, lang='english'):
    global money_tokens_input
    global money_tokens_output
    print("---RE---")
    mess = [{"role": "system", "content": "You are an expert assistant in entity and relation extraction and I need you to help me recognize the entities and relation between every two entities."},] # chatgpt对话历史

    out = [] 
    try:
        print('---stage1---')

        stage1_tl = list(typelist.keys())
        s1p = re_s1_p[lang].format(query, str(stage1_tl))
        print(s1p)
        money_tokens_input+=len(s1p.split())


        mess.append({"role": "user", "content": s1p})
        text1 = chat(mess)
        mess.append({"role": "assistant", "content": text1})
        print(text1)
        money_tokens_output+=len(text1.split())


        res1 = re.findall(r'\(.*?\)', text1)
        #print(res1)
        if res1!=[]:
            rels = [temp[1:-1].split(',') for temp in res1]
            rels = list(set([re.sub('[\'"]','', j).strip() for i in rels for j in i]))
            #print(rels)
        else: 
            text1 = text1.strip().rstrip('.')
            rels = [text1]
        #print(rels)
    except Exception as e:
        print(e)
        print('re stage 1 none out or error')
        # 'error-stage1:' + str(e)
        return [], mess

    print('---stage2')
    try:
        for r in rels:
            if r in typelist:

                st, ot = typelist[r]


                sts=st.split('/')
                ots=ot.split('/')
                tempso = []
                for st in sts:
                    for ot in ots:
                        tempso.append((st,ot))
                for st, ot in tempso:
                    s2p = re_s2_p[lang].format(st, ot, r, st, ot)
                    print(s2p)
                    money_tokens_input+=len(s2p.split())


                    mess.append({"role": "user", "content": s2p})
                    text2 = chat(mess)
                    mess.append({"role": "assistant", "content": text2})
                    print(text2)
                    money_tokens_output+=len(text2.split())


                    res2 = re.findall(r'\|.*?\|.*?\|', text2)
                    #print(res2)

                    if res2==[]:
                        res2 = re.findall(r'.*\|.*', text2)
                        #print(res2)


                    count=0
                    for so in res2:
                        count+=1
                        if count <=2: 
                            continue

                        so = so.strip('|').split('|')
                        so = [re.sub('[\'"]','', i).strip() for i in so]
                        if len(so)==2:
                            s, o = so
                            #if st in s and ot in o or '---' in s and '---' in o:
                            #    continue
                            stop = ['', 'none']
                            if s.lower() in stop or o.lower() in stop: 
                                continue 
                            out.append((s, st, r, o, ot))
                    #break
    
    except Exception as e:
        print(e)
        print('re stage 2 none out or error')
        #if out == []:
        #    out.append('error-stage2:' + str(e))
        return out, mess

    if out == []:
        #out.append('none-none')
        pass
    else:
        out = list(set(out))
    
    #print(mess)
    return out, mess

if __name__=='__main__':
    #query = '1) Cloning of BAP in pDisplay.  The BAP sequence is cloned between the N-terminal epitope of hemagglutinin A \\(HA), which is preceded by a signal sequence from the murine Ig kappa-chain V-J2-C, and the C-terminal PDGF receptor transmembrane \\(TM) domain generating the pBAP-TM plasmid encoding the HA-BAP-TM protein \\(BAP-TM reporter)<sup>1</sup>'
    #query = '''Polymerase chain reaction. The PSTCD BAP sequence \\(387 bp) is first amplified by PCR from pXa-1 plasmid \\(Promega, Madison, WI) as follows: for each reaction, add 5 µl  Pfu buffer \\(10x buffer), 2.5 µl of each upstream and downstream primer \\(10 μM), 1.5 µl dNTPs mix \\(5 mM), 1 µl  pXa-1 \\(20 mg/L), 5 µl DMSO \\(Critical, PCR of BAP is not successful without it), 1 µl Pfu DNA polymerase \\(2.5 units/µl) and increase the volume to 50 µl with autoclaved ddH\n2\nO. Set the conditions for the PCR as follows: an initial denaturation step at 95\no\nC for 3 min followed by 35 cycles of 30 sec 95\no\nC denaturation, 30 sec 48\no\nC annealing, 1 min 72\no\nC extension and a final extension step of 10 min.'''
    

    import argparse
    import json
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='chatie', type=str)
    parser.add_argument('--input', default='../AllUnityBigSplit', type=str)
    parser.add_argument('--output', default='outputs/triples-all.json', type=str)
    parser.add_argument('--output_dir', default='../AllUnityBigSplitOntology', type=str)
    args,_ = parser.parse_known_args()

    if args.method == 'joint':
        method = chatJoint
    elif args.method == 'chatie':
        method = chatie_re
    else:
        exit('error method:{}'.format(args.method))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    filenames = os.listdir(args.input)
    random.seed(99) 
    random.shuffle(filenames) 
    #print(filenames[:5])
    #exit()
    
    for fn in filenames[550:600]:
        print('-----------------------------------------------')
        print(fn)
        print(filenames.index(fn))
        print('-----------------------------------------------')
        fr = open(args.input+'/'+fn, 'r')
        dic = json.load(fr)
        procedures = dic['procedures']

        with open(args.output, 'a') as fw:
            cur_triples = []
            for prot in procedures:
                triples, mess = method(prot)
                cur_triples.extend(triples)
            cur_triples = list(set(cur_triples))
            fw.write(json.dumps(cur_triples, ensure_ascii=False)+'\n')
            fw.flush()
        with open(args.output_dir+'/'+fn, 'w') as fw:
            dic['ontology'] = cur_triples
            json.dump(dic, fw, indent=2, ensure_ascii=False)
    
    print('total cost:$', money_tokens_input/1000*0.01+money_tokens_output/1000*0.03) 
        


