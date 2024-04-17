from langchain.embeddings import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
import re
import argparse
import json
import random
import csv
from tqdm import tqdm
import time

def chat(mess):
    responde = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        #model="gpt-4-0314",
        messages=mess,
    )

    res = responde['choices'][0]['message']['content']
    return res


rate_template = '''[Background]
Domain experts expect to describe the experimental protocols of the specific domain with the DSL programs.
Domain experts expect to be guided by the DSL programs step-by-step, where the input, output, and configurations of each step is well-detailed.
Domain experts come without any training on programming.
Domain experts are subjective and their mindsets come from the specific domain.

[Instruction]
The [Rating Basis] presents a series of statements or questions about the [DSL]. Please evaluate your opinion and confidence level in the statements or questions referring to related domain knowledge.
That is, your evaluation should be based primarily [Rating Basis]. You should refer to and use [Domain Knowledge] to help you evaluate. Begin your evaluation by providing a short explanation. After providing your explanation, please rate the [DSL] on [Rating Criteria] by strictly following this format: "Rating: [[<opinion>, <confidence>]]", for example: "Rating: [[3, 4]]".

[Rating Criteria]
For each item, you need give an <Opinion> and <Confidence> rate:
   1. Opinion on a 1-5 Scale (int): Rate each item on a scale where 1 indicates 'Strongly Disagree' and 5 indicates 'Strongly Agree'. The scale is nuanced as follows:
  - 1: Strong Disagreement or Major Issues
  - 2: Disagreement or Notable Concerns
  - 3: Neutral or Mixed Feelings
  - 4: Agreement or Minor Concerns
  - 5: Strong Agreement or Highly Favorable
   2. Confidence in Your Response on a 1-5 Scale (int): Indicate your level of confidence in your response to each item, where 1 is 'Not Confident at All' and 5 is 'Extremely Confident'. The scale implies:
  - 1: Very Low Confidence
  - 2: Low Confidence
  - 3: Moderate Confidence
  - 4: High Confidence
  - 5: Very High Confidence

[Rating Basis]
{}

[DSL]
{}

[Domain Knowledges]
{}

[Output]
Let's first understand/follow the [Instruction] to evaluate the [DSL] and give an explanation about rating.
Then let's give a final rating based on the explanation.
'''


def myqa(query, rate_basis):
    docs = rdb.similarity_search(query, k=10) 

    count=0
    threshold = 10
    while True:
        try:
            context = [doc.page_content for doc in docs]

            context ="\n\n".join(context)
            prompt = rate_template
            prompt = prompt.format(rate_basis, query, context)
            print(prompt)

            mess = [{"role": "system", "content": "Please act as an impartial judge and evaluate the given developed Domain-Specific Language ([DSL]) according to the [Rating Basis], [Rating Criteria], and related [Domain Knowledges]. Please follow the [Instruction]. Moreover, the [Background] describes something about Domain-Specific Language ([DSL])."},] # chatgpt对话历史
            mess.append({"role": "user", "content": prompt})

            text = chat(mess)
            print('Output: ', text)
            print('-----------------------------------------')

            scores = re.search(r'\[\[(.+?),(.+?)\]\]', text)
            opinion, confident = int(scores.group(1).strip()), int(scores.group(2).strip())
            break
        except Exception as e:
            print(e)
            count+=1
            if count>=threshold:
                text=''
                opinion=0
                confident=0
                break
    return text, (opinion, confident)


def my_csv_writer(filename, header, data, delimiter='\t'):
    with open(filename, 'w') as fw:
        fw = csv.writer(fw, delimiter=delimiter)
        fw.writerow(header)
        for d in data:
            fw.writerow(d)
        




if __name__=='__main__':
    random.seed(42) 

    parser = argparse.ArgumentParser()
    parser.add_argument('--isa', type=str, default='../DSL/bioinfo/bioinformatics_and_computational_biology_dsl.json', help='DSL')
    parser.add_argument('--isa_type', type=str, default='json')
    parser.add_argument('--dk', type=str, default='docs_all_1000_100', help='domian knowledge')
    parser.add_argument('--ratebs', type=str, default='rate_basis.txt')
    parser.add_argument('--k', type=int, default=5, help='group window')
    parser.add_argument('--fw', type=str, default='outputs2/result-temp-3.5.csv')
    args, _ = parser.parse_known_args()


    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.api_base = os.getenv('API_BASE')

    embeddings_model = OpenAIEmbeddings()

    persist_directory = args.dk 

    rdb = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)

    print('Number of vectors:', rdb._collection.count())
    print('Vector loaded!')


    print('DSL:', args.isa)

    with open(args.isa, 'r') as fr1,\
        open(args.ratebs, 'r') as fr2:
        if args.isa_type=='txt':
            isalist = fr1.read().strip().split('\n\n\n\n')
            print('ISA number:', len(isalist))
            isalist = isalist * args.k
            print('*The number of ISAs after k:', len(isalist))
        elif args.isa_type=='json':
            isa = json.load(fr1)
            isalist = []
            for k, v in isa.items():
                isalist.append({k:v})
            print('ISA number:', len(isalist))
            isalist = isalist * args.k
            print('*The number of ISAs after k:', len(isalist))


        isa_groups=[]
        random.shuffle(isalist) 
        while isalist:
            temp = [] 
            curgroup = [] 
            for idx, dic in enumerate(isalist):
                if dic not in curgroup:
                    curgroup.append(dic)
                    if len(curgroup)==args.k:
                        temp.extend(isalist[idx+1:])
                        break
                else:
                    temp.append(dic)
            
            isalist=temp
            isa_groups.append(curgroup)
        print('Number of groups after grouping: ', len(isa_groups))
        # print(isa_groups[0])
        # exit()   

        rbs = fr2.read().strip().split('\n')
        print('rate basis number:', len(rbs))

    #rbs=rbs[0:2] #debug
    header = ['id', 'dsl']
    for i in range(len(rbs)):
        header.extend(['res-'+str(i), 'rate-'+str(i)])
    header.append('rate-total')

    data = []

    filen = open(args.fw, 'a')
    fw = csv.writer(filen, delimiter='\t')
    fw.writerow(header)
    filen.flush()

    for idx, dsl in tqdm(enumerate(isa_groups[:5])):
        curd = [idx]
        if args.isa_type=='txt':
            dsl = '\n\n\n'.join(dsl)

        curd.append(dsl)
        query = str(dsl)
        totalop, totalco = 0, 0
        for rb in rbs:
            text, scores = myqa(query, rb)
            #time.sleep(5)
            curd.append(text)
            curd.append(scores)
            totalop+=scores[0]
            totalco+=scores[1]
        curd.append((totalop/len(rbs), totalco/len(rbs)))
        data.append(curd)
        fw.writerow(curd)
        filen.flush()




    # debug
    # query = [{'REPEAT': {'slot': {'begin': {'require': 1, 'type': 'NUM', 'commit': ''}, 'end': {'require': 1, 'type': 'NUM', 'commit': ''}, 'repeat': {'require': 1, 'type': 'NUM', 'commit': ''}}, 'emit': {}, 'example': ['[Repeat]{operation} [steps 11]{begin}-[13]{end} to [collect more samples]{purpose}.', '[Repeat]{operation} [wash]{predict begin and end}.', '[Repeat]{operation} [this step]{predict begin and end} [one more time]{repeat}.'], 'notice': ['REPEAT cannot handle nested loops (Actually, no nested loop is actually observed).', 'The interval composed of begin and end is a closed interval'], 'synonym': []}}, {'STERILIZE': {'slot': {'reagent1': {'require': 1, 'type': 'REG', 'commit': 'The reagent to be sterilized'}, 'reagent2': {'require': 0, 'type': 'REG', 'commit': 'Sterilant'}, 'component': {'require': 0, 'type': 'STR', 'commit': ''}, 'device': {'require': 0, 'type': 'STR', 'commit': ''}, 'time': {'require': 0, 'type': 'TIME', 'commit': ''}, 'temperature': {'require': 0, 'type': 'TEMP', 'commit': ''}}, 'emit': {'reagent0': {'require': 1, 'type': 'REG', 'commit': ''}}, 'example': ['[Sterilize]{operation} the [both sides]{component} of [flanks]{reagent1} with [a Betadine dipped gauze sponge]{reagent2}.', '[Sterilize]{operation} [the medium]{reagent1} in [an autoclave]{device} at [110C]{tempurature} for [15 minutes]{time}.', '[Sterilize]{operation} [20-50 seeds]{reagent1} for [30s]{time} with [20 ml of 75% ethanol]{reagent2} in [a sterile Petri dish with a lid]{device}.'], 'notice': ['Sometimes reagents are used for sterilization, and sometimes equipment is used instead of sterilant, so reagent2 is optional.', 'Sometimes only one part of reagent1 is sterilized (such as mouse skin), so the parameter "component" should be added'], 'synonym': []}}, {'PRECIPITATE': {'slot': {'reagent1': {'require': 1, 'type': 'REG', 'commit': 'The reagent to be precipitated, keep alive'}, 'reagent2': {'require': 0, 'type': 'REG', 'commit': 'The precipitant'}, 'component': {'require': 1, 'type': 'STR', 'commit': ''}, 'time': {'require': 0, 'type': 'TIME', 'commit': ''}, 'temperature': {'require': 0, 'type': 'TEMP', 'commit': ''}, 'method': {'require': 0, 'type': 'STR', 'commit': ''}}, 'emit': {'reagent0': {'require': 1, 'type': 'REG', 'commit': ''}}, 'example': ['[Precipitate]{operation} the [RNA]{component} [overnight]{time} with [one volume of 5 M lithium chloride]{reagent2} at [-80°C]{temperature}.', '[Precipitate]{operation} [DNA]{component} by [adding  3.5 ml or  10.5 ml  of room temperature isopropanol to the eluted DNA]{method}.', '[Precipitate]{operation} [DNA]{component} at [-20&#xB0;C]{temperature} for [at least 30 min]{time}.'], 'notice': ['The precipitate must be a part of the original reagent reagent1, so use "component" to indicate the part to precipitate', 'There is a large amount of instruction nesting ("by adding", etc.), which is putted in "method".'], 'synonym': []}}, {'SHAKE': {'slot': {'reagent1': {'require': 1, 'type': 'REG', 'commit': 'The reagent to be shaked'}, 'speed': {'require': 0, 'type': 'SPD', 'commit': ''}, 'time': {'require': 0, 'type': 'TIME', 'commit': ''}, 'method': {'require': 0, 'type': 'STR', 'commit': ''}, 'device': {'require': 0, 'type': 'STR', 'commit': ''}, 'repeat': {'require': 0, 'type': 'NUM', 'commit': 'This parameter is usually given if SHAKE is performed by hand'}, 'temperature': {'require': 0, 'type': 'TEMP', 'commit': ''}}, 'emit': {'reagent0': {'require': 1, 'type': 'REG', 'commit': ''}}, 'example': ['[Shake]{operation} at [moderate speed]{speed} for [approximately 30 minutes]{time}.', '[Shake]{operation} the [tubes]{reagent1} [vigorously]{speed} by [hand]{method}.', '[Shake]{operation} [PCR clean-up plate]{reagent1} using the [plate shaker]{device} at [1800 rpm]{speed} for [2 minutes]{time}.'], 'notice': ['For most of times, SHAKE uses rpm or xg to represent strength, which is similar to CENTRIFUGATION.'], 'synonym': []}}, {'MIX': {'slot': {'reagent1': {'require': 1, 'type': 'List<REG>', 'commit': 'Contains at least one reagent'}, 'method': {'require': 0, 'type': 'STR', 'commit': ''}, 'container': {'require': 0, 'type': 'STR', 'commit': ''}, 'purpose': {'require': 0, 'type': 'STR', 'commit': ''}, 'repeat': {'require': 0, 'type': 'NUM', 'commit': ''}, 'notice': {'require': 0, 'type': 'STR', 'commit': ''}}, 'emit': {'reagent0': {'require': 1, 'type': 'REG', 'commit': ''}}, 'example': ['[Mix]{operation} by [pipetting carefully]{method} to [make the stock of the single cell suspension]{purpose}.', '[Mix]{operation} [20 μl of 10× PCR buffer]{reagent1[0]}, [2 μl of 250 mM MgCl-2]{reagent1[1]}, [10 μl of 5 mM dNTPs]{reagent1[2]}, [1 μl of 100 μM mDNA-F2 primer]{reagent1[3]}, [1 μl of 100 μM mDNA-R3 primer]{reagent1[4]}, [2 μl of Taq DNA polymerase]{reagent1[5]}, [5 μl of the 1st PCR product from step 71]{reagent1[6]}, and [159 μl of RNase- free water]{reagent1[7]}in a [200-μl PCR tubes]{container}.', '[Mix]{operation} well by [pipetting up and down]{method} [at least 10 times]{repeat}.'], 'notice': ['MIX sometimes fully mixes the reagents already added in the previous sections, and sometimes new reagents are mixed in.', 'Most MIX instructions are relatively simple.'], 'synonym': []}}]
    # query = str(query)
    # rate_basis = '''If the DSL uses graphical elements, each type of model element has a distinct representation that is easily identifiable.'''
    # text, scores = myqa(query, rate_basis)
    #print(text)
    #print(scores)
    #myqa_v2(query)