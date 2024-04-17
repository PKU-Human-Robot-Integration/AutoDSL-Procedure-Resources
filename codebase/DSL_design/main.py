import argparse
import json
import random
import nltk
import sys
import os
import ast

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

from utils.util import seed_set, print_pattern, load_json, load_csv
from utils.logger import create_logger
from data_process.protocol import Protocol
from data_process.data_type import DataType
from algorithm.DPMM import DPMM
from algorithm.EM_feature import EM_feature
from algorithm.Utility import Utility_Analyse
from algorithm.Compiler1 import Compiler

name_mapping = {
    "Molecular Biology & Genetics": "molecular_biology_and_genetics",
    "Biomedical & Clinical Research": "biomedical_and_clinical_research",
    "Ecology & Environmental Biology": "ecology_and_environmental_environmental",
    "Bioengineering & Technology": "bioengineering_and_technology",
    "Bioinformatics & Computational Biology": "bioinformatics_and_computational_biology"
}

simple_name = {
    "molecular_biology_and_genetics" : "Genetics",
    "biomedical_and_clinical_research" : "Medical",
    "ecology_and_environmental_environmental" : "Ecology",
    "bioengineering_and_technology" : "BioEng",
    "bioinformatics_and_computational_biology" : "InfoBio"
}

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='cluster', help='cluster')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--reclac_datatype', action='store_true', default=False)
parser.add_argument('--reclac_datatype_from_zero', action='store_true', default=False)
parser.add_argument('--data', type=str, default="protocol_exchange")
parser.add_argument('--data_type', type=str, default="_datatype")
parser.add_argument('--prefix_data', type=str, default="_prefix")
parser.add_argument('--origin_data', type=str, default="_origin")
parser.add_argument('--annotated_data', type=str, default="_annotated")
parser.add_argument('--dsl_result', type=str, default="_dsl")
parser.add_argument('--demo', action='store_true', default=False) # cluster inside corpora
parser.add_argument('--max_param', type=int, default=10, help='max num for slot and emit')
parser.add_argument('--iter_times', type=int, default=10000)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--regular', type=float, default=0.1)
parser.add_argument('--boom_iter', type=int, default=10000)
parser.add_argument('--sample_num', type=int, default=5)
parser.add_argument('--eps', type=float, default=0.5)
args = parser.parse_args()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet') 

if __name__ == "__main__":
    seed_set(args.seed)

    origin_data = load_json("data/" + args.data + args.origin_data + ".json")
    datatype = DataType(args.data + args.data_type)
    if args.reclac_datatype == True:
        datatype.recalc_datatype(origin_data, args.reclac_datatype_from_zero, args.data)
        exit(0)

    if args.mode == "analyse":
        emalgorithm = EM_feature()
        logger = create_logger("data/log_" + args.data + ".txt")
        phi_list, Q = emalgorithm.extraction(args.data, origin_data, logger, iter_num=args.iter_times, boom_iter=args.boom_iter, sample_num=args.sample_num, eps=args.eps)
        logger.info(">>> result:")
        emalgorithm.print(phi_list, Q, logger=logger)
    if args.mode == "cluster":
        protocol = Protocol(args.data + args.prefix_data, args.data + args.annotated_data, args.max_param, datatype) 
        print(protocol.corpora_num)
        if args.demo:
            opcode = "FIT"
            protocol.feature_vector_extraction(opcode=opcode)
            protocol.dump_feature_data()
            result = DPMM.cluster(protocol.data_feature[opcode], feature_dim=protocol.feature_dim, iter_times=args.iter_times, alpha=args.alpha, regular=args.regular)
            print(result['K'])
            pattern = protocol.corpora_feature.analyse(protocol.data_annotated[opcode], protocol.data_feature[opcode], result['K'], result['label'])
            print_pattern(pattern=pattern)
        else:
            dsl_dict = {}
            dsl_result_path = "data/" + args.data + args.dsl_result + ".json"
            dsl_dict = load_json(dsl_result_path, create=True)
            curve = load_csv('data/' + args.data + '_curve.csv')
            curve.to_csv('data/' + args.data + '_curve.csv', index=True)
            
            tot_num = len(protocol.data_corpora)
            for i, opcode in enumerate(protocol.data_corpora):
                if opcode in dsl_dict:
                    continue
                sample_num = 21-int((i/(tot_num-1))*19)
                print(opcode, sample_num)
                protocol.feature_vector_extraction(opcode=opcode, sample_num=sample_num)
                protocol.dump_feature_data()
                result = DPMM.cluster(protocol.data_feature[opcode], feature_dim=protocol.feature_dim, iter_times=args.iter_times, alpha=args.alpha, regular=args.regular)
                if result['K'] == 0:
                    continue
                print(result['K'])
                pattern = protocol.corpora_feature.analyse(protocol.data_annotated[opcode], protocol.data_feature[opcode], result['K'], result['label'])
                print_pattern(pattern=pattern)
                dsl_dict[opcode] = pattern
                curve[opcode] = [float(num) for num in result["log_likelihood_list"].split()]
                curve.to_csv('data/' + args.data + '_curve.csv', index=True)
                with open(dsl_result_path, 'w') as json_file:
                    json.dump(dsl_dict, json_file)
    if args.mode == "syntax_utility":
        utility_analyse = Utility_Analyse()
        utility_analyse.syntax_utility()
    if args.mode == "compiler":
        utility_analyse = Utility_Analyse()
        compiler = Compiler()
        utility_analyse.semantic_translation_new(compiler)
    if args.mode == "semantic_utility":
        utility_analyse = Utility_Analyse()
        utility_analyse.semantic_utility("compile_result.tsv")