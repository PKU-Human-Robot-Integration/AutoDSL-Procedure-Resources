import numpy as np
import nltk
import re
import math
import json
import spacy
import gensim
import random
import scispacy
import matplotlib.pyplot as plt
import pandas as pd

from nltk.corpus import wordnet
from functools import reduce
from tqdm import tqdm
from utils.util import get_list, load_json
from matplotlib.gridspec import GridSpec

class EM_feature:
    def __init__(self) -> None:
        self.feature_list = {
            "statement": {
                "imperative-model": {
                    "loop-statement": {
                        "for-loop": self.__for_loop_prob,
                        "while-loop": self.__while_loop_prob
                    },
                    "jump-statement": {
                        "break-statement": self.__break_statement_prob,
                        "continue-statement": self.__continue_statement_prob
                    },
                    "if-statement": {
                        "if-branch": self.__if_branch_prob,
                        "if-else-branch": self.__if_else_branch_prob
                    },
                    "memory-management": {
                        "allocate-statement": self.__allocate_statement_prob,
                        "deallocate-statement": self.__deallocate_statement_prob
                    },
                    "function-procedure": {
                        "function-procedure-call": self.__function_procedure_call_prob,
                        "function-procedure-declaration": self.__function_procedure_declaration
                    },
                    "arithmetic-expression": {
                        "add-arithmetic-operator": self.__add_arithmetic_operator_prob,
                        "minus-arithmetic-operator": self.__minus_arithmetic_operator_prob,
                        "multi-arithmetic-operator": self.__multi_arithmetic_operator_prob,
                        "devid-arithmetic-operator": self.__devid_arithmetic_operator_prob
                    },
                    "logical-expression": {
                        "and-arithmetic-operator": self.__and_arithmetic_operator_prob,
                        "or-arithmetic-operator": self.__or_arithmetic_operator_prob,
                        "not-arithmetic-operator": self.__not_arithmetic_operator_prob
                    },
                    "assignment-expression": self.__assignment_expression_prob
                },
                "runtime-error-handling": {
                    "raise-statement": self.__raise_statement_prob,
                    "resolve-statement": self.__resolve_statement_prob
                },
                "type-system":{
                    "data-type": {
                        "interger-type-declaration": self.__interger_type_declaration_prob,
                        "floatingpoint-type-declaration": self.__floatingpoint_type_declaration_prob,
                        "boolean-type-declaration": self.__boolean_type_declaration_prob,
                        "string-type-declaration": self.__string_type_declaration_prob,
                        "vector-type-declaration": self.__vector_type_declaration_prob,
                        "dict-type-declaration": self.__dict_type_declaration_prob,
                        "set-type-declaration": self.__set_type_declaration_prob
                    },
                    "domain-specified-type": {
                        "temporal-type-declaration": self.__temporal_type_declaration_prob,
                        "reg-type-declaration": self.__reg_type_declaration_prob,
                        "device-type-declaration": self.__device_type_declaration_prob,
                        "container-type-declaration": self.__container_type_declaration_prob,
                        "scientific-type-declaration": self.__scientific_type_declaration_prob
                    },
                    "class-type": self.__class_type_declaration_prob, 
                },
                "concurrent": {
                    "data-parallel": {
                        "parallel-for": self.__parallel_for_prob,
                        "parallel-map": self.__parallel_map_prob
                    },
                    "message-passing": {
                        "spawn-process": self.__spawn_process_prob,
                        "send-message": self.__send_message_prob,
                        "receive-message": self.__receive_message_prob
                    }
                },
                "react": {
                    "event-statement": self.__event_statement_prob,
                    "response-statement": self.__response_statement_prob
                }      
            }
        }
        self.phi_list = get_list(self.feature_list)
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.spacysci_ner_model = spacy.load("en_ner_craft_md")
        self.container_list = ["cup","beaker","flask","cylinder","crucible","pot","tube","funnel","burette","condenser","dish","pipette"]
        self.time_list =  ["second", "minute", "hour", "sec", "min", "day", "s", "m", "d", "week", "month", "year", "y", "decade", "century"]
        return 
    
    def extraction(self, dataset, origin_data, logger, iter_num, boom_iter, sample_num, eps):
        '''
            Extracts parameters using the Expectation-Maximization algorithm.

            @Arguments:
                self: The object instance.
                dataset: The dataset used for parameter extraction.
                origin_data: The original dataset.
                logger: The logger object for logging information.
                iter_num: The number of iterations for the EM algorithm.
                boom_iter: The iteration after which the algorithm filters out parameters below a certain threshold.
                sample_num: The number of samples used in each iteration.
                eps: The threshold below which parameters are filtered out.

            @Output:
                phi_list: List of parameters.
                result: Resultant parameter values.

            @Functionality:
                This function implements the Expectation-Maximization (EM) algorithm to extract parameters. 
                It iteratively updates parameters using the E-step and M-step. In the E-step, it calculates 
                the probability distribution for each data point given the current parameter estimates. 
                In the M-step, it updates the parameters using the calculated probabilities. After a certain 
                number of iterations (boom_iter), it filters out parameters below a specified threshold (eps).

        '''
        phi_list = self.phi_list
        result = np.full(len(phi_list), (math.sqrt(5)-1)/2)
        tot = sample_num*2
        traj = {x:[] for x in self.phi_list}
        for ti in range(iter_num):
            data = random.sample(origin_data, sample_num)

            # E-step:
            if ti > 0:
                p_list = [self.Q(x, phi_list=phi_list) for x in data]
                p_new = np.sum(np.array(p_list), axis=0) / sample_num
            else:
                p_new = np.full(len(phi_list), (math.sqrt(5)-1)/2)

            # M-step
            if ti < boom_iter:
                result = (result*tot + p_new*1.5*sample_num)/(tot+1.5*sample_num)
                tot = tot + sample_num
                for i, phi in enumerate(phi_list):
                    traj[phi].append(result[i])
            else:
                result = (result*tot + p_new*1.5*sample_num)/(tot+1.5*sample_num)
                tot = tot + sample_num
                new_phi_list = []
                new_result = []
                for i, phi in enumerate(phi_list):
                    if result[i] > eps:
                        traj[phi].append(result[i])
                        new_result.append(result[i])
                        new_phi_list.append(phi)
                phi_list = new_phi_list
                result = np.array(new_result)

            logger.info("# epoch " + str(ti))
            self.print(phi_list=phi_list, result=result, logger=logger)
        
        self.draw_traj(traj, iter_num, dataset)
        return phi_list, result

    def Q(self, x, phi_list):
        if phi_list is None:
            raise "phi_list must not be None"
        
        doc = self.spacysci_ner_model(x)
        protocol = [[]]
        for token in doc:
            if token.pos_ == "VERB":
                protocol.append([token])
            else:
                protocol[-1].append(token)
        protocol =protocol[1:]
        result = np.zeros(len(phi_list))

        for i, phi in enumerate(phi_list):
            if phi in self.phi_list:
                result[i] = self.calulate_q_function(phi, protocol, phi_list)
            else:
                raise "phi_list dose not match"
        return result
    
    def calulate_q_function(self, phi, protocol, phi_list):
        if isinstance(self.phi_list[phi], list):
            results = [self.calulate_q_function(p, protocol, phi_list) for p in self.phi_list[phi]]
            product = reduce(lambda x, y: x * y, [1-prob for prob in results])
            return 1-product
        else:
            if phi in phi_list:
                return self.phi_list[phi](protocol)
            else:
                return 0

    def print(self, phi_list, result, logger=None):
        for i, phi in enumerate(phi_list):
            if logger is not None:
                logger.info(phi + " " + str(result[i]))
            else:
                print(phi, result[i])

    def word2vec_similarity(self, a:str, b:str):
        if a in self.word2vec_model and b in self.word2vec_model:
            vector_a = np.array(self.word2vec_model[a])
            vector_b = np.array(self.word2vec_model[b])
            cosine_similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
            return cosine_similarity
        else:
            return 0
        
    def wordnet_similarity(self, a:str, b:str):
        synsets1 = wordnet.synsets(a)
        synsets2 = wordnet.synsets(b)
        max_similarity = 0
        for synset1 in synsets1:
            for synset2 in synsets2:
                similarity = synset1.wup_similarity(synset2)
                if similarity is not None:
                    max_similarity = max(max_similarity, similarity)
        return max_similarity
    
    def draw_traj(self, traj, iter_num, dataset):
        height = 1 + len(self.feature_list["statement"])
        width = 3 + max([len([self.feature_list["statement"][x][y] for y in self.feature_list["statement"][x] if isinstance(self.feature_list["statement"][x][y], dict)]) for x in self.feature_list["statement"]])
        fig = plt.figure(figsize=(6*width, 4*height), dpi=300)
        gs = GridSpec(height, width)

        self.__draw_subplot_draj(fig.add_subplot(gs[0,1:width+1]), self.feature_list["statement"], \
                                 traj, "", left=True, length=width, iter_num=iter_num, tit=dataset.replace("_", " "))
        for i, feature in enumerate(self.feature_list["statement"]):
            self.__draw_subplot_draj(fig.add_subplot(gs[i+1,1:3]), \
                                     self.feature_list["statement"][feature], \
                                     traj, feature, left=True, length=2, iter_num=iter_num)
            for j, sub_feature in enumerate([x for x in self.feature_list["statement"][feature] if isinstance(self.feature_list["statement"][feature][x], dict)]):
                self.__draw_subplot_draj(fig.add_subplot(gs[i+1,j+3]), \
                                         self.feature_list["statement"][feature][sub_feature], \
                                         traj, sub_feature, left=False, length=1, iter_num=iter_num) 

        plt.savefig("data/feature_extraction_" + dataset + ".png")
        return 
    
    def __draw_subplot_draj(self, ax, phi_list, traj, name, left, length, iter_num, tit=None):
        for phi in phi_list:
            ax.plot(range(1, len(traj[phi])+1), traj[phi], label=phi)
        ax.legend(loc='upper left')
        plt.xlim(1, iter_num)
        plt.xscale('log') 
        plt.ylim(-0.1, 1.1)
        if left == True:
            ax.text(-1/length, 0.5, name, transform=plt.gca().transAxes, fontsize=20)
        else:
            ax.set_title(name, fontsize=20)

        if tit is not None:
            ax.set_title(tit, fontsize=30)

    def __interger_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', token.lemma_) and not ('.' in token.lemma_ or 'e' in token.lemma_ or 'E' in token.lemma_):
                    result = 1
        return result

    def __floatingpoint_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', token.lemma_) and ('.' in token.lemma_ or 'e' in token.lemma_ or 'E' in token.lemma_):
                    result = 1
        return result

    def __boolean_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if token.lemma_.lower() == "true" or token.lemma_.lower() == "false":
                    result = 1
        return result

    def __string_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if token.lemma_ == "\"" or token.lemma_ == "\'":
                    result = 1
        return result

    def __vector_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                result = max(result, self.word2vec_similarity(token.lemma_, "vector"))
        return result

    def __dict_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                result = max(result, self.word2vec_similarity(token.lemma_, "dictionary"))
        return result
    
    def __set_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                result = max(result, self.word2vec_similarity(token.lemma_, "set"))
        return result

    def __temporal_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if token.lemma_.lower() in self.time_list:
                    result = 1
        return result
    
    def __reg_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            REG = []
            for token in line[1:]:
                if token.ent_type_ == "CHEBI":
                    REG.append(token.lemma_)
            if len(REG) > 0:
                result = 1
        return result
    
    def __device_type_declaration_prob(self, protocol):
        return 1
    
    def __container_type_declaration_prob(self, protocol):
        result = 0
        for line in protocol:
            Container = []
            for token in line[1:]:
                if token.lemma_.lower() in self.container_list:
                    Container.append(token.lemma_.lower())
            if len(Container) > 0:
                result = 1
        return result

    def __scientific_type_declaration_prob(self, protocol):
        return 1

    def __for_loop_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "repeat"))
        return result

    def __while_loop_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            S = self.word2vec_similarity(verb.lemma_, "repeat")
            for token in line[1:]:
                if token.pos_ == "NUM":
                    S = 0
            result = max(result, S)
        return result

    def __if_branch_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line:
                if token.lemma_.lower() == "if":
                    result = 1
        return result

    def __if_else_branch_prob(self, protocol):
        if_count = 0
        else_count = 0
        for line in protocol:
            for token in line:
                if token.lemma_.lower() == "if":
                    if_count = True
                if token.lemma_.lower() == "else":
                    else_count = True
        return int(if_count and else_count)

    def __function_procedure_call_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "call"))
        return result

    def __function_procedure_declaration(self, protocol):
        return self.__function_procedure_call_prob(protocol)

    def __break_statement_prob(self, protocol):
        return self.__while_loop_prob(protocol) * self.__if_branch_prob(protocol)

    def __continue_statement_prob(self, protocol):
        return 0

    def __allocate_statement_prob(self, protocol):
        result = 0
        for line in protocol:
            REG, Container = [], []
            for token in line[1:]:
                if token.ent_type_ == "CHEBI":
                    REG.append(token.lemma_)
                if token.lemma_.lower() in self.container_list:
                    Container.append(token.lemma_.lower())
            if len(REG) > 0 or len(Container) > 0:
                result = 1
        return result

    def __deallocate_statement_prob(self, protocol):
        return self.__allocate_statement_prob(protocol)

    def __data_parallelism_prob(self, protocol):
        result = 0
        for line1, line2 in zip(protocol, protocol[1:]):
            NOUN1 = [token.lemma_.lower() for token in line1 if token.ent_type_ == "CHEBI"]
            NOUN2 = [token.lemma_.lower() for token in line2 if token.ent_type_ == "CHEBI"]
            PRON2 = [token.lemma_ for token in line2 if token.pos_ == "PRON"]
            if len(PRON2) == 0 and len(NOUN1) > 0 and len(NOUN2) > 0:
                result = 1
                for x in NOUN1:
                    for y in NOUN2:
                        if x == y:
                            result = 0
        return result

    def __task_parallelism_prob(self, protocol):
        result = 0
        max_result = []
        for line1, line2 in zip(protocol, protocol[1:]):
            NOUN1 = [token.lemma_ for token in line1 if token.pos_ == "NOUN"]
            NOUN2 = [token.lemma_ for token in line2 if token.pos_ == "NOUN"]
            PRON2 = [token.lemma_ for token in line2 if token.pos_ == "PRON"]
            if len(PRON2) == 0 and len(NOUN1) > 0 and len(NOUN2) > 0:
                pair_result = []
                for noun2 in NOUN2:
                    pair_result.append(max([self.word2vec_similarity(noun1, noun2) for noun1 in NOUN1]))
                max_result.append(max(pair_result)) 
        if len(max_result) > 0:
            result = 1 - np.average(np.array(max_result))
        else:
            result = 0
        return result

    def __add_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "add"))
            for token in line:
                if token.lemma_ == "+":
                    result = 1
        return result

    def __minus_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "minus"))
            for token in line:
                if token.lemma_ == "-":
                    result = 1
        return result

    def __multi_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "multiply"))
            for token in line:
                if token.lemma_ == "*" or token.text == "times":
                    result = 1
        return result

    def __devid_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "divide"))
            for token in line:
                if token.lemma_ == "/":
                    result = 1
        return result

    def __and_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line:
                if token.lemma_.lower() == "and":
                    result = 1
        return result

    def __or_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line:
                if token.lemma_.lower() == "or":
                    result = 1
        return result

    def __not_arithmetic_operator_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line:
                if token.lemma_.lower() == "not":
                    result = 1
        return result

    def __assignment_expression_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "equal"))
            for token in line:
                if token.lemma_ == "=":
                    result = 1
        return result

    def __raise_statement_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                result = max(result, self.word2vec_similarity(token.lemma_, "error"))
        return result
    
    def __resolve_statement_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            S = self.word2vec_similarity(verb.lemma_, "resolve")
            for token in line[1:]:
                result = max(result, S * self.word2vec_similarity(token.lemma_, "error"))
        return result
    
    def __class_type_declaration_prob(self, protocol):
        result = 0
        REG = {}
        for line in protocol:
            verb = line[0].lemma_.lower()
            for token in line[1:]:
                if token.ent_type_ == "CHEBI":
                    if token.lemma_ in REG:
                        REG[token.lemma_].add(verb)
                    else:
                        REG[token.lemma_] = set()
                        REG[token.lemma_].add(verb)
        for reg in REG:
            verbs = REG[reg]
            if len(verbs) >= 4:
                result = 1
        return result 
    
    def __spawn_process_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if token.text == "persons":
                    result = 1
        return result 
    
    def __send_message_prob(self, protocol):
        result = 0
        for line in protocol:
            verb = line[0]
            result = max(result, self.word2vec_similarity(verb.lemma_, "say"))
        return result * self.__spawn_process_prob(protocol)
    
    def __receive_message_prob(self, protocol):
        return self.__send_message_prob(protocol)
    
    def __event_statement_prob(self, protocol):
        result = 0
        for line in protocol:
            for token in line[1:]:
                if token.lemma_.lower() == "when":
                    result = 1
        return result 
    
    def __response_statement_prob(self, protocol):
        return self.__event_statement_prob(protocol)

    def __parallel_for_prob(self, protocol):
        result = 0
        for line in protocol:
            num_count = 0
            for token in line[1:]:
                if token.pos_ == "NUM":
                    num_count = num_count + 1
            if num_count > 4:
                result = 1
        return result

    def __parallel_map_prob(self, protocol):
        result = 0
        max_result = []
        for line1, line2 in zip(protocol, protocol[1:]):
            CHEM1 = [token.lemma_ for token in line1 if token.ent_type_ == "CHEBI"]
            CHEM2 = [token.lemma_ for token in line2 if token.ent_type_ == "CHEBI"]
            NOUN1 = [token.lemma_ for token in line1 if token.pos_ == "NOUN"]
            NOUN2 = [token.lemma_ for token in line2 if token.pos_ == "NOUN"]
            PRON2 = [token.lemma_ for token in line2 if token.pos_ == "PRON"]
            if len(PRON2) == 0 and len(NOUN1) > 0 and len(NOUN2) > 0 and len(CHEM1) > 0 and sorted(CHEM1) == sorted(CHEM2):
                pair_result = []
                for noun2 in NOUN2:
                    pair_result.append(max([self.word2vec_similarity(noun1, noun2) for noun1 in NOUN1]))
                max_result.append(max(pair_result)) 
        if len(max_result) > 0:
            result = 1 - np.average(np.array(max_result))
        else:
            result = 0
        return result
