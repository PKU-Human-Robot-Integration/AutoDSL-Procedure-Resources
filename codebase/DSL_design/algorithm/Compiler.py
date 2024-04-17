import openai
import os
import ast
import gensim
import json
import re

from typing import List
from openai import OpenAI
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

from data_process.data_type import DataType
from data_process.statement import Statement
from data_process.corpora_feature import Corpora_Feature
from utils.token_count_decorator import token_count_decorator
from utils.util import load_json, check_annotated_format

class Compiler:
    def __init__(self) -> None:
        '''
            Compiler: A class for compiling protocols into executable statements and generating variations using different baselines.

            @Arguments:
                None

            @Public Methods:
                compile(subject, protocol, enum=0):
                    Compiles a protocol into executable statements and generates variations using different baselines.
                
                compile_1(subject, protocol, enum=0, old_protocol_log=None):
                    Compiles a protocol using baseline 4 and baseline 5.

                similarity_params(subject, opcode, entity_list):
                    Finds the parameters most similar to the opcode.
        '''

        self.name_mapping = {
            "Molecular Biology & Genetics": "molecular_biology_and_genetics",
            "Biomedical & Clinical Research": "biomedical_and_clinical_research",
            "Ecology & Environmental Biology": "ecology_and_environmental_environmental",
            "Bioengineering & Technology": "bioengineering_and_technology",
            "Bioinformatics & Computational Biology": "bioinformatics_and_computational_biology"
        }
        self.dsl = {name:load_json("data/"+subject+"_dsl.json") for name, subject in self.name_mapping.items()}
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.lemmatizer = WordNetLemmatizer()
        with open("data/operation_extraction.txt") as file:
            self.operation_extraction_prompt = file.read()
        with open("data/emit_extraction.txt") as file:
            self.emit_extraction_prompt = file.read()
        with open("data/semantic_baseline1.txt") as file:
            self.baseline1_prompt = file.read() 
        with open("data/semantic_baseline4_1.txt") as file:
            self.baseline4_1_prompt = file.read() 
        with open("data/semantic_baseline4.txt") as file:
            self.baseline4_2_prompt = file.read() 
        pass

    def compile(self, subject, protocol, enum=0):
        '''
            Compile Protocol to DSL Statements.

            @Arguments:

                self: the instance of the class.
                
                subject (str): The subject of the protocol.

                protocol (str): The protocol to be compiled.

                enum (int): An optional enum value.

            @Returns:

                result (dict): A dictionary containing compiled DSL statements and baselines.
                    Keys:
                        - "dsl": List of compiled DSL statements.
                        - "baseline1": List of statements compiled using baseline 1.
                        - "baseline2": List of statements compiled using baseline 2.
                        - "baseline3": List of statements compiled using baseline 3.
                
                log_info (dict): A dictionary containing logging information.
                    Keys:
                        - "subject": The subject of the protocol.
                        - "enum": An optional enum value.
                        - "protocol": The protocol being compiled.
                        - "compile": List containing information about the compilation process.

            @Functionality:

                This function compiles a given protocol into DSL statements and generates baseline versions of the statements.
                It iterates through each sentence in the protocol, processes it, and compiles it into DSL statements.
                For each sentence, it also generates baseline versions using three different prompts.
                Finally, it logs the compilation process and returns the compiled DSL statements along with the logging information.
        '''

        datatype = DataType(self.name_mapping[subject] + "_datatype")
        sentense_list = self.__convert_to_sentense_list(protocol)
        result = {
            "dsl": [],
            "baseline1": [],
            "baseline2": [],
            "baseline3": []
        }
        log_info = {"subject": subject, "enum": enum, "protocol": protocol, "compile": []}  

        for i, sentense in enumerate(sentense_list):
            print(i, sentense)
            log_sentense = {"i": i, "sentense": sentense}

            statement_dsl = Statement()
            operation = self.__operation_extraction(sentense)
            if "NONE" in operation:
                log_info["compile"].append(log_sentense)
                continue
            statement_dsl.opcode = self.__similarity_opcode(subject, operation)
            if "NONE" in statement_dsl.opcode:
                log_info["compile"].append(log_sentense)
                continue
            entity_list = self.__entity_extraction(subject, sentense, datatype)
            statement_dsl.slot = self.similarity_params(subject, statement_dsl.opcode, entity_list)
            statement_dsl.emit = self.__get_emits(sentense)
            result["dsl"].append(statement_dsl)
            
            baseline1_prompt = self.baseline1_prompt.replace("-*-*-*", str(list(datatype.type.keys()))).replace("*-*-*-", sentense + "\nEntity list as follows: " + str(entity_list))
            for _ in range(5):
                json_output = self.__chatgpt_function(baseline1_prompt)
                format, statement_baseline1 = self.__check_json_output(json_output, datatype)
                if format:
                    result["baseline1"].append(statement_baseline1)
                    break
            
            baseline2_prompt = self.baseline1_prompt.replace("-*-*-*", str(list(datatype.type.keys()))).replace("*-*-*-", sentense)
            for _ in range(5):
                json_output = self.__chatgpt_function(baseline2_prompt)
                format, statement_baseline2 = self.__check_json_output(json_output, datatype)
                if format:
                    result["baseline2"].append(statement_baseline2)
                    break

            baseline3_prompt = self.baseline1_prompt.replace("-*-*-*", str(list(datatype.type.keys()))).replace("*-*-*-", sentense)
            for _ in range(5):
                json_output = self.__chatgpt_function(baseline3_prompt, gpt_model="gpt-4")
                format, statement_baseline3 = self.__check_json_output(json_output, datatype)
                if format:
                    result["baseline3"].append(statement_baseline3)
                    break

            log_sentense["operation"] = operation
            log_sentense["entity_list"] = entity_list
            log_sentense["result_dsl"] = statement_dsl.dict()
            log_sentense["result_baseline1"] = statement_baseline1.dict()
            log_sentense["result_baseline2"] = statement_baseline2.dict()
            log_sentense["result_baseline3"] = statement_baseline3.dict()
            log_info["compile"].append(log_sentense)
        return result, log_info   
    

    def compile_1(self, subject, protocol, enum=0, old_protocol_log=None):
        datatype = DataType(self.name_mapping[subject] + "_datatype")
        sentense_list = self.__convert_to_sentense_list(protocol)
        result = {
            "baseline4": [],
            "baseline5": []
        }
        log_info = {"subject": subject, "enum": enum, "protocol": protocol, "compile": []}  

        for i, sentense in enumerate(sentense_list):
            print(i, sentense)
            log_sentense = {"i": i, "sentense": sentense}
            old_log = old_protocol_log[i]

            if "operation" not in old_log:
                log_info["compile"].append(log_sentense)
                continue
            
            reference = old_log["result_dsl"]["slot"].copy()
            reference.append(["output", old_log["result_dsl"]["emit"]])
            reference = {old_log["operation"]:reference}
        
            for _ in range(5):
                baseline4_1_prompt = self.baseline4_1_prompt.replace("*-*-*-", "Python").replace("*+*+*+", sentense)
                program_output = self.__chatgpt_function(baseline4_1_prompt)
                program = re.findall(r'```([^`]*)```', program_output, re.DOTALL)
                if len(program) > 0:
                    program = program[0]
                    log_sentense["python"] = program
                    break
            if isinstance(program, list):
                program = "No program"
                log_sentense["python"] = program
            for _ in range(5):
                baseline4_2_prompt = self.baseline4_2_prompt.replace("*+*+*+", "Python").replace("*-*-*-", sentense).replace("-*-*-*", str(list(datatype.type.keys()))).replace("+*+*+*", str(reference)).replace("+-+-+-", program)
                json_output = self.__chatgpt_function(baseline4_2_prompt)
                format, statement_baseline4 = self.__check_json_output(json_output, datatype)
                if format:
                    result["baseline4"].append(statement_baseline4)
                    break
            
            for _ in range(5):
                baseline5_1_prompt = self.baseline4_1_prompt.replace("*-*-*-", "Biocoder").replace("*+*+*+", sentense)
                program_output = self.__chatgpt_function(baseline5_1_prompt, gpt_model="gpt-4")
                program = re.findall(r'```([^`]*)```', program_output, re.DOTALL)
                if len(program) > 0:
                    program = program[0]
                    log_sentense["biocoder"] = program
                    break
            if isinstance(program, list):
                program = "No program"
                log_sentense["biocoder"] = program
            for _ in range(5):
                baseline5_2_prompt = self.baseline4_2_prompt.replace("*+*+*+", "Biocoder").replace("*-*-*-", sentense).replace("-*-*-*", str(list(datatype.type.keys()))).replace("+*+*+*", str(reference)).replace("+-+-+-", program)
                json_output = self.__chatgpt_function(baseline5_2_prompt)
                format, statement_baseline5 = self.__check_json_output(json_output, datatype)
                if format:
                    result["baseline5"].append(statement_baseline5)
                    break

            log_sentense["result_baseline4"] = statement_baseline4.dict()
            log_sentense["result_baseline5"] = statement_baseline5.dict()
            log_info["compile"].append(log_sentense)
        return result, log_info   

    
    def __convert_to_sentense_list(self, protocol):
        sentences = sent_tokenize(protocol)
        return sentences
    
    def __operation_extraction(self, sentense):
        prompt = self.operation_extraction_prompt.replace("------", sentense)
        for _ in range(5):
            result = self.__chatgpt_function(prompt)
            if "NONE" in result.upper():
                return "NONE"
            words = word_tokenize(result)
            tagged_words = pos_tag(words)
            if len(tagged_words) == 1:
                return tagged_words[0][0].upper()
        return "NONE"
            
    def __similarity_opcode(self, subject, operation):
        closest_word = None
        max_similarity = -1
        if operation not in self.word2vec_model:
            return "NONE"
        for target_word in self.dsl[subject]:
            if target_word in self.word2vec_model:
                similarity = self.word2vec_model.similarity(operation.lower(), self.lemmatizer.lemmatize(target_word.lower(), pos='v'))
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_word = target_word
        return closest_word
    
    def __entity_extraction(self, subject, sentense, datatype):
        corpora_feature = Corpora_Feature(100, datatype)
        annotated_corpora = corpora_feature.data_annotate(sentense)
        _, origin, label = check_annotated_format(annotated_corpora)
        result = []
        for ori, lab in zip(origin, label):
            if lab in datatype.type:
                result.append([lab, ori])
        return result
    
    def similarity_params(self, subject, opcode, entity_list):
        dsl = self.dsl[subject]
        pattern = [x["pattern"] for x in dsl[opcode]]
        max_similarity = -1
        max_pattern = []
        matched_pattern = []
        matched_result = []
        for p in pattern:
            sim, lcs, result = self.__lcs_similarity(p, entity_list)
            if sim > max_similarity or (sim == max_similarity and len(p) > len(max_pattern)):
                max_similarity = sim
                max_pattern = p
                matched_pattern = lcs
                matched_result = result

        if len(matched_pattern) == len(entity_list):
            return matched_result
        else:
            return entity_list
        
    def __get_emits(self, sentense):
        prompt = self.emit_extraction_prompt.replace("+-+-+-", sentense)
        for _ in range(5):
            result = self.__chatgpt_function(prompt)
            words = word_tokenize(result)
            if len(words) < 50:
                return result
        return ""
            
    def __check_json_output(self, json_text, datatype):
        try:
            dict = ast.literal_eval(json_text)
            if len(dict) != 1:
                return False, Statement()
            result = Statement()
            result.opcode = list(dict.keys())[0]
            params = dict[result.opcode]
            for param in params:
                if len(param) != 2 or (param[0] != "output" and param[0] not in datatype.type):
                    return False, Statement()
                if param[0] == "output" and isinstance(param[1], str):
                    result.emit = param[1]
                else:
                    result.slot.append(param)
            return True, result
        except:
            return False, Statement()
    
    @token_count_decorator
    def __chatgpt_function(self, content, gpt_model="gpt-3.5-turbo"):
        while True:
            # time.sleep(3)
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    model=gpt_model,
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as error:
                print(error)

    def __lcs_similarity(self, list1, list2):
        '''
            Longest Common Subsequence (LCS) Similarity Calculation.

            @Arguments:
                list1 (list): First list of elements.
                list2 (list): Second list of elements.

            @Returns:
                (tuple): A tuple containing:
                    - Length of the Longest Common Subsequence (LCS).
                    - The LCS itself.
                    - Result list indicating matches and mismatches.

            @Functionality:
                This function calculates the Longest Common Subsequence (LCS) similarity between two lists.
                It finds the length of the LCS and extracts the LCS elements along with additional information 
                indicating mismatches. 

            @Implementation:
                - Initialize a dynamic programming (DP) table to store the length of LCS for each subproblem.
                - Iterate through the lists and populate the DP table.
                - Track the LCS elements and additional mismatches using traceback.
                - Reverse the LCS and result lists before returning.
        '''
        m = len(list1)
        n = len(list2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if list1[i - 1] == list2[j - 1][0]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = []
        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if list1[i - 1] == list2[j - 1][0]:
                lcs.append(list2[j - 1])
                result.append(list2[j - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                result.append([list1[i-1], None])
                i -= 1
            else:
                j -= 1
        while i > 0:
            result.append([list1[i-1], None])
            i -= 1
        lcs.reverse()
        result.reverse()
        return len(lcs) , lcs, result