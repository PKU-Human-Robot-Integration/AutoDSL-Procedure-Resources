import json
import random
import os
import nltk

from nltk.stem import WordNetLemmatizer
from data_process.corpora_feature import Corpora_Feature
from data_process.data_type import DataType
from utils.util import load_json

class Protocol:
    def __init__(self, data_path, annotated_data_path, max_param, datatype:DataType):
        '''
            Protocol class for data processing and feature extraction.

            @Arguments:
                
                data_path [str]: Path to the raw data JSON file.
                
                annotated_data_path [str]: Path to the annotated data JSON file.
                
                max_param [int]: Maximum number of parameters.
                
                datatype [DataType]: Object specifying the data type.
            
            @Public Methods:
            
                load_data(): Loads data from JSON files, preprocesses it, and prepares it for feature extraction.
                
                feature_vector_extraction(opcode, sample_num=None): Extracts feature vectors from the data.
                
                dump_feature_data(): Dumps the annotated data to a JSON file.
        '''
        self.data_path = "data/" + data_path + ".json"
        self.annotated_data_path = "data/" + annotated_data_path + ".json"
        self.data_corpora = {}
        self.data_annotated = {}
        self.data_feature = {}
        self.opcode_num = 0
        self.corpora_num = {}
        self.stop_words = []
        self.max_param = max_param
        self.corpora_feature = Corpora_Feature(max_param, datatype)
        self.feature_dim = self.corpora_feature.feature_dim
        self.datatype = datatype
        self.lemmatizer = WordNetLemmatizer()
        self.load_data()

    def load_data(self):
        data_corpora = load_json(self.data_path)
        self.data_annotated = load_json(self.annotated_data_path, create=True)
        with open('data/stop_word.txt', 'r') as file:
            for line in file:
                self.stop_words.append(line.strip())

        for opcode in data_corpora:
            if opcode in self.stop_words:
                continue

            corporas = data_corpora[opcode]
            cleaned_corporas = []
            for corpora in corporas:
                words = nltk.word_tokenize(corpora)
                first_word = words[0].lower() 
                lemmatized_word = self.lemmatizer.lemmatize(first_word, pos='v')
                if lemmatized_word != first_word:
                    continue
                if ',' in corpora:
                    continue
                corpora = corpora.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
                if len(corpora) > 0:
                    cleaned_corporas.append(corpora)

            if len(cleaned_corporas) > 10:
                self.data_corpora[opcode] = cleaned_corporas
                self.corpora_num[opcode] = len(cleaned_corporas)
        self.data_corpora = dict(sorted(self.data_corpora.items(), key=lambda item: len(item[1]), reverse=True))
        self.corpora_num = dict(sorted(self.corpora_num.items(), key=lambda item: item[1], reverse=True))
        self.opcode_num = len(self.data_corpora)

        data_annotated_datatype = self.data_annotated
        if "%DataType%" in data_annotated_datatype:
            data_annotated_datatype = data_annotated_datatype.pop("%DataType%")
            if self.datatype.data_annotated_check(data_annotated_datatype):
                for opcode in self.data_annotated:
                    self.data_feature[opcode] = []
                    for annotated_corpora in self.data_annotated[opcode]:
                        feature = self.corpora_feature.annotated_feature_extraction(annotated_corpora)
                        self.data_feature[opcode].append(feature)
        return 
    
    def feature_vector_extraction(self, opcode, sample_num=None):
        '''
            Extracts feature vectors from opcode corpora.

            @Arguments:
                self: The instance of the class.
                opcode (str): The opcode for which feature vectors are to be extracted.
                sample_num (int, optional): Number of samples to be considered. If None, all samples will be considered.

            @Returns:
                None

            @Functionality:
                This method extracts feature vectors from opcode corpora. It shuffles the corpora, iterates through them, annotates each corpora, extracts features from the annotated corpora, and stores the annotated corpora and corresponding features in respective data structures.

                It first checks if the opcode already exists in the data_feature dictionary. If it does, it returns without performing any operation. If sample_num is not provided, it defaults to the length of the opcode corpora. It then initializes data_feature and data_annotated dictionaries for the given opcode. Then, it shuffles the corpora for randomness. It iterates through the shuffled corpora, annotates each corpora using the corpora_feature.data_annotate() method, extracts features from the annotated corpora using corpora_feature.annotated_feature_extraction() method, and stores the annotated corpora and features in data_annotated and data_feature dictionaries respectively.

                If the extracted feature is None for any corpora, it continues to the next corpora. Finally, it returns None.
        '''
        if opcode in self.data_feature:
            return 
        if sample_num is None:
            sample_num = len(self.data_corpora[opcode])
        self.data_feature[opcode] = []
        self.data_annotated[opcode] = []
        self.corpora_feature.examples = []
        random.shuffle(self.data_corpora[opcode])
        for corpora in self.data_corpora[opcode]:
            if len(self.data_annotated[opcode]) >= sample_num:
                break
            annotated_corpora = self.corpora_feature.data_annotate(corpora)
            feature = self.corpora_feature.annotated_feature_extraction(annotated_corpora)
            if feature is None:
                continue
            print(annotated_corpora)
            print(feature) 
            print()
            self.data_annotated[opcode].append(annotated_corpora)
            self.data_feature[opcode].append(feature)
        return 

    def dump_feature_data(self):
        '''
            Writes annotated data to a JSON file specified by annotated_data_path attribute.
        '''
        self.data_annotated["%DataType%"] = self.datatype.type
        with open(self.annotated_data_path, 'w') as json_file:
            json.dump(self.data_annotated, json_file)
        del self.data_annotated["%DataType%"]