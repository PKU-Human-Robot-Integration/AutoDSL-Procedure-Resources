import os
import re
import time
import openai
import numpy as np

from openai import OpenAI
from huggingface_hub import InferenceClient
from utils.util import match_and_remove_first_occurrence, check_annotated_format
from utils.token_count_decorator import token_count_decorator
from data_process.data_type import DataType

class Corpora_Feature:
    def __init__(self, max_param, datatype:DataType):
        '''
            Corpora_Feature: A class for managing corpora features extraction and analysis.

            @Arguments:
                
                max_param [int]: Maximum number of parameters per example.
                
                datatype [DataType]: An instance of the DataType class specifying the data type.
                
            @Public Methods:

                data_annotate(corpora, model="G"): 
                    Annotates the provided corpora data using either GPT model or a custom model.
                
                annotated_feature_extraction(annotated_corpora): 
                    Extracts features from annotated corpora data.
                
                analyse(data_corpora, data, K, label): 
                    Analyzes the corpora data to cluster similar examples based on features and labels.
        '''
        self.max_param = max_param
        self.feature_dim = max_param * len(datatype.type)
        self.examples = []
        self.datatype = datatype
        
        with open("data/entity_extraction.txt", 'r') as file:
            self.entity_extraction_prompt = file.read()
        return 
    
    def data_annotate(self, corpora, model="G"):
        '''
            Annotates data for a given corpus using a specified model.

            @Arguments:

                corpora [list]: List of strings representing corpora to be annotated.
                
                model (str): Optional. Specifies the model to be used for annotation. Defaults to "G" for ChatGPT.
                
            @Returns:
            
                annotated_corpora [str or list]: Annotated corpora with entities marked. If the annotation meets specific format requirements, it returns the annotated corpora. Otherwise, it returns the original corpora.
        '''
        stored_corpora = corpora[:]
        for _ in range(4):
            corpora = stored_corpora[:]
            example = self.datatype.get_example(self.examples)
            type = self.datatype.get_type()
            prompt = self.entity_extraction_prompt.replace("&&&&&&", example).replace("%%%%%%", type) + corpora
            if model == "G":
                annotated_corpora = self.__chatgpt_function(prompt)
            format, origin, label = check_annotated_format(annotated_corpora)
            if format and sum([1 for x in label if x in self.datatype.type]) == len(label):
                return annotated_corpora
            else:
                print("format worng: "+annotated_corpora)
        return stored_corpora
    
    def annotated_feature_extraction(self, annotated_corpora):
        '''
            Annotated feature extraction.

            @Arguments:

                annotated_corpora (str): Annotated corpora containing labels enclosed in curly braces.

            @Output:

                feature (numpy.ndarray or None): Extracted feature vector representing the presence of labels in the corpora. None if no labels found.
        '''
        label = re.findall(r'\{([^}]*)\}', annotated_corpora)
        feature = np.zeros(self.feature_dim)
        if len(label) == 0:
            return None
        if len(label) > self.max_param:
            label = label[:self.max_param]
        for i, lab in enumerate(label):
            for j, t in enumerate(self.datatype.type):
                if lab == t:
                    feature[len(self.datatype.type) * i + j] = 1
        self.examples = [annotated_corpora] + self.examples
        self.examples = self.examples[:2]
        return feature

    def analyse(self, data_corpora, data, K, label):
        '''
            Analyze data and create clusters based on specified labels.

            @Arguments:
                
                data_corpora (list): A list of data representing the corpora.

                data (list): A list of data to be analyzed.

                K (int): Number of clusters to be created.

                label (list): A list containing labels corresponding to each data point.

            @Output:

                result (list): A list of dictionaries, where each dictionary represents a cluster. Each cluster contains:
                
                    - "pattern": A list of patterns representing the common features among data points in the cluster.
                    
                    - "example": A list of examples or protocols belonging to the cluster.
                    
                    - "example_feature": A list of extractions representing the features of each example in the cluster.

            This function implements a clustering algorithm to group data points based on their labels. It identifies common features within each cluster and organizes the data accordingly. The process involves iterating through each cluster, determining anchor features, extracting patterns, and creating clusters with relevant information.
        '''
        N, result = len(data), []
        for k in range(K):
            features = [data[x] for x in range(N) if label[x] == k]
            protocols = [data_corpora[x] for x in range(N) if label[x] == k]
            anchor_feature = None
            min_param = self.max_param + 1
            for feature in features:
                param_num = int(np.sum(feature))
                if param_num < min_param:
                    min_param = param_num
                    anchor_feature = feature
            
            pattern = []
            for i in range(min_param):
                for j, t in enumerate(self.datatype.type):
                    if anchor_feature[len(self.datatype.type) * i + j] == 1:
                        pattern.append(t)
            
            extractions = []
            for feature in features:
                ext = []
                for i in range(int(np.sum(feature))):
                    for j, t in enumerate(self.datatype.type):
                        if feature[len(self.datatype.type) * i + j] == 1:
                            ext.append(t)
                extractions.append(ext)

            
            cluster = {
                "pattern": pattern,
                "example": protocols,
                "example_feature": extractions
            }
            result.append(cluster)

        return result
    
    @token_count_decorator
    def __chatgpt_function(self, content):
        while True:
            # time.sleep(3)
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a natural language processing model designed for performing basic NLP tasks."},
                        {"role": "user", "content": content}
                    ],
                    model="gpt-3.5-turbo",
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as error:
                print(error)