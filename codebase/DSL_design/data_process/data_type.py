import json
import random
import time
import os
import openai
import matplotlib.pyplot as plt

from openai import OpenAI
from tqdm import tqdm

from utils.token_count_decorator import token_count_decorator
from utils.util import match_and_remove_first_occurrence, check_annotated_format, load_json

class DataType:
    def __init__(self, data_type_path) -> None:
        '''
            DataType class for processing and analyzing data types and generating examples.

            @Arguments:
                
                data_type_path [str]: path to the data type JSON file;
                
            @Attributes:
            
                data_type_path [str]: path to the data type JSON file;
                
                data_type_extraction_path [str]: path to the data type extraction JSON file;
                
                quantities_extraction_prompt [str]: prompt for extracting quantities;
                
                scientific_example_generator [str]: prompt for generating scientific examples;
                
                type [list]: list of data types;
                
                examples [list]: list of example data.
                    
            @Public Methods:
            
                recalc_datatype(origin_data, recalc_datatype_from_zero, dataset): 
                    Recalculates data types from the given data and updates the data type extraction file.
                        
                data_annotated_check(annotated_data_type): 
                    Checks if annotated data types match the current data types.
                      
                get_example(new_examples): 
                    Retrieves examples of data types.
                        
                get_type(): 
                    Retrieves a description of data types.
        '''

        self.data_type_path = "data/" + data_type_path + ".json"
        self.data_type_extraction_path = "data/" + data_type_path + "_extraction.json"
        content = load_json(self.data_type_path, create=True, content={"datatype": [], "example": []})
        self.type = content["datatype"]
        self.examples = content["example"]

        with open("data/scientific_quantities_extraction.txt", 'r') as file:
            self.quantities_extraction_prompt = file.read()
        with open("data/scientific_example_generator.txt", 'r') as file:
            self.scientific_example_generator = file.read()

    def recalc_datatype(self, origin_data, reclac_datatype_from_zero, dataset):
        '''
            Recalculate datatype annotations based on extracted quantities.

            @Arguments:

                self: Instance of the class containing this method.
                
                origin_data (list): List of original data samples containing annotations.
                
                reclac_datatype_from_zero (bool): Flag indicating whether to recalculate datatype annotations from scratch.
                
                dataset (str): Name of the dataset being processed.

            @Returns:

                None. Updates the datatype extraction and example generation files.

            @Functionality:

                This function recalculates datatype annotations based on the extracted quantities from the given original data. It then generates new examples incorporating these annotations. If the flag 'reclac_datatype_from_zero' is set, it recalculates annotations from scratch; otherwise, it loads existing annotations. It generates new examples by querying a language model and checking the validity of the generated examples against predefined criteria. Finally, it updates the datatype extraction and example generation files with the new annotations and examples.
        '''

        # extract scientific datatype
        datatype = {
            "REG": "experimental material", 
            "Device": "laboratory equipment",
            "Container": "laboratory containers",
            "Time": "time",
            "String": "string",
            "Bool": "boolean"
        }
        example = [
            "Remove the [MPSC]{REG} from the [splenic vein]{REG} and the [IVC graft]{REG} on the [IHIVC]{REG}",
            "Transfer the [solution]{REG} with {pasteur pipette}[Device] into the [vials]{Container}",
            "Centrifuge for [70 min]{Time}",
            "See PDF file [\"Explant Protocol Procedure\"]{String}",
            "The opposite is [true]{Bool}"
        ]
        sampled_data = random.sample(origin_data, 100)
        result = {}
        stored = {"original_protocol": [], "result": {}}
        if reclac_datatype_from_zero:
            for data in tqdm(sampled_data):
                stored["original_protocol"].append(data[:])
                prompt = self.quantities_extraction_prompt + data.replace("\n", " ")[:7000]
                lines = self.__chatgpt_function(prompt).split("\n")
                for line in lines:
                    parts = line.split(":")
                    if len(parts) == 2:
                        origin_text = parts[0].strip()
                        quantity = parts[1].strip()
                        index, data = match_and_remove_first_occurrence(data, origin_text)
                        if index == -1:
                            continue
                        if quantity in result:
                            result[quantity].append(origin_text)
                        else:
                            result[quantity] = [origin_text]
            stored["result"] = result
            with open(self.data_type_extraction_path, 'w') as json_file:
                json.dump(stored, json_file)
        else:
            stored = load_json(self.data_type_extraction_path)
            result = stored["result"]

        keys = [x for x in list(result.keys()) if x in ["Volume","Temperature","Length", \
                "Energy","Concentration", "Mass","Speed","Acceleration","Density","Frequency", \
                "Force", "Acidity", "Flow Rate", "Pressure", "Voltage"] and len(result[x]) >= 5]
        values = [len(result[key]) for key in keys]
        plt.figure(figsize=(12, 5), dpi=500)
        plt.bar(keys, values)
        plt.xlabel('quantity')
        plt.ylabel('number')
        plt.savefig("data/scientific_quantities_extraction_"+dataset+".png")
        for key in keys:
            datatype[key] = key
        while True:
            like = ""
            for i, key in enumerate(keys):
                like = like + "[" + result[key][0] + "]{" + key + "}" + (", " if i != len(keys) - 1 else "") 
            prompt = self.scientific_example_generator.replace("******", str(keys)).replace("------", like)
            # print(prompt)
            new_example = self.__chatgpt_function(prompt)
            format, origin, label = check_annotated_format(new_example)
            if format and all(item in datatype for item in label) and all(item1.lower() != item2.lower() for item1, item2 in zip(origin, label)):
                new_example = new_example.split("\n")
                for ne in new_example:
                    example.append(ne)
                datatype_extraction = {
                    "datatype": datatype,
                    "example": example
                }
                with open(self.data_type_path, 'w') as json_file:
                    json.dump(datatype_extraction, json_file)
                return 

    def data_annotated_check(self, annotated_data_type):
        return self.type == annotated_data_type
    
    def get_example(self, new_examples):
        example = ""
        examples = self.examples + new_examples
        for i, s in enumerate(examples):
            example = example + s
            if i != len(examples) - 1:
                example = example + "\n"
        return example
    
    def get_type(self):
        datatype = ""
        for i, t in enumerate(self.type):
            discription = self.type[t]
            datatype = datatype + ("" if i != len(self.type)-1 else "and ")
            datatype = datatype + discription + " " + "\"" + t + "\""
            datatype = datatype + (", " if i != len(self.type)-1 else "")
        return datatype
    
    @token_count_decorator
    def __chatgpt_function(self, content):
        while True:
            # time.sleep(8)
            try:
                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a natural language processing model designed for performing NLP tasks."},
                        {"role": "user", "content": content}
                    ],
                    model="gpt-3.5-turbo",
                )
                return chat_completion.choices[0].message.content
            except openai.APIError as error:
                print(error)