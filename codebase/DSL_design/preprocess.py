import json
import os

if __name__ == "__main__":
    '''
        Main function for processing JSON files and classifying data.

        @Input:
            None

        @Output:
            JSON files with classified data

        This function iterates through JSON files in a specified folder, extracts relevant data, and classifies it into predefined categories based on the "bigAreas" field in each JSON file. It then writes the classified data into separate JSON files corresponding to each category.

        Implementation:
        1. Define a folder path containing JSON files and a dictionary for classification.
        2. Define a dictionary mapping categories to their corresponding file paths.
        3. Iterate through JSON files in the folder.
        4. Extract relevant data from each JSON file and classify it based on the "bigAreas" field.
        5. Write the classified data into separate JSON files for each category.
    '''

    
    folder_path = "data/NCBWJ/"
    classification = {
        "Molecular Biology & Genetics": [],
        "Biomedical & Clinical Research": [],
        "Ecology & Environmental Biology": [],
        "Bioengineering & Technology": [], 
        "Bioinformatics & Computational Biology": []
    }
    file_paths = {
        "Molecular Biology & Genetics": "data/molecular_biology_and_genetics_origin.json",
        "Biomedical & Clinical Research": "data/biomedical_and_clinical_research_origin.json",
        "Ecology & Environmental Biology": "data/ecology_and_environmental_environmental_origin.json",
        "Bioengineering & Technology": "data/bioengineering_and_technology_origin.json",
        "Bioinformatics & Computational Biology": "data/bioinformatics_and_computational_biology_origin.json"
    }
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                areas = data["bigAreas"]
                for area in areas:
                    if area in classification:
                        classification[area].append(''.join(data["procedures"]))
                    else:
                        print("Error!")
                        raise "Area does not match!"
    for category, file_name in file_paths.items():
        with open(file_name, 'w') as json_file:
            json.dump(classification[category], json_file)