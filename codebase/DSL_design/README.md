# AutoDSL

## Enviorment Setup

```
git clone git@github.com:AutoDSL/DSL_design.git
cd DSL_design/
export OPENAI_API_KEY="your OPENAI key"
conda create -n autodsl python==3.8
pip install -r requirements.txt
cd ..
gdown https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_craft_md-0.5.3.tar.gz
cd DSL_design
python preprocess.py
```
Final ./data directory structure is shown below:

```
DSL_design
├── algorithm
├── data
├── data_process
├── distribution
└── utils
```

The rough structure of the code is as follows:
The algorithm folder contains the core algorithm code.
The data folder contains our data.
The data_process folder contains our data structures.
The distribution folder contains auxiliary mathematical structures used in DPMM.

Please copy the content of dataset/original_protocol to codebase/DSL_design/data/NCBWJ

## Gibbs-DPMM for non-parametric pattern integration

Demo run:
```
python main.py --mode cluster --data protocol_exchange --demo --max_param 5 --regular 0.1 --alpha 0.1 --data_type _datatype_3 --annotated_data _annotated_3
```
Run the demo, and the console will display the clustered results of parameter types for the `FIT` opcode.

Complete run:
```
python main.py --mode cluster --data molecular_biology_and_genetics --max_param 5 --regular 0.1 --alpha 0.1
```

## E-M algorithm for language feature assignment

```
python main.py --mode analyse --data molecular_biology_and_genetics --iter_times 500 --boom_iter 35 --sample_num 10 --eps 0.48
```
The console can display intermediate outputs, and all results along with the convergence curve can be viewed in the `log_molecular_biology_and_genetics.txt` file and `feature_extraction_molecular_biology_and_genetics.png` in the `data` folder.

## Semantic Utility
```
python main.py --mode compiler
python main.py --mode semantic_utility
```
After running the compiler, the program will compile the data stored in `data/utility_demo.json`. The results will be saved in `semantic_translation_log.json` and `semantic_translation_log_1.json` in the `data` folder, which include our method and five baselines results. Upon executing semantic_utility, the results of various methods will be compared with the data expertly checked and saved in the number of errors obtained in `result_for_expert_checked.xlsx`. Result will be saved in the data/csv folder.

## Syntax Utility
```
python main.py --mode syntax_utility
```
After running syntax_utility, the program will analyze the language features of the data in `data/utility_demo.json`, and the results will be output to the console. Intermediate results will be saved in `data/syntax_log_3.json`.