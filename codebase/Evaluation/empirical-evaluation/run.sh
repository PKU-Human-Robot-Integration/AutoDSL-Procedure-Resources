
nohup python -u rag-loadvectorstore-inference-v3.py --isa ../DSL/bioinfo/bioinformatics_and_computational_biology_dsl.json --isa_type json --fw outputs2/result-InfoBio.csv > outputs2/infer-infobio.log 2>&1 &

nohup python -u rag-loadvectorstore-inference-v3.py --isa ../DSL/bioeng/bioengineering_and_technology_dsl.json --isa_type json --fw outputs2/result-BioEng.csv > outputs2/infer-bioeng.log 2>&1 &

nohup python -u rag-loadvectorstore-inference-v3.py --isa ../DSL/biomed/biomedical_and_clinical_research_dsl.json --isa_type json --fw outputs2/result-Medical.csv > outputs2/infer-medical.log 2>&1 &

nohup python -u rag-loadvectorstore-inference-v3.py --isa ../DSL/ecolo/ecology_and_environmental_environmental_dsl.json --isa_type json --fw outputs2/result-Ecology.csv > outputs2/infer-ecology.log 2>&1 &

nohup python -u rag-loadvectorstore-inference-v3.py --isa ../DSL/mole/molecular_biology_and_genetics_dsl.json --isa_type json --fw outputs2/result-Genetics.csv > outputs2/infer-genetics.log 2>&1 &

nohup python -u rag-loadvectorstore-inference-v3.py --isa biocoder_function.txt --isa_type txt --fw outputs2/result-Biocoder.csv > outputs2/infer-biocoder.log 2>&1 &