# wikhow

- process-v2.py stats wikhow step, verb, noun, etc.

# pre-process

- classmap.py maps the protocol's subjectArea to the 5 domains we've developed as experts.
- split_procedure.py splits protocol's procedure into segments.
- split_domain.py split protocols by domain
- statistic.py protocols related properties statistics
- get_prefix.py Get instructions of type prefix

# empirical-evaluation

- rag-load-split-vectorstore.py Build local vector store
- rag-loadvectorstore-inference-v3.py rag+gpt4 evaluation system
- run.sh Evaluation script

# quantity-evaluation

- extract.py ontology/entity/relation extraction
- check_convergence.py verify convergence of extracted ontology quantities
- cal_metric-v2.py Quantitative evaluation DSL
- cal_metric-v2-biocoder Quantitative evaluation of biocoder
- statistic-v2.py statistic-step, verb, noun, etc.
- get_domain_ontology.py Separate extracted ontology by domain