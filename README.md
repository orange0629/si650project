# SI650 Final Project (Lechen Zhang)

## How to run the code:

### Required packages:
1. Pyterrier (pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier)
2. OpenNIR (pip install --upgrade git+https://github.com/Georgetown-IR-Lab/OpenNIR)
3. Sentence Transformers (pip install -U sentence-transformers)
4. Fastrank (pip install fastrank)
5. Natural Language Toolkit (Optional because the results were cached in dataset_sentiment.json)

### Required files:
1. Basic files: dataset_full.csv, qrels.csv, query.csv
2. Cache: dataset_sentiment.json, trained_bert55.tar.gz

### Running instruction

1. If you want to run the whole project, please go to Project_code.ipynb and run all blocks. It will redraw all plots and recalculate all evaluation metrics. The last block of this notebook is interactive, which means you can input your queries there and get retrieval results.
2. If you only want to run the interactive part, please go to demo.py and run it. But please remember that we didn't find a good way to store pipelines, so it may take 30 seconds on GPU or 5-10 minutes on CPU for the training process. After the training is over, you can interactive with it, input your queries there and get retrieval results.