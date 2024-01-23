import os, math
os.environ["JDK_HOME"] = "E:\Program Files (x86)\Java\jdk-19"
os.environ["JAVA_HOME"] = "E:\Program Files (x86)\Java\jdk-19"
index_path = 'C:\\Users\\zhang\\Downloads\\si650_proj_leczhang_ourox\\proj_file'


import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd
import onir_pt
import numpy as np
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

dataset = pd.read_csv('./dataset_full.csv')
query = pd.read_csv('./query.csv')
qrels = pd.read_csv('qrels.csv')


combined_list = []
for i in range(len(dataset)):
    combined_list.append(dataset['Description'][i] + dataset['pos_review_content'][i])
dataset['combined'] = combined_list
dataset['revised_title']=dataset['pos_review_title'].apply(lambda row: row[0:int(len(row)/2)])
dataset = dataset.fillna('None')



if not os.path.exists(index_path + "/data.properties"):
    indexer = pt.DFIndexer(index_path, overwrite=True, blocks=True)
    index_ref = indexer.index(dataset['combined'], dataset['docno'])
else:
    index_ref = pt.IndexRef.of(index_path + "/data.properties")
index = pt.IndexFactory.of(index_ref)


bm25 = pt.BatchRetrieve(index,wmodel="BM25")
qe = pt.rewrite.Bo1QueryExpansion(index)

import json
SENTIMENT_CACHE_NAME = 'dataset_sentiment.json'
if not os.path.exists(SENTIMENT_CACHE_NAME):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download([
            "names",
            "stopwords",
            "averaged_perceptron_tagger",
            "vader_lexicon",
            "punkt",
    ])
    sia = SentimentIntensityAnalyzer()
    senti = dataset['neg_review_content'].apply(lambda x: sia.polarity_scores(x)['neg']).tolist()
    fw = open(SENTIMENT_CACHE_NAME,"w")
    fw.write(json.dumps(senti))
    fw.close() 
else:
    cache_file = open(SENTIMENT_CACHE_NAME, 'r')
    cache_content = cache_file.read()
    senti = json.loads(cache_content)
    cache_file.close()


SEED=13
from sklearn.model_selection import train_test_split
tr_va_data,test_data=train_test_split(query,test_size=8,random_state=SEED)
train_data,valid_data=train_test_split(tr_va_data,test_size=8,random_state=SEED)

vbert=onir_pt.reranker.from_checkpoint('trained_bert55.tar.gz', text_field='revised_title')

ltr_feats_all = ((bm25>>qe>>bm25)%50) >>   pt.apply.generic(lambda df: df.merge(dataset,on='docno'))>> (
        pt.transformer.IdentityTransformer()
        ** # new feature: TF-IDF indicator
        pt.BatchRetrieve(index, wmodel="TF_IDF")
        ** #new feature: review number
        (pt.apply.doc_score(lambda row: math.log(row['review_num'])))
        ** #new feature: overall rate
        (pt.apply.doc_score(lambda row: row['overall_rate']))
        **# new feature: positive review title length
        (pt.apply.doc_score(lambda row: len(row['pos_review_title']) ))
        ** # new feature: state in query
        (pt.apply.doc_score(lambda row: int( row["query_0"] is not None and (row ['State'].lower()in row["query_0"].lower()) )))
        ** # negative relevance
        (pt.text.scorer(body_attr="neg_review_content", wmodel='BM25') >> (pt.apply.doc_score(lambda row: row['score']* senti[int(row['docid'])])))
        **
        (pt.apply.doc_score(lambda row: float(util.pytorch_cos_sim(model.encode(row["query_0"], convert_to_tensor=True), model.encode(row["Description"], convert_to_tensor=True)))))
        **
        (vbert)
    )

import fastrank
qrels['label'] = qrels['label'].apply(lambda x: float(x))
train_request = fastrank.TrainRequest.coordinate_ascent()
params = train_request.params
params.init_random = True
params.normalize = True
params.seed = 1234567
ca_pipe_all = ltr_feats_all >> pt.ltr.apply_learned_model(train_request, form='fastrank')
ca_pipe_all.fit(train_data, qrels)


x = input("Input a query, or 'exit' to leave: ")
while(x != 'exit'):
    df = ca_pipe_all.search(x)
    for idx in df.index:
        result_id = int(df['docid'][idx])
        print('\n')
        print('\n')
        print('Rank: '+str(df['rank'][idx]))
        print('Document '+ str(result_id) + ': '+ dataset['Name'][result_id], dataset['State'][result_id])
        print("Rate: "+ str(dataset['overall_rate'][result_id]), "Review Number: "+ str(dataset['review_num'][result_id]))
        print(dataset['url'][result_id])
        print(dataset['Description'][result_id])
    
    x = input("Input a new query, or 'exit' to leave: ")
