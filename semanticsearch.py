#https://huggingface.co/
#https://www.sbert.net/
from sentence_transformers import SentenceTransformer, util
import time
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np

#Load the model
#model = SentenceTransformer('intfloat/multilingual-e5-base')    #(1.011GB)
#model = SentenceTransformer('intfloat/multilingual-e5-small') #(over 400mbs)
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct',device="cuda") #(over 400mbs)
print(model)
#print("Max Sequence Length:", model.max_seq_length)
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())
#mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es
#intfloat/multilingual-e5-small
#symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli
#sentence-transformers/multi-qa-MiniLM-L6-cos-v1
docs = []
doc_emb = None

def get_detailed_instruct(query: str) -> str:
    task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {task_description}\nQuery: {query}'

def loadDoc(name=None):
    text = ""
    with open("conveniotext1.txt", encoding="utf8") as input:
        file_content = input.readlines()
        for line in file_content:
            if len(line) > 1 and line[-2] != ".":
                line = line[:-1]
            #if len(line) > 0:
            text = text + line

    docs = text.split("\n")
    docs = [i.replace('\r\n', '') for i in docs if len(i.strip())>0]    #clean
    #docs = ["passage: "+ i for i in docs if i]
    print("Number of paragraphs: ",len(docs))
    start = time.process_time()
    #doc_emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True) 
    doc_emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True,device="cuda")    
    end = time.process_time()
    print("Processing time:",end - start)
    return docs, doc_emb


def semantic_search(text):
    #Encode query
    start = time.process_time()
    #query_emb = model.encode(["query: " + text], normalize_embeddings=True, show_progress_bar=True)
    query_emb = model.encode([get_detailed_instruct(text)], normalize_embeddings=True, show_progress_bar=True, device="cuda")
    #Compute dot score between query and all document embeddings
    #scores = util.cos_sim(query_emb, doc_emb)[0]    
    #scores = util.dot_score(query_emb, doc_emb)[0]#.cpu().tolist()      
    hits = util.semantic_search(query_emb, doc_emb, top_k=5, score_function=util.cos_sim)
    end = time.process_time()
    print("Processing results time:", end - start)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print("(Score: {:.4f})".format(hit['score']), docs[hit['corpus_id']].lstrip("passage: "))

    #showResults(scores)
    showEmbeddings(doc_emb,query_emb)
    

def pca_reduction(embeddings):    
    print(embeddings.shape)
    pca_model = PCA(n_components = 2)
    pca_model.fit(embeddings)
    pca_embeddings_values = pca_model.transform(embeddings)
    print(pca_embeddings_values.shape)
    return pca_embeddings_values

def tSNE_reduction(embeddings):
    print(embeddings.shape)
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_embeddings_values = tsne_model.fit_transform(embeddings)
    print(tsne_embeddings_values.shape)
    return tsne_embeddings_values
    
def showEmbeddings(embeddings_array, query_embeddings):

    concatenated = np.concatenate((embeddings_array,query_embeddings), axis=0)
    #embeddings_values = pca_reduction(concatenated)
    embeddings_values = tSNE_reduction(concatenated)
    colors = ["paragraph" for i in docs]
    colors.append("query")
    
    names = ["paragraph_" + str(i) for i,item in enumerate(docs)]
    names.append("query")
    
    fig = px.scatter(
        x = embeddings_values[:,0], 
        y = embeddings_values[:,1],
        hover_name = names,
        title = 'Text embeddings', width = 800, height = 600,
        #color_discrete_sequence = plotly.colors.qualitative.Alphabet_r
        color = colors
    )

    #fig.update_layout(
    #    xaxis_title = 'first component', 
    #    yaxis_title = 'second component')
    fig.show()
'''
def showResults(scores):
    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))
    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    for doc, score in doc_score_pairs[:5]:
        print(score, doc, len(doc),"\n")
'''    
docs, doc_emb = loadDoc()
query = "¿De cuanto tiempo es la jornada de trabajo?"
print(query)
semantic_search(query)









    
