#https://huggingface.co/
#https://www.sbert.net/
from sentence_transformers import SentenceTransformer, util
import time
import torch
#Load the model
model = SentenceTransformer('intfloat/multilingual-e5-base')    #(1.011GB)
#model = SentenceTransformer('intfloat/multilingual-e5-small') #(over 400mbs)

print("Max Sequence Length:", model.max_seq_length)
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())
#mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es
#intfloat/multilingual-e5-small
#symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli
#sentence-transformers/multi-qa-MiniLM-L6-cos-v1
docs = []
doc_emb = None

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
    docs = ["passage: "+ i for i in docs if i]
    start = time.process_time()
    doc_emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)     
    end = time.process_time()
    print("Processing time:",end - start)
    return docs, doc_emb


def semantic_search(text):
    #Encode query
    start = time.process_time()
    query_emb = model.encode("query: " + text, normalize_embeddings=True)
    #Compute dot score between query and all document embeddings
    #scores = util.cos_sim(query_emb, doc_emb)[0]    
    #scores = util.dot_score(query_emb, doc_emb)[0]#.cpu().tolist()      
    hits = util.semantic_search(query_emb, doc_emb, top_k=5, score_function=util.cos_sim)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print("(Score: {:.4f})".format(hit['score']), docs[hit['corpus_id']].lstrip("passage: "))

    end = time.process_time()
    print(end - start)
    #showResults(scores)        
    
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











    
