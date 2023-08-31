#https://huggingface.co/
#https://www.sbert.net/
from sentence_transformers import SentenceTransformer, util

#Load the model
model = SentenceTransformer('intfloat/multilingual-e5-small')
print("Max Sequence Length:", model.max_seq_length)
#mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es
#intfloat/multilingual-e5-small
#symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli
#sentence-transformers/multi-qa-MiniLM-L6-cos-v1
text = ""
with open("conveniotext1.txt", encoding="utf8") as input:
    file_content = input.readlines()
    for line in file_content:
        if len(line) > 1 and line[-2] != ".":
            line = line[:-1]
        #if len(line) > 0:
        text = text + line

docs = text.split("\n")
docs = [i for i in docs if i]
doc_emb = model.encode(docs)

def semantic_search(text):
    #Encode query
    query_emb = model.encode(text)
    #Compute dot score between query and all document embeddings
    #scores = util.cos_sim(query_emb, doc_emb)[0]
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))
    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    for doc, score in doc_score_pairs[:5]:
        print(score, doc)

query = "¿De cuanto tiempo es la jornada de trabajo?"
semantic_search(query)











    
