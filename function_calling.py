from sentence_transformers import SentenceTransformer, util
import json 

#Load the model
model = SentenceTransformer('intfloat/multilingual-e5-small')
print("Max Sequence Length:", model.max_seq_length)

passage = []
doc_emb = None

def loadJson(fileName= None):
    if fileName is None:
        fileName = 'actions.json'
    with open(fileName, encoding='utf-8') as f:
        data = json.load(f)
    
    passage = []
    for element in data:
        passage.append("passage: " + element["description"])  
    doc_emb = model.encode(passage, normalize_embeddings=True)
    return doc_emb, passage
    
def semantic_search(text):
    #Encode query
    print(text)
    query_emb = model.encode("query: " + text, normalize_embeddings=True)
    #Compute dot score between query and all document embeddings
    #scores = util.cos_sim(query_emb, doc_emb)[0]
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    #Combine docs & scores
    doc_score_pairs = list(zip(passage, scores))
    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    for doc, score in doc_score_pairs[:5]:
        print(score, doc)


doc_emb, passage = loadJson()
query = "¿Cuantas horas sindicales tengo?"
semantic_search(query)