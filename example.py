import pandas as pd
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.nodes import TransformersReader
from haystack.pipelines import DocumentSearchPipeline, ExtractiveQAPipeline
from sentence_transformers import SentenceTransformer
from haystack.utils import print_documents
from haystack.utils import print_answers
import torch, os
from datetime import datetime


TRANSFORMER = "intfloat/multilingual-e5-small"
PREFIX_PASSAGE = "passage: "
PREFIX_QUERY = "query: "
DOC_DIR = "docs"
# separate process to create table with embeddings
model = SentenceTransformer(TRANSFORMER)
print("Max Sequence Length:", model.max_seq_length)
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())

document_store = InMemoryDocumentStore(embedding_dim=384)   #large_model=1024
#document_store = InMemoryDocumentStore()

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=TRANSFORMER,
    use_gpu=True,
    scale_score=True,
)

reader = FARMReader("MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac", use_gpu=True) 
#reader = FARMReader("timpal0l/mdeberta-v3-base-squad2", use_gpu=True)

def loadData():
    files_to_index = [DOC_DIR + "/" + f for f in os.listdir(DOC_DIR)]
    for f in files_to_index:
        name = os.path.basename(f)
        texts = ""
        with open(f, encoding="utf8") as input:
            file_content = input.readlines()
            for line in file_content:
                if line.strip():    #remove empty lines
                    texts = texts + line

        texts = texts.split("\n")
        #start = time.process_time()
        #add prefix to all elements in texts only when encoding
        embeddings = model.encode([PREFIX_PASSAGE + i for i in texts if i], normalize_embeddings=True, show_progress_bar=True)     
        #end = time.process_time()
        #print("Processing time:",end - start)
        table_with_embeddings = pd.DataFrame(
            {"content": texts, "embedding": embeddings.tolist()}
        )

        docs = []
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # select from table and build in memory search pipeline without re-computing embeddings
        for i, row in table_with_embeddings.iterrows():
            docs.append(Document(content=row["content"], embedding=row["embedding"], content_type="text", meta={'name': name, "timestamp":dt_string}))
        document_store.write_documents(docs)
        #print(docs[2:5])
        result = document_store.get_all_documents(return_embedding=False)
        #print(result[3:6])

def search(query):
    pipeline = DocumentSearchPipeline(retriever=retriever)
    prediction = pipeline.run(query=PREFIX_QUERY + query, params={"Retriever": {"top_k": 3}})
    print_documents(prediction, max_text_len=500, print_name=True, print_meta=True)
    return prediction

def ask(query):
    myDocs = search(query)    
    result = reader.predict(
        query=query,
        documents=myDocs["documents"],
        top_k=3
    )
    print_answers(result, details="all")   
    
loadData()
query ="¿como es mi horario de trabajo?"
ask(query)