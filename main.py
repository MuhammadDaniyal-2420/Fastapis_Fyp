from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import FastEmbedEmbeddings
import os

# Load existing vector databases
def load_vector_databases():
    vector_stores = []
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    i = 0
    while True:
        db_path = f"myCroma_db/chroma_db_{i}"
        if os.path.exists(db_path):
            vs = Chroma(persist_directory=db_path, embedding_function=embed_model)
            vector_stores.append(vs)
            i += 1
        else:
            break

    return vector_stores, embed_model

# Initialize FastAPI app
app = FastAPI()

# Load or create vector databases
vector_stores, embed_model = load_vector_databases()

# Ensure there are vector stores before proceeding
if not vector_stores:
    raise ValueError("No vector stores loaded. Please check the DB paths and ensure they exist.")

# Combine all vector stores into one retriever
combined_retriever = Chroma(persist_directory='/combined_chroma_db', embedding_function=embed_model)

for i, vs in enumerate(vector_stores):
    document_files = [f'myCroma_db/chroma_db_{i}/doc_{j}.md' for j in range(len(vs))]
    for doc_file in document_files:
        if os.path.exists(doc_file):
            with open(doc_file, 'r') as f:
                doc_content = f.read()
                documents = [doc_content]
                combined_retriever.add_documents(documents, collection_name=f'combined_rag_chunks_{i}_{j}')

# Define retriever and prompt
retriever = combined_retriever.as_retriever(search_kwargs={'k': 3})
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.and provide answer only according to pakistan's laws

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Initialize the retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatGroq(api_key="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"),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Define request and response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        response = qa.invoke({"query": request.query})
        return QueryResponse(result=response['result'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
