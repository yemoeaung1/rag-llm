from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
import os 
from langchain_core.prompts import ChatPromptTemplate
import sys
from langchain_community.llms.ollama import Ollama


def load_documents():
    document_loader = PyPDFDirectoryLoader('data')
    return document_loader.load()

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings()
    return embeddings

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def fetch_db(chunks, embeddings):
    idx_name= 'faiss_idx'
    if os.path.exists(f'db/{idx_name}'):
        faiss_idx = FAISS.load_local(folder_path=idx_name, embeddings=embeddings)
        print("Index loaded from disk")
    else:
        faiss_idx = FAISS.from_documents(chunks, embeddings)
        faiss_idx.save_local(folder_path=idx_name)
        print("New index created and saved to disk")
    retriever = faiss_idx.as_retriever(search_kwargs={"k": 3})
    return retriever


def add_to_db(chunks: list[Document]):
    db = FAISS.from_documents(chunks, get_embedding_function())
    return db


def query_rag(query_text: str):
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embedding_function()
    db = add_to_db(chunks)
    PROMPT_TEMPLATE = """
    Anser the question based only on the following context:{context}

    ---
    Answer the question based on the above context: {question}
    """

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    print(response_text)

# documents = load_documents()
# print(split_documents(documents)[0])
query_rag(sys.argv[1])
