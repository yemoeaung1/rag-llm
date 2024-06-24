# PDF-based Q&A System using Retrieval-Augmented Generation (RAG)

This project allows you to extract text from a PDF and use it as a knowledge base for answering user questions. The system leverages the power of Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant responses based on the extracted text.

## Features
- Text Extraction from PDF: Automatically extracts text from PDF documents.
- Question Answering: Responds to user queries using the extracted text as a source.
- RAG Integration: Utilizes Retrieval-Augmented Generation for enhanced accuracy and relevance in answers.

## Installation
1. Clone repository:
``` 
git@github.com:yemoeaung1/rag-llm.git
```

2. Install dependencies:
Ensure you have Python 3.10 or later installed. Then, install the required packages:
```
pip install -r requirements.txt
```

## Usage 
1. Create a folder called `data` in the directory
2. Add pdf(s) of choice. This is the knowledge base the LLM will use to respond to the queries.
3. Run `main.py` and enter query of choice.

```
python main.py <query>
```

## Customization
There are many customization options available. This program uses HuggingFace embedding functions, FAISS vector store, and Mistral LLM as my purpose was to be fully local. There are many options available through LangChain including but not limited to: OpenAI, Google, Azure, etc.

You can replace and modify the embedding, vector storage, and LLM as you like. You can reference the documentation at Langchain to see the available options and use what you like.

***If using non local options, you should create an .env file to store and access your private API keys.***

LangChain Website: https://python.langchain.com/v0.2/docs/introduction/


