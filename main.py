import getpass
import os
os.environ['USER_AGENT'] = 'myagent'

from langchain_community.chat_models import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.", category=UserWarning)


def setup_components():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

    # model = ChatAnthropic(model="claude-sonnet-4-6")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store

def read_webpage(url):
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")
    print(docs[0].page_content[:500])
    return docs

def setup_text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")
    return all_splits

def add_docs_to_vector_store(vector_store, all_splits):
    document_ids = vector_store.add_documents(documents=all_splits)
    print(document_ids[:3])
    return document_ids

def main():
    print("Hello from rag-agent!")
    # Set up all 3 components: chat model, embeddings model, and vector store.
    vector_store = setup_components()
    # Load the webpage from the URL and extract the relevant content.
    docs = read_webpage("https://lilianweng.github.io/posts/2023-06-23-agent/")
    # Split the webpage content into smaller chunks to fit within the context window of the language model.
    all_splits = setup_text_splitter(docs)
    # Add the chunks to the vector store and get back their document IDs.
    document_ids = add_docs_to_vector_store(vector_store, all_splits)

if __name__ == "__main__":
    main()
