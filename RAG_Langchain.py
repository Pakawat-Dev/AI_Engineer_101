from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# 1. Load documents
loader = TextLoader("data.txt",encoding="utf-8")
documents = loader.load()
# print(documents)

# 2. Split data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 3. Convert data to vectors
embedding = OpenAIEmbeddings()

# 4. Store data in vector store
vectorstore = FAISS.from_documents(chunks,embedding)

# 5. Retriever to fetch data from store
retrievers = vectorstore.as_retriever()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","Use information from documents to answer questions concisely with polite and friendly tone"),
    ("human","Question: {question}, Related information: {context}")
])

# Model 
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")

# Chain 
rag_chain = (
    {"context":retrievers,"question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

result = rag_chain.invoke("What 's the company name and its address?")
print(result)