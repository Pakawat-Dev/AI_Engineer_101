# AI_Engineer_101
RAG using LangChain and OpenAI to create an intelligent question-answering system that can retrieve information from your documents.

## What is RAG?

RAG (Retrieval-Augmented Generation) is an AI technique that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using AI to create natural language responses based on retrieved information

This allows AI models to answer questions using your specific documents and data, rather than just their training knowledge.

## How This Project Works

This RAG system follows these steps:

1. **Load Documents** → Read text files containing your data
2. **Split Text** → Break large documents into smaller, manageable chunks
3. **Create Embeddings** → Convert text chunks into numerical vectors
4. **Store in Vector Database** → Save vectors for fast similarity search
5. **Retrieve Relevant Info** → Find chunks most similar to user questions
6. **Generate Answers** → Use AI to create responses based on retrieved information

## Prerequisites

Before you start, make sure you have:

- Python 3.8 or higher installed
- An OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/))
- Basic knowledge of Python

## Step-by-Step Setup

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <your-repository-url>
cd <project-folder>

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

The required packages are:
- `langchain-community` - Document loaders and vector stores
- `langchain-text-splitters` - Text splitting utilities
- `langchain-openai` - OpenAI integration
- `langchain-core` - Core LangChain functionality
- `faiss-cpu` - Vector database for similarity search
- `python-dotenv` - Environment variable management

### Step 4: Set Up Your OpenAI API Key

1. Create a `.env` file in the project root:
```bash
touch .env  # On Windows: type nul > .env
```

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: Never share your API key or commit it to version control!

### Step 5: Prepare Your Data

1. The system reads from `data.txt` by default
2. You can replace the content in `data.txt` with your own information
3. The current example contains company information about "DevForever"

## Running the Application

### Basic Usage

```bash
python RAG_Langchain.py
```

This will:
1. Load and process the `data.txt` file
2. Ask the question: "What's the company name and its address?"
3. Print the AI-generated answer based on the document content

### Customizing Questions

To ask different questions, modify the last line in `RAG_Langchain.py`:

```python
# Change this line to ask your own questions
result = rag_chain.invoke("Your question here")
print(result)
```

Example questions you can try:
- "What products does the company sell?"
- "Who is the founder?"
- "What are the contact details?"
- "When was the company founded?"

## Understanding the Code

### 1. Document Loading
```python
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()
```
- Loads text from your data file
- Handles UTF-8 encoding for international characters

### 2. Text Splitting
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)
```
- Breaks large text into smaller chunks (100 characters each)
- Overlaps chunks by 20 characters to maintain context
- Smaller chunks = more precise retrieval

### 3. Creating Embeddings
```python
embedding = OpenAIEmbeddings()
```
- Converts text chunks into numerical vectors
- Similar text will have similar vectors
- Enables semantic search (meaning-based, not just keyword matching)

### 4. Vector Store
```python
vectorstore = FAISS.from_documents(chunks, embedding)
```
- FAISS (Facebook AI Similarity Search) stores and indexes vectors
- Enables fast similarity search across thousands of documents
- CPU-based version (no GPU required)

### 5. Retrieval System
```python
retrievers = vectorstore.as_retriever()
```
- Creates a retriever that finds most relevant chunks
- Returns top matching pieces of information for any question

### 6. AI Model and Prompt
```python
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")
prompt = ChatPromptTemplate.from_messages([...])
```
- Uses OpenAI's language model for generating responses
- Custom prompt ensures polite, friendly, and concise answers
- Instructs AI to use only information from your documents

### 7. RAG Chain
```python
rag_chain = (
    {"context": retrievers, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
- Combines all components into a processing pipeline
- Question → Retrieve context → Generate answer → Parse output

## Customization Options

### Adjusting Chunk Size
```python
# Smaller chunks = more precise, larger chunks = more context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,    # Increase for more context
    chunk_overlap=50   # Increase for better continuity
)
```

### Changing the AI Model
```python
# Use different OpenAI models
llm = ChatOpenAI(model="gpt-4")  # More powerful but expensive
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Faster and cheaper
```

### Modifying the Prompt
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your custom system instructions here"),
    ("human", "Question: {question}, Context: {context}")
])
```

### Using Different File Types
```python
# For PDF files
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")

# For web pages
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
```

## Troubleshooting

### Common Issues

1. **"No module named 'langchain'"**
   - Solution: Make sure you activated your virtual environment and installed requirements

2. **"OpenAI API key not found"**
   - Solution: Check your `.env` file and ensure `OPENAI_API_KEY` is set correctly

3. **"File not found: data.txt"**
   - Solution: Make sure `data.txt` exists in the same directory as the script

4. **Poor answer quality**
   - Try adjusting chunk size and overlap
   - Improve your source documents with clearer, more detailed information
   - Modify the system prompt for better instructions

### Performance Tips

- **Larger documents**: Increase chunk size to 500-1000 characters
- **Better accuracy**: Decrease chunk size to 50-100 characters
- **Faster responses**: Use `gpt-3.5-turbo` instead of `gpt-4`
- **Better context**: Increase chunk overlap to 30-50 characters

## Next Steps

Once you're comfortable with the basics, try:

1. **Multiple Documents**: Load multiple text files
2. **Web Integration**: Create a web interface with Flask or Streamlit
3. **Different File Types**: Support PDF, Word, or CSV files
4. **Advanced Retrieval**: Implement hybrid search (keyword + semantic)
5. **Memory**: Add conversation history for follow-up questions

## File Structure

```
project/
├── RAG_Langchain.py      # Main application
├── data.txt              # Your knowledge base
├── requirements.txt      # Python dependencies
├── .env                  # API keys (create this)
├── README.md            # This file
└── .venv/               # Virtual environment (optional)
```

## Resources for Learning More

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [RAG Concepts Explained](https://python.langchain.com/docs/tutorials/rag/)

## License

This project is for educational purposes. Make sure to comply with OpenAI's usage policies and any applicable licenses for the libraries used.
