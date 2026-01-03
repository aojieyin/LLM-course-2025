from llmsherpa.readers import LayoutPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Cinfigure Settings 
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Read PDF
llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/2024q1-alphabet-earnings-release-pdf.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
doc = pdf_reader.read_pdf(pdf_url)

# 3. Dynamically load all sections to avoid hardcoding
documents = []
for chunk in doc.chunks():
    # Convert each chunk into a LlamaIndex Document object
    documents.append(Document(text=chunk.to_context_text(), extra_info={}))

# 4. Construction of the vector store index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


# 5. Test reasoning and calculation capabilities
print("--- Start to test ---")

q1 = "What was Google's operating margin for Q1 2024 vs Q1 2023?"
print(f"Question: {q1}\nAnswer: {query_engine.query(q1)}\n")

q2 = "What is the total of 'Google Search & others' and 'YouTube ads' revenues for 2024?"
print(f"Question: {q2}\nAnswer: {query_engine.query(q2)}\n")


q3 = "Calculate the percentage increase in Total Revenues from 2023 to 2024."
print(f"Question: {q3}\nAnswer: {query_engine.query(q3)}\n")