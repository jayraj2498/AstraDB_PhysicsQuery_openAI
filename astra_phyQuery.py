# First cell content
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.chains import VectorDBQA

# Load the PDF and convert to a list of documents
loader = PyPDFLoader("physics.pdf")
documents = loader.load()

# Initialize OpenAI embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize AstraDB
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv("ASTRA_DB_SECURE_BUNDLE_PATH")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

vectorstore = AstraDB.from_documents(
    documents,
    embeddings,
    astra_db_secure_bundle_path=ASTRA_DB_SECURE_BUNDLE_PATH,
    astra_db_application_token=ASTRA_DB_APPLICATION_TOKEN,
    namespace="pdf-physics",
)

# Initialize the QA chain
qa = VectorDBQA.from_chain_type(
    llm=openai.ChatCompletion(temperature=0),
    chain_type="stuff",
    vectorstore=vectorstore,
)

# Ask a question
question = "Explain the concept of Coulomb's Law."
answer = qa.run(question)
print(answer)

# Second cell content
import requests

# Define a function to fetch data from an API
def fetch_data(api_url, params=None):
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Example usage
api_url = "https://api.example.com/data"
params = {"key": "value"}
data = fetch_data(api_url, params)
print(data)

