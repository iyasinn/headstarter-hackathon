import openai
import utils

openai.api_key = utils.get_openai_api_key()

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_dir="data").load_data()

# print()
# print(type(documents), "\n")
# print(len(documents), "\n")
# print(type(documents[0]), "\n")
# print(documents[0])

# Basic RAG pipeilne

print("part 1")
from llama_index import Document

document = Document(text="\n\n".join(doc.text for doc in documents))

from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI

print("part 2")

# Temperature is 0.1, and high temperature can give goofy words
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)

print("part 3")

index = VectorStoreIndex.from_documents([document], service_context=service_context)
query_engine = index.as_query_engine()

print("part 4)")

query = "What are the steps to become a successful stock trader?"
response = query_engine.query(query)
print(str(response))

# How does this differ from chatpgt
# huggin face is a website where everyhting has models
