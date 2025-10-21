from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0,
    separator='',
)

result = splitter.split_documents(docs)

print(result[2].page_content)