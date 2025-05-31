import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from chunker import Chunker


class InMemoryKnowledgeBase:
    def __init__(self, openai_api_key: str, chunker: Chunker):
        self._vectorstore = InMemoryVectorStore(
            OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=openai_api_key
            )
        )
        self._chunker = chunker

    def process_txt_to_vectorstore(self, path_to_file: str) -> None:
        with open(path_to_file, "r", encoding="utf-8") as open_file:
            chunks = self._chunker.chunk(open_file.read())

        documents = [Document(page_content=chunk) for chunk in chunks]

        self._vectorstore.add_documents(documents)

    def process_pdf_to_vectorstore(self, path_to_file: str) -> None:
        with open(path_to_file, "rb") as file_handler:
            reader = PyPDF2.PdfReader(file_handler)

            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()

        chunks = self._chunker.chunk(full_text)

        documents = [Document(page_content=chunk) for chunk in chunks]

        self._vectorstore.add_documents(documents)

    def search(self, query: str, k: int = 3) -> list[Document]:
        return self._vectorstore.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        return self._vectorstore.similarity_search_with_score(query, k=k)
