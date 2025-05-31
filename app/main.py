import os
from dotenv import load_dotenv

from chunker import RecursiveChunker
from knowledgebase import InMemoryKnowledgeBase
from semantic_splitter import SemanticTextChunker

from llm import OpenAILLM


class CliApp:
    def __init__(self, llm: OpenAILLM):
        self._llm = llm

    def run(self) -> None:
        while True:
            question: str = input("Zadaj pytanie: ")
            if question.lower() == "exit":
                break
            answer: str = self._llm.run(question)
            print(f"Odpowiedź: {answer}")


if __name__ == "__main__":
    load_dotenv(f"{os.path.dirname(os.path.abspath(__file__))}/../.env")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")

    chunker = RecursiveChunker()
    # semantic_chunker = SemanticTextChunker(
    #     api_key=openai_api_key,
    # )

    knowledge_base = InMemoryKnowledgeBase(openai_api_key, chunker)
    knowledge_base.process_txt_to_vectorstore("../data/paracetamol.txt")

    openai_llm = OpenAILLM(openai_api_key)
    app = CliApp(openai_llm)
    app.run()
