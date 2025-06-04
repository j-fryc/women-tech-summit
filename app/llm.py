# -*- encoding: utf-8 -*-

from openai import OpenAI

from knowledgebase import InMemoryKnowledgeBase


class OpenAILLM:
    def __init__(
        self,
        openai_api_key: str,
        knowledge_base: InMemoryKnowledgeBase = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
    ):
        self._model_name = model_name
        self._knowledge_base = knowledge_base
        self._temperature = temperature
        self._llm = OpenAI(api_key=openai_api_key)

    def augment(self, docs_with_score, user_question) -> tuple[str, str]:
        docs_filtered_by_score = [(doc, score) for doc, score in docs_with_score if score > 0.5]

        if docs_filtered_by_score:
            docs_formatted = "\n".join(
                [f"Document: {doc.page_content} Score: {score}" for doc, score in docs_filtered_by_score])
        else:
            docs_formatted = "Brak dokumentów o wystarczającym poziomie podobieństwa."

        system_prompt = "Jesteś pomocnym asystentem. Odpowiedź zwięźle i na temat TYLKO na bazie podanego kontekstu\n"
        user_prompt = (f"kontekst z miarą podobieństwa do pytania użytkownika: {docs_formatted}\n"
                       f"pytanie użytkownika: {user_question}")
        return system_prompt, user_prompt

    def run(self, question: str) -> str:
        docs_with_score = self._knowledge_base.search_with_score(question)

        system_prompt, user_prompt = self.augment(docs_with_score, question)

        response = self._llm.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=self._temperature,
        )
        return response.choices[0].message.content
