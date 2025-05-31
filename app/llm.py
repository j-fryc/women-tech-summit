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

    def run(self, question: str) -> str:
        response = self._llm.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś pomocnym asystentem. Odpowiedź zwięźle i na temat.",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=self._temperature,
        )
        return response.choices[0].message.content
