# Zbuduj AI Chatbota w Pythonie

Repozytorium pomocniczne na warsztaty.

## Guide

1. Zapoznaj się z treścią jupyter notebooków w katalogu `notebooks`
    - (opcjonalnie) na potrzeby samodzielnego uruchomienia kodu:
        - uzupełnij `.env` tokenem z openai <https://platform.openai.com/api-keys>
        - dla `02_knowledge.ipynb`, uruchom bazę postgres poleceniem `docker compose up`
1. Zapoznaj się z aplikacją w katalogu `app`.
1. Przygotuj środowisko:
    - Przygotuj virtualenv:
        - pip
            - `python -m venv venv`
            - `.\venv\Scripts\Activate` lub `source venv/bin/activate`
            - `pip install -r requirements.txt`
        - conda
            - `conda create -n venv`
            - `conda activate venv`
            - `conda install --yes --file requirements.txt`
    - Uzupełnij `.env` tokenem z openai <https://platform.openai.com/api-keys>
    - Uruchom main `python main.py`
    - Uruchom testy `python -m unittest discover .`
1. **Zadanie 1**: Użyj bazy wiedzy w prompcie do LLM
    - Dodaj użycie klasy `InMemoryKnowledgeBase` w `OpenAILLM`
    - Przeszukaj `InMemoryKnowledgeBase` używając jako query pytania użytkownika
    - Dopisz wyszukane chunki z bazy wiedzy jako kontekst w prompcie do LLM
1. **Zadanie 2**: Dodaj własny chunker w pliku chunker.py
    - Zaimplementuj klasę `TextSplitterChunker`
    - (opcjonalnie) zaimplementuj klasę `TextSplitterOverlapChunker`


Kontakt do prowadzących:

Maciej Jagiełło:
- https://www.linkedin.com/in/maciej-jagiełło-233b8386/
- maciej.jagiello@external.t-mobile.pl

Jakub Fryc:
- https://www.linkedin.com/in/jakub-fryc-04a813224/
- jakub.fryc@t-mobile.pl