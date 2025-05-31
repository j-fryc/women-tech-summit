from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def chunk(self, data: str) -> list[str]:
        raise NotImplementedError("This method should be overridden by subclasses.")


class TextSplitterChunker(Chunker): ...


class TextSplitterOverlapChunker(Chunker): ...


class RecursiveChunker(Chunker):
    def __init__(self):
        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=0,
            separators=["\n\n", "\n", "."],
            is_separator_regex=False
        )

    def chunk(self, data: str) -> list[str]:
        test = self._chunker.split_text(data)
        return test