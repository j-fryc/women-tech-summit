from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def chunk(self, data: str) -> list[str]:
        raise NotImplementedError("This method should be overridden by subclasses.")


class TextSplitterChunker(Chunker):
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def chunk(self, data: str) -> list[str]:
        return [
            data[i : i + self.chunk_size] for i in range(0, len(data), self.chunk_size)
        ]


class TextSplitterOverlapChunker(Chunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, data: str) -> list[str]:
        chunks = []
        i = 0
        while i < len(data):
            chunk = data[i : i + self.chunk_size]
            chunks.append(chunk)
            if i + self.chunk_size >= len(data):
                break
            i += self.chunk_size - self.chunk_overlap
        return chunks


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