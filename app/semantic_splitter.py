import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from chunker import Chunker


class SemanticTextChunker(Chunker):
    def __init__(
            self,
            api_key: str,
            buffer_size: int = 1,
            breakpoint_percentile_threshold: int = 95
    ):
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.ai_handler = OpenAI(api_key=api_key)

    def chunk(self, text: str) -> list[str]:
        single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
        buffered_chunks = self._create_buffered_chunks(sentences=single_sentences_list)
        embeddings = self._get_embeddings(text=buffered_chunks)
        distances = self._calculate_cosine_distances(embeddings=embeddings)
        breakpoints = self._find_breakpoints(distances=distances)
        return self._split_into_chunks(sentences=single_sentences_list, breakpoints=breakpoints)

    def _create_buffered_chunks(self, sentences: list[str]) -> list[str]:
        buffered_chunks = []

        for i, sentence in enumerate(sentences):
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(len(sentences), i + 1 + self.buffer_size)

            chunk_sentences = sentences[start_idx:end_idx]
            buffered_chunk = ' '.join(chunk_sentences)
            buffered_chunks.append(buffered_chunk)
        return buffered_chunks

    def _get_embeddings(self, text: list[str]) -> list[list[float]]:
        response = self.ai_handler.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return [item.embedding for item in response.data]

    def _calculate_cosine_distances(self, embeddings: list[list[float]]) -> list[float]:
        return [
            1 - cosine_similarity(
                [embeddings[embedding_index]],
                [embeddings[embedding_index + 1]]
            )[0][0]
            for embedding_index in range(len(embeddings) - 1)
        ]

    def _find_breakpoints(self, distances: list[float]) -> list[int]:
        threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
        return [i for i, distance in enumerate(distances, start=1) if distance > threshold]

    def _split_into_chunks(self, sentences: list[str], breakpoints: list[int]) -> list[str]:
        chunks = []
        start_idx = 0
        for breakpoint in breakpoints:
            chunk = ' '.join(sentences[start_idx:breakpoint])
            if chunk.strip():
                chunks.append(chunk)
            start_idx = breakpoint

        final_chunk = ' '.join(sentences[start_idx:])
        if final_chunk.strip():
            chunks.append(final_chunk)
        return chunks
