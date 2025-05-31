# -*- encoding: utf-8 -*-
import unittest
from chunker import TextSplitterOverlapChunker, TextSplitterChunker


class TestTextSplitterChunker(unittest.TestCase):
    def test_split_basic(self):
        chunker = TextSplitterChunker(chunk_size=3)
        text = "abcdef"
        chunks = chunker.chunk(text)
        expected = ["abc", "def"]
        self.assertEqual(chunks, expected)

    def test_split_with_remainder(self):
        chunker = TextSplitterChunker(chunk_size=4)
        text = "abcdefg"
        chunks = chunker.chunk(text)
        expected = ["abcd", "efg"]
        self.assertEqual(chunks, expected)

    def test_split_short_text(self):
        chunker = TextSplitterChunker(chunk_size=10)
        text = "abc"
        chunks = chunker.chunk(text)
        expected = ["abc"]
        self.assertEqual(chunks, expected)


class TestTextSplitterOverlapChunker(unittest.TestCase):
    def test_split_basic(self):
        chunker = TextSplitterOverlapChunker(chunk_size=5, chunk_overlap=2)
        text = "abcdefghij"
        chunks = chunker.chunk(text)
        expected = ["abcde", "defgh", "ghij"]
        self.assertEqual(chunks, expected)

    def test_split_short_text(self):
        chunker = TextSplitterOverlapChunker(chunk_size=5, chunk_overlap=2)
        text = "abc"
        chunks = chunker.chunk(text)
        expected = ["abc"]
        self.assertEqual(chunks, expected)

    def test_split_exact_overlap(self):
        chunker = TextSplitterOverlapChunker(chunk_size=4, chunk_overlap=2)
        text = "abcdefgh"
        chunks = chunker.chunk(text)
        expected = ["abcd", "cdef", "efgh"]
        self.assertEqual(chunks, expected)


if __name__ == "__main__":
    unittest.main()
