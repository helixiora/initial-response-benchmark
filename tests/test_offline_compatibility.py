"""Validate tool helpers without model downloads, credentials, or provider calls."""

import importlib
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


class OfflineToolCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        cls.repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(cls.repo))
        stack.callback(sys.path.remove, str(cls.repo))
        stack.enter_context(patch.dict(
            os.environ, {"OPENAI_API_KEY": "offline-test", "PINECONE_API_KEY": "offline-test"}
        ))
        stack.enter_context(patch.object(
            socket.socket, "connect", side_effect=AssertionError("Offline tests cannot use the network")
        ))
        stack.enter_context(patch("openai.OpenAI"))
        stack.enter_context(patch("pinecone.Pinecone"))
        stack.enter_context(patch("dotenv.load_dotenv", return_value=False))
        cls.setup = importlib.import_module("tool_poc.vector_setup")
        cls.interactive = importlib.import_module("tool_poc.vector_interactive")

    def test_chunks_have_bounded_content_and_source_metadata(self):
        text = "A searchable sentence about synthetic records. " * 50
        chunks = self.setup.chunk_text(text, "records.txt", "Synthetic")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk["text"]) <= 500 for chunk in chunks))
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["metadata"], {
                "dataset": "Synthetic", "filename": "records.txt",
                "chunk_index": index, "chunk_count": len(chunks),
            })

    def test_chunk_ids_are_stable_for_the_same_source(self):
        text = "Deterministic synthetic input. " * 30
        self.assertEqual(
            self.setup.chunk_text(text, "sample.md", "Synthetic"),
            self.setup.chunk_text(text, "sample.md", "Synthetic"),
        )

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(self.setup.chunk_text("", "empty.txt", "Synthetic"), [])

    def test_dataset_discovery_only_includes_text_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "synthetic"
            dataset.mkdir()
            (dataset / "notes.md").write_text("Synthetic notes", encoding="utf-8")
            (dataset / "ignored.bin").write_bytes(b"not a text source")
            found = self.setup.scan_datasets_directory(directory)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["files"], ["notes.md"])

    def test_json_tool_response_parses_without_provider_calls(self):
        self.assertEqual(self.interactive.safe_json_parse('{"answer": "synthetic"}'), {"answer": "synthetic"})


if __name__ == "__main__":
    unittest.main()
