"""Streaming preparation preserves selected text and rejects missing passages."""

from types import SimpleNamespace
import unittest

from scripts.prepare_pilot import collect_document_text


class StreamingDocumentsTests(unittest.TestCase):
    def test_stream_matches_store_without_building_store(self):
        records = [SimpleNamespace(doc_id=str(i), text=f"passage {i}") for i in range(4)]
        by_id = {doc.doc_id: doc for doc in records}
        store_dataset = SimpleNamespace(docs_store=lambda: SimpleNamespace(get=by_id.__getitem__))
        visited = []

        def stream():
            for doc in records:
                visited.append(doc.doc_id)
                yield doc

        stream_dataset = SimpleNamespace(docs_iter=stream)
        expected = collect_document_text(store_dataset, {"0", "2"}, ["text"])
        actual = collect_document_text(stream_dataset, {"0", "2"}, ["text"], "stream")
        self.assertEqual(actual, expected)
        self.assertEqual(visited, ["0", "1", "2"])

    def test_missing_document_is_fatal(self):
        dataset = SimpleNamespace(docs_iter=lambda: iter([SimpleNamespace(doc_id="a", text="text")]))
        with self.assertRaisesRegex(ValueError, "Missing 1 selected documents"):
            collect_document_text(dataset, {"a", "b"}, ["text"], "stream")


if __name__ == "__main__":
    unittest.main()
