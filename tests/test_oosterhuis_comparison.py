"""Check top-10 adaptation and isolation of unseen gold from metric inference."""
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.oosterhuis_inputs import binary_top10
from scripts.run_oosterhuis_matched import reference_dataset


class OosterhuisComparisonTests(unittest.TestCase):
    def test_binary_conversion_preserves_rank_and_excludes_rank_eleven(self):
        original = SimpleNamespace(ranks=np.array([2, 11, 1]), scores=np.array([5., 1., 6.]),
            query_segments=np.array([0, 0, 0]),
            pred_label_dist=np.array([[.1, .2, .3, .4], [.4, .3, .2, .1], [.2, .3, .1, .4]]),
            true_label_dist=np.eye(4)[[1, 2, 3]])
        rows = binary_top10(original)
        self.assertEqual([r["rank"] for r in rows], [2, 1])
        self.assertEqual([r["true_binary"] for r in rows], [0, 1])
        np.testing.assert_allclose([r["prob_relevant"] for r in rows], [.7, .5])

    def test_unseen_gold_is_replaced_by_placeholders(self):
        rows = [{"query_id": "q", "doc_id": "d", "prob_relevant": .7,
                 "true_binary": 1, "rank": 1, "retrieval_score": 9.}]
        first = reference_dataset(rows, SimpleNamespace)
        second = reference_dataset([{**r, "true_binary": 0} for r in rows], SimpleNamespace)
        np.testing.assert_array_equal(first.true_label_dist, second.true_label_dist)
        np.testing.assert_array_equal(first.true_label_dist, [[1, 0]])
        np.testing.assert_allclose(first.pred_label_dist, [[.3, .7]], rtol=1e-14, atol=0)

    def test_calibration_uses_only_explicitly_purchased_labels(self):
        rows = [{"query_id": "q", "doc_id": "d", "prob_relevant": .2,
                 "true_binary": 1, "rank": 2, "retrieval_score": 4.}]
        result = reference_dataset(rows, SimpleNamespace, {("q", "d"): 0})
        np.testing.assert_array_equal(result.true_label_dist, [[1, 0]])
        with self.assertRaises(KeyError):
            reference_dataset(rows, SimpleNamespace, {})
