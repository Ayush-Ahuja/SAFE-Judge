"""Regression checks for tied confidence and incomplete retrieval judgments."""

import unittest
import numpy as np

from scripts.evaluate_proposal_metrics import confidence_aurc, DL2020Rankings


class ProposalMetricTests(unittest.TestCase):
    def test_tie_averaging_does_not_depend_on_arbitrary_row_order(self):
        self.assertEqual(confidence_aurc([.8, .8], [1, 0]), .5)
        self.assertEqual(confidence_aurc([.8, .8], [0, 1]), .5)
        self.assertEqual(confidence_aurc([.9, .9], [1, 1]), 0.)
        self.assertEqual(confidence_aurc([.9, .9], [0, 0]), 1.)

    def test_known_confidence_order_has_correct_discrete_area(self):
        self.assertAlmostEqual(confidence_aurc([.9, .7], [0, 1]), .75)
        self.assertAlmostEqual(confidence_aurc([.9, .7], [1, 0]), .25)

    def test_unjudged_zero_gain_preserves_original_rank_positions(self):
        ranking = object.__new__(DL2020Rankings)
        ranking.discount = np.asarray([1., 1/np.log2(3)])
        ranking.query_indices = [np.asarray([0, 1]), np.asarray([2, 3])]
        # Index four represents an unjudged position, distinct from a judged label.
        ranking.indices = np.asarray([[[0, 4], [2, 3]], [[1, 0], [3, 2]]])
        scores = ranking.scores(np.asarray([1, 0, 1, 0]))
        self.assertAlmostEqual(scores[0], 1.)
        self.assertAlmostEqual(scores[1], 1/np.log2(3))


if __name__ == "__main__":
    unittest.main()
