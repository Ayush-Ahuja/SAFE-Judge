"""Numerical checks for grade posteriors, sparse grades and error aggregation."""

import unittest
import numpy as np

from src.calibration.gaussian_grades import GradeDensityModel, relevance_probability, binary_error_probability


class GaussianGradeTests(unittest.TestCase):
    def test_symmetric_lda_matches_equal_posterior_at_midpoint(self):
        x = np.asarray([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
        model = GradeDensityModel("lda", [0, 1]).fit(x, [0, 0, 0, 1, 1, 1])
        p = model.predict_proba([[0.], [-3.], [3.]])
        np.testing.assert_allclose(p[0], [.5, .5], atol=1e-12)
        self.assertGreater(p[1, 0], .99)
        self.assertGreater(p[2, 1], .99)

    def test_qda_uses_class_variance_for_center_and_tails(self):
        model = GradeDensityModel("qda", [0, 1], covariance_shrinkage=0.).fit(
            [[-2.], [0.], [2.], [-.2], [0.], [.2]], [0, 0, 0, 1, 1, 1])
        p = model.predict_proba([[0.], [4.]])
        self.assertGreater(p[0, 1], .8)
        self.assertGreater(p[1, 0], .99)

    def test_sparse_and_absent_grades_have_finite_normalized_posteriors(self):
        for family in ("lda", "qda", "gmm2", "logistic", "prior"):
            model = GradeDensityModel(family, [0, 1, 2]).fit([[0., 1.], [0., 1.], [1., 1.]], [0, 0, 1])
            p = model.predict_proba([[0., 1.], [10., 1.]])
            self.assertEqual(p.shape, (2, 3))
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1), 1.)
            self.assertEqual(model.parameters()["missing_grades"], [2])

    def test_grade_probability_aggregation_predicts_error_of_fixed_label(self):
        p = relevance_probability([[.2, .3, .5], [.1, .2, .7]], [0, 1, 2], 1)
        np.testing.assert_allclose(p, [.8, .9])
        np.testing.assert_allclose(binary_error_probability(p, [0, 1]), [.8, .1])


if __name__ == "__main__":
    unittest.main()
