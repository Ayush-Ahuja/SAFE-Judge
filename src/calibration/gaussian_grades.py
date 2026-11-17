"""Class-conditional Gaussian models for graded relevance and binary error risk."""

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture


class GradeDensityModel:
    """Regularized LDA/QDA or per-grade mixtures on calibration features.

    Feature scaling, priors and densities are fitted using the supplied labeled
    calibration rows. Missing grades use the pooled Gaussian and a smoothed
    prior. This fallback is recorded in the model parameters.
    """

    def __init__(self, kind, grades, covariance_shrinkage=.25, prior_pseudocount=.5, seed=0):
        if kind not in ("lda", "qda", "gmm2", "logistic", "prior"):
            raise ValueError("Unknown grade model")
        self.kind = kind
        self.grades = np.asarray(grades, dtype=int)
        if len(self.grades) < 2 or len(set(self.grades.tolist())) != len(self.grades):
            raise ValueError("Distinct known grade labels required")
        if not 0 <= covariance_shrinkage <= 1 or prior_pseudocount <= 0:
            raise ValueError("Invalid covariance regularization or prior")
        self.shrinkage, self.pseudocount, self.seed = covariance_shrinkage, prior_pseudocount, seed

    @staticmethod
    def covariance(x):
        centered = x-x.mean(axis=0)
        return centered.T @ centered / len(x)

    @staticmethod
    def log_gaussian(x, mean, covariance):
        sign, determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Gaussian covariance must be positive definite")
        centered = x-mean
        squared = np.sum(centered*np.linalg.solve(covariance, centered.T).T, axis=1)
        return -.5*(x.shape[1]*np.log(2*np.pi)+determinant+squared)

    def fit(self, features, labels):
        x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=int)
        if x.ndim != 2 or not len(x) or len(x) != len(y) or not np.isfinite(x).all() or not np.isin(y, self.grades).all():
            raise ValueError("Finite calibration features and known grades required")
        self.center, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.
        x = (x-self.center)/self.scale
        self.counts = np.asarray([(y == grade).sum() for grade in self.grades])
        self.priors = (self.counts+self.pseudocount)/(len(y)+len(self.grades)*self.pseudocount)
        pooled_global = self.covariance(x)+np.eye(x.shape[1])*1e-4
        means, within = [], []
        for grade in self.grades:
            group = x[y == grade]
            means.append(group.mean(axis=0) if len(group) else x.mean(axis=0))
            within.append(self.covariance(group) if len(group) >= 2 else pooled_global)
        self.means = np.asarray(means)
        shared = sum(n*cov for n, cov in zip(self.counts, within))/len(y)
        shared = shared+np.eye(x.shape[1])*1e-4
        self.covariances, self.mixtures = [], []
        for grade, covariance in zip(self.grades, within):
            covariance = shared if self.kind == "lda" else (1-self.shrinkage)*covariance+self.shrinkage*shared
            covariance = covariance+np.eye(x.shape[1])*1e-4
            self.covariances.append(covariance)
            group = x[y == grade]
            mixture = None
            if self.kind == "gmm2" and len(group) >= 2:
                components = 2 if len(group) >= 10 else 1
                mixture = GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-3,
                                          max_iter=500, n_init=3, random_state=self.seed).fit(group)
                if not mixture.converged_:
                    raise RuntimeError("Grade mixture did not converge")
            self.mixtures.append(mixture)
        self.covariances = np.asarray(self.covariances)
        self.logistic = None
        if self.kind == "logistic" and len(np.unique(y)) > 1:
            self.logistic = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, tol=1e-8).fit(x, y)
            if self.logistic.n_iter_.max() >= 2000:
                raise RuntimeError("Multiclass logistic regression did not converge")
        return self

    def predict_proba(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.center) or not np.isfinite(x).all():
            raise ValueError("Prediction feature shape or values invalid")
        x = (x-self.center)/self.scale
        if self.kind == "prior" or (self.kind == "logistic" and self.logistic is None):
            return np.broadcast_to(self.priors, (len(x), len(self.grades))).copy()
        if self.kind == "logistic":
            result = np.full((len(x), len(self.grades)), 1e-12)
            scores = self.logistic.predict_proba(x)
            for column, grade in enumerate(self.logistic.classes_):
                result[:, int(np.flatnonzero(self.grades == grade)[0])] = scores[:, column]
            return result/result.sum(axis=1, keepdims=True)
        columns = []
        for index, (mean, covariance, mixture) in enumerate(zip(self.means, self.covariances, self.mixtures)):
            likelihood = mixture.score_samples(x) if mixture is not None else self.log_gaussian(x, mean, covariance)
            columns.append(likelihood+np.log(self.priors[index]))
        joint = np.column_stack(columns)
        result = np.exp(joint-logsumexp(joint, axis=1, keepdims=True))
        if not np.isfinite(result).all():
            raise ArithmeticError("Non-finite grade posterior")
        return result

    def parameters(self):
        return {"kind": self.kind, "grades": self.grades.tolist(), "calibration_counts": self.counts.tolist(),
                "missing_grades": self.grades[self.counts == 0].tolist(), "class_priors": self.priors.tolist(),
                "feature_center": self.center.tolist(), "feature_scale": self.scale.tolist(),
                "standardized_class_means": self.means.tolist(), "standardized_covariances": self.covariances.tolist(),
                "covariance_shrinkage": self.shrinkage, "prior_pseudocount": self.pseudocount,
                "mixture_components": [m.n_components if m is not None else 0 for m in self.mixtures]}


def relevance_probability(grade_probabilities, grades, relevance_threshold):
    return np.asarray(grade_probabilities)[:, np.asarray(grades) >= relevance_threshold].sum(axis=1)


def binary_error_probability(relevant_probability, fixed_labels):
    p = np.asarray(relevant_probability)
    return np.where(np.asarray(fixed_labels) == 1, 1-p, p)
