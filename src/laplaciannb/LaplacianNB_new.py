import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)


class LaplacianNB(_BaseDiscreteNB):
    """Naive Bayes classifier for Laplacian modified models.

    Like BernoulliNB, this classifier is suitable for binary/boolean data. The
    difference is that while BernoulliNB processes all features, the
    Laplacian modified approach uses only positive (non-zero) features.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will be set to 1e-10.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    feature_count_ : ndarray of shape (n_classes,)
        Sum of positive features for each class.

    feature_count_per_class_ : ndarray of shape (n_classes, n_features_in_)
        Number of positive bits encountered for each (class, feature) during fitting.

    feature_all_ : float
        Total number of positive features encountered.

    feature_log_prob_ : ndarray of shape (n_classes, n_features_in_)
        Empirical log probability of positive bit features given a class, P(x_i|y).

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,), optional
        Names of features seen during fit. Defined only when X
        has feature names that are all strings.

    References
    ----------
    Nidhi; Glick, M.; Davies, J. W.; Jenkins, J. L. Prediction of biological targets
    for compounds using multiple-category Bayesian models trained on chemogenomics
    databases. J. Chem. Inf. Model. 2006, 46, 1124– 1133,
    https://doi.org/10.1021/ci060003g

    Lam PY, Kutchukian P, Anand R, et al.
    Cyp1 inhibition prevents doxorubicin-induced cardiomyopathy
    in a zebrafish heart-failure model. Chem Bio Chem. 2020:cbic.201900741.
    https://doi.org/10.1002/cbic.201900741
    """

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _check_X(self, X):
        """Validate X for predict methods."""
        # Detect legacy input formats first, before sklearn validation
        self._detect_legacy_input_format(X)

        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32, np.int64, np.int32, bool], ensure_2d=True
        )

        # Convert to binary if needed (handle sparse matrices properly)
        if sparse.issparse(X):
            # For sparse matrices, check if any value is not 0 or 1
            if X.dtype != bool and not np.all((X.data == 0) | (X.data == 1)):
                X = (X != 0).astype(np.float64)
        else:
            # For dense matrices
            if not np.array_equal(X, X.astype(bool)):
                X = (X != 0).astype(np.float64)

        return X

    def _check_X_y(self, X, y, reset=True):
        """Validate X and y for fit."""
        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csr", "csc"],
            dtype=[np.float64, np.float32, np.int64, np.int32, bool],
            ensure_2d=True,
        )

        # Convert to binary if needed (handle sparse matrices properly)
        if sparse.issparse(X):
            # For sparse matrices, check if any value is not 0 or 1
            if X.dtype != bool and not np.all((X.data == 0) | (X.data == 1)):
                X = (X != 0).astype(np.float64)
        else:
            # For dense matrices
            if not np.array_equal(X, X.astype(bool)):
                X = (X != 0).astype(np.float64)

        return X, y

    def _count_feature_occurrences(self, X, Y):
        """Count how many times each feature appears positive for each class.

        This implements the core Laplacian NB algorithm: counting only positive bits.
        """
        n_classes = Y.shape[1]
        n_features = X.shape[1]

        # Initialize counters
        feature_count_per_class = np.zeros((n_classes, n_features), dtype=np.float64)
        feature_sum_per_class = np.zeros(n_classes, dtype=np.float64)

        # Count positive features for each class
        if sparse.issparse(X):
            X = X.tocsr()
            for i in range(n_classes):
                class_mask = Y[:, i].astype(bool)
                if np.any(class_mask):
                    # Sum positive features for samples in this class
                    X_class = X[class_mask]
                    feature_count_per_class[i] = np.asarray(X_class.sum(axis=0)).ravel()
                    feature_sum_per_class[i] = feature_count_per_class[i].sum()
        else:
            for i in range(n_classes):
                class_mask = Y[:, i].astype(bool)
                if np.any(class_mask):
                    # Sum positive features for samples in this class
                    X_class = X[class_mask]
                    feature_count_per_class[i] = X_class.sum(axis=0)
                    feature_sum_per_class[i] = feature_count_per_class[i].sum()

        # Count total positive features across all samples
        total_feature_counts = np.asarray(X.sum(axis=0)).ravel() if sparse.issparse(X) else X.sum(axis=0)

        return feature_count_per_class, feature_sum_per_class, total_feature_counts

    def _init_counters(self, n_classes, n_features):
        """Initialize counters."""
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_per_class_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.feature_count_ = np.zeros(n_classes, dtype=np.float64)

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        (self.feature_count_per_class_, self.feature_count_, self.total_feature_counts_) = (
            self._count_feature_occurrences(X, Y)
        )

        self.feature_all_ = self.feature_count_.sum()
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities."""
        # Prior probability for each class (based on positive feature counts)
        prior = self.feature_count_ / (self.feature_all_ + np.finfo(float).eps)

        # Laplacian smoothing for feature probabilities
        # P(feature_i | class_j) = (count_ij + alpha) / (prior_j * total_i + alpha)
        denominator = np.outer(prior, self.total_feature_counts_) + alpha
        numerator = self.feature_count_per_class_ + alpha

        self.feature_prob_ = numerator / (denominator + np.finfo(float).eps)
        self.feature_log_prob_ = np.log(self.feature_prob_)

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X.

        Only considers positive (non-zero) features as per Laplacian NB.

        Note: This method returns the feature contributions only,
        following the original implementation. Class priors are added
        in predict_log_proba if needed.
        """
        check_is_fitted(self)

        # For Laplacian NB, we only use positive features
        if sparse.issparse(X):
            # Efficient sparse matrix multiplication
            # Only non-zero elements contribute to the sum
            jll = X @ self.feature_log_prob_.T
        else:
            # Dense matrix: mask zero elements
            X_binary = (X > 0).astype(np.float64)
            jll = X_binary @ self.feature_log_prob_.T

        # Do NOT add class priors here - follow original implementation
        # jll += self.class_log_prior_  # Commented out to match original

        return jll

    def _detect_legacy_input_format(self, X):
        """Detect and reject legacy input formats with helpful error message."""
        # Check for single set
        if isinstance(X, set):
            raise ValueError(
                "LEGACY INPUT FORMAT ERROR: You are trying to use a single set as input. "
                "This is no longer supported in the new version. "
                "\n\nTo fix this:\n"
                "1. Use the legacy version: from laplaciannb.legacy import LaplacianNB\n"
                "2. Or convert to proper format: from laplaciannb import convert_fingerprints\n"
                "   X = convert_fingerprints([your_set], n_bits=desired_size)"
            )

        # Check for list of sets
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], set):
            raise ValueError(
                "LEGACY INPUT FORMAT ERROR: You are trying to use the old list-of-sets format. "
                "This is no longer supported in the new version. "
                "\n\nTo fix this:\n"
                "1. Use the legacy version: from laplaciannb.legacy import LaplacianNB\n"
                "2. Or convert to proper format: from laplaciannb import convert_fingerprints\n"
                "   X = convert_fingerprints(your_sets, n_bits=desired_size)"
            )

        # Check for numpy array with object dtype containing sets
        if hasattr(X, "dtype") and X.dtype == object and len(X) > 0:
            if isinstance(X.flat[0], set):
                raise ValueError(
                    "LEGACY INPUT FORMAT ERROR: You are trying to use the old numpy array of sets format. "
                    "This is no longer supported in the new version. "
                    "\n\nTo fix this:\n"
                    "1. Use the legacy version: from laplaciannb.legacy import LaplacianNB\n"
                    "2. Or convert to proper format: from laplaciannb import convert_fingerprints\n"
                    "   X = convert_fingerprints(your_sets, n_bits=desired_size)"
                )

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors. Binary/boolean features expected.
            Non-zero values are treated as positive bits.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Detect legacy input formats first, before sklearn validation
        self._detect_legacy_input_format(X)

        X, y = self._check_X_y(X, y)

        # Store number of features
        _, self.n_features_in_ = X.shape

        # Encode labels
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_

        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # Handle sample weights
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        # Count raw events from data
        n_classes = Y.shape[1]
        self._init_counters(n_classes, self.n_features_in_)
        self._count(X, Y)

        # Update probabilities
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def predict_log_proba(self, X):
        """Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute classes_.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        jll = self._joint_log_likelihood(X)

        # Normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute classes_.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
