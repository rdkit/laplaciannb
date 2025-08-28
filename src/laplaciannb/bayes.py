from functools import reduce

import numpy as np
from scipy.special import logsumexp
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data


class LaplacianNB(_BaseDiscreteNB):
    """Naive Bayes classifier for laplacian modified models.

    DEPRECATED: This is the legacy LaplacianNB implementation.
    Please use the new sklearn-compatible version instead:

        from laplaciannb import LaplacianNB  # New version (recommended)

    The new implementation offers:
    - Full sklearn compatibility (pipelines, cross-validation, grid search)
    - Memory-efficient sparse matrix support
    - Better error handling and validation
    - Consistent API with other sklearn estimators
    - Enhanced fingerprint utility functions

    This legacy version will be removed in a future release.

    Like BernoulliNB, this classifier is suitable for binary/boolean data. The
    difference is that while BernoulliNB processes all features, while
    laplacian modified approach is using only positive bits.
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
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
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of 1' bits encountered for each (class, feature)
        during fitting.
    feature_all_ : total number of features encountered.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of 1' bit features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.
        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    See Also
    --------
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : The Complement Naive Bayes classifier
        described in Rennie et al. (2003).
    GaussianNB : Gaussian Naive Bayes (GaussianNB).
    MultinomialNB : Naive Bayes classifier for multinomial models.
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
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> arr = rng.randint(2, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> Xlist = []
    >>> for i in arr:
    >>>     Xlist.append(set(i.nonzero()[0]))
    >>> X = np.array(Xlist)
    >>> from bayes.LaplacianNB import LaplacianNB
    >>> clf = LaplacianNB()
    >>> clf.fit(X, Y)
    LaplacianNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    # see https://github.com/scikit-learn/scikit-learn/pull/22269 for an explanation

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.force_alpha = force_alpha

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = validate_data(self, X, reset=False, accept_sparse=["csr"])
        return X

    def _check_X_y(self, X, y, reset=True):
        X, y = validate_data(self, X, y, reset=reset, accept_sparse=["csr"])
        return X, y

    def _sum_sets(self, set_list):
        def reducer(accumulator, element):
            for key in element:
                accumulator[key] = accumulator.get(key, 0) + 1
            return accumulator

        return reduce(reducer, set_list, {})

    # Even more memory-efficient version that avoids creating huge matrices
    def _count_feature_count(self, X_sparse, Y):
        """Most efficient version that handles 2^32 feature space gracefully."""
        from collections import defaultdict

        # Get active features to avoid working with full 2^32 space
        X_coo = X_sparse.tocoo()

        # 1. Total feature counts
        all_feature_counts = defaultdict(int)
        for col_idx, data_val in zip(X_coo.col, X_coo.data):
            all_feature_counts[col_idx] += data_val
        all_feature_counts = dict(sorted(all_feature_counts.items()))

        # 2. Class-specific counts by iterating samples
        class_feature_counts = [defaultdict(int) for _ in range(len(self.classes_))]
        feature_sum = np.zeros(len(self.classes_))

        # Group elements by sample (row)
        sample_features = defaultdict(list)
        for row_idx, col_idx, data_val in zip(X_coo.row, X_coo.col, X_coo.data):
            sample_features[row_idx].append((col_idx, data_val))

        # Count features per class
        for sample_idx, features in sample_features.items():
            # Find which classes this sample belongs to
            sample_classes = Y[sample_idx].nonzero()[0]

            for class_idx in sample_classes:
                class_weight = Y[sample_idx, class_idx]
                for col_idx, data_val in features:
                    weighted_count = data_val * class_weight
                    class_feature_counts[class_idx][col_idx] += weighted_count
                    feature_sum[class_idx] += weighted_count

        # Convert to sorted dictionaries
        class_feature_counts = [dict(sorted(d.items())) for d in class_feature_counts]

        return all_feature_counts, feature_sum, class_feature_counts

    def _init_counters(self, n_classes):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        # self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        (
            self.feature_count_all_dict_,
            self.feature_count_,
            self.feature_count_dict_,
        ) = self._count_feature_count(X, Y)
        self.feature_all_ = sum(self.feature_count_)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        dictvectorizer = DictVectorizer(sparse=False)
        total = dictvectorizer.fit_transform(self.feature_count_all_dict_)
        classc = dictvectorizer.fit_transform(self.feature_count_dict_)
        self.feature_names_ = [int(i) for i in dictvectorizer.get_feature_names_out()]
        self.feature_names_ = dict(zip(self.feature_names_, range(len(self.feature_names_))))
        prior = self.feature_count_ / self.feature_all_
        self.feature_prob_ = (classc + alpha) / (np.outer(prior, total) + alpha)
        self.feature_log_prob_ = np.log(self.feature_prob_).astype("float32")

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_features = self.feature_log_prob_.shape[1]

        new_X = np.zeros([X.shape[0], n_features], dtype=bool)

        for i, row in enumerate(X):
            # Handle sparse matrix row
            row_coo = row.tocoo()
            for col_idx in row_coo.col:
                if self.feature_names_.get(col_idx) is not None:
                    new_X[i, self.feature_names_[col_idx]] = 1
        jll = np.dot(new_X, self.feature_log_prob_.T)
        return jll

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._check_X_y(X, y)

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
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
