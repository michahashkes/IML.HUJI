from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import mean_square_error
import numpy as np


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]
            lam_matrix = self.lam_ * np.identity(n_features + 1)
            lam_matrix[0, 0] = 0
        else:
            lam_matrix = self.lam_ * np.identity(n_features)
        self.coefs_ = np.linalg.inv((X.T @ X) + lam_matrix) @ X.T @ y

        # if self.include_intercept_:
        #     X = np.c_[np.ones(len(X)), X]
        #
        # U, sigma, V = np.linalg.svd(X)
        # sigma = sigma / (sigma ** 2 + self.lam_)
        # s = np.zeros((V.shape[1], U.shape[1]))
        # np.fill_diagonal(s, sigma)
        #
        # self.coefs_ = V @ s @ U.T @ y

        # n_samples, n_features = X.shape[0], X.shape[1]
        # lam_matrix = np.sqrt(self.lam_) * np.identity(n_features)
        # X_lam = np.vstack((X, lam_matrix))
        # y_lam = np.concatenate((y, np.zeros(n_features)))
        # if self.include_intercept_:
        #     X_lam = np.c_[np.ones(n_samples + n_features), X_lam]
        # self.coefs_ = np.linalg.pinv(X_lam) @ y_lam

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))
