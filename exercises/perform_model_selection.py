from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(-1.2, 2, n_samples)
    noise_data = np.random.normal(0, noise, n_samples)
    y_no_noise = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = y_no_noise + noise_data
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2 / 3)
    fig = go.Figure([go.Scatter(x=np.sort(X), y=y_no_noise[np.argsort(X)], mode='lines', name='No Noise'),
                     go.Scatter(x=train_X.iloc[:, 0], y=train_y, mode='markers', name='Train Set'),
                     go.Scatter(x=test_X.iloc[:, 0], y=test_y, mode='markers', name='Test Set')])
    fig.update_layout(title_text=f"Train set vs. test set, {noise} noise, {n_samples} samples", title_x=0.5,
                      xaxis_title='x', yaxis_title='y')
    fig.write_image(fr"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex5\polynomial_sets_{noise}_{n_samples}.jpeg")
    # fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    scores = np.array([cross_validate(PolynomialFitting(d), train_X.values, train_y.values, mean_square_error, 5)
                       for d in range(11)])
    fig = go.Figure([go.Scatter(x=np.arange(0, 11), y=scores[:, 0], mode='lines', name='Training Errors'),
                     go.Scatter(x=np.arange(0, 11), y=scores[:, 1], mode='lines', name='Validation Errors')])
    fig.update_layout(title_text=f"Training and validation errors for polynomial fitting, {noise} noise, {n_samples} samples",
                      title_x=0.5, xaxis_title='Degree for polynomial fitting', yaxis_title='Loss error')
    fig.write_image(fr"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex5\cross_validation_{noise}_{n_samples}.jpeg")
    # fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(scores[:, 1])
    pf = PolynomialFitting(best_k)
    pf.fit(train_X.values, train_y.values)
    loss = pf.loss(test_X.values, test_y.values)
    print(f"{noise} noise, {n_samples} samples: The lowest validation error is {round(scores[best_k, 1], 2)},"
          f"the best degree is {best_k}, and the test error is {round(loss, 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), n_samples / X.shape[0])

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.001, 2, n_evaluations)

    ridge_scores = np.array([cross_validate(RidgeRegression(lam), train_X.values, train_y.values, mean_square_error, 5)
                       for lam in lambdas])
    fig = go.Figure([go.Scatter(x=lambdas, y=ridge_scores[:, 0], mode='lines', name='Training Errors'),
                     go.Scatter(x=lambdas, y=ridge_scores[:, 1], mode='lines', name='Validation Errors')])
    fig.update_layout(title_text=f"Training and validation errors for Ridge Regression",
                      title_x=0.5, xaxis_title='lambda values', yaxis_title='Loss error')
    fig.write_image(fr"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex5\cross_validation_ridge.jpeg")
    fig.show()

    lasso_scores = np.array([cross_validate(Lasso(lam), train_X.values, train_y.values, mean_square_error, 5)
                             for lam in lambdas])
    fig = go.Figure([go.Scatter(x=lambdas, y=lasso_scores[:, 0], mode='lines', name='Training Errors'),
                     go.Scatter(x=lambdas, y=lasso_scores[:, 1], mode='lines', name='Validation Errors')])
    fig.update_layout(title_text=f"Training and validation errors for Lasso Regression",
                      title_x=0.5, xaxis_title='lambda values', yaxis_title='Loss error')
    fig.write_image(fr"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex5\cross_validation_loss.jpeg")
    fig.show()

    # rr = RidgeRegression(2)
    # rr.fit(test_X.values, test_y.values)
    # y_pred1 = rr.predict(train_X)
    #
    # from sklearn.linear_model import Ridge
    # lr = Ridge(2)
    # lr.fit(test_X.values, test_y.values)
    # y_pred2 = lr.predict(train_X)
    #
    # print()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_idx = np.argmin(ridge_scores[:, 1])
    best_ridge_lambda, best_ridge_loss = lambdas[best_ridge_idx], ridge_scores[best_ridge_idx, 1]
    rr = RidgeRegression(best_ridge_lambda)
    rr.fit(train_X.values, train_y.values)
    ridge_loss = rr.loss(test_X.values, test_y.values)
    print(f"Ridge Regression: The lowest validation error is {round(best_ridge_loss, 2)},"
          f"the best lambda is {round(best_ridge_lambda, 2)}, and the test error is {round(ridge_loss, 2)}")

    best_lasso_idx = np.argmin(lasso_scores[:, 1])
    best_lasso_lambda, best_lasso_loss = lambdas[best_lasso_idx], lasso_scores[best_lasso_idx, 1]
    lr = Lasso(best_lasso_lambda)
    lr.fit(train_X.values, train_y.values)
    lasso_loss = mean_square_error(lr.predict(test_X.values), test_y.values)
    print(f"Lasso Regression: The lowest validation error is {round(best_lasso_loss, 2)},"
          f"the best lambda is {round(best_lasso_lambda, 2)}, and the test error is {round(lasso_loss, 2)}")

    lr = LinearRegression()
    lr.fit(train_X.values, train_y.values)
    linear_loss = lr.loss(test_X.values, test_y.values)
    print(f"Linear Regression: The test error is {round(linear_loss, 2)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    print()
    select_regularization_parameter()
