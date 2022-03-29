from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    print(df.shape)
    columns_to_check = {'id': 'all', 'date': 'nan', 'price': 'all', 'bedrooms': 'all',
                        'bathrooms': 'negative_nan', 'sqft_living': 'all', 'sqft_lot': 'all',
                        'floors': 'all', 'waterfront': 'nan', 'view': 'nan', 'condition': 'all',
                        'grade': 'all', 'sqft_above': 'all', 'sqft_basement': 'negative_nan', 'yr_built': 'all',
                        'yr_renovated': 'negative_nan', 'zipcode': 'all', 'lat': 'zero_nan', 'long': 'zero_nan',
                        'sqft_living15': 'all', 'sqft_lot15': 'all'}
    drop_functions = {'all': lambda x, col: x.drop(x[(x[col] <= 0) | (x[col].isna())].index),
                      'negative_nan': lambda x, col: x.drop(x[(x[col] < 0) | (x[col].isna())].index),
                      'zero_nan': lambda x, col: x.drop(x[(x[col] == 0) | (x[col].isna())].index),
                      'nan': lambda x, col: x.drop(x[(x[col].isna())].index)}
    for column in columns_to_check:
        df = drop_functions[columns_to_check[column]](df, column)
    print(df.shape)
    df = df.drop(df[df['sqft_living'] > df['sqft_lot']].index)
    print(df.shape)
    df['is_renovated'] = df['yr_renovated']
    df.loc[df['is_renovated'] != 0, 'is_renovated'] = 1

    features = ['bedrooms', 'bathrooms', 'sqft_living',
                'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'sqft_living15', 'sqft_lot15', 'is_renovated']

    return df[features], df['price']


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    correlations = X.apply(lambda x: np.cov(x, y)[0, 1] / (x.std() * y.std()), axis=0)

    highest_feature = correlations.max()
    highest_feature_col = correlations.index[correlations.argmax()]
    fig = px.scatter(x=X[highest_feature_col], y=y,
                     title=f"Plot of {highest_feature_col} and price, with correlation {highest_feature}",
                     labels={'x': highest_feature_col, 'y': 'price'})
    fig.write_image(f"{output_path}/highest_corr_feature.jpeg")
    # fig.show()

    lowest_feature = correlations.min()
    lowest_feature_col = correlations.index[correlations.argmin()]
    fig = px.scatter(x=X[lowest_feature_col], y=y,
                     title=f"Plot of {lowest_feature_col} and price, with correlation {lowest_feature}",
                     labels={'x': lowest_feature_col, 'y': 'price'})
    fig.write_image(f"{output_path}/lowest_corr_feature.jpeg")
    # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2")

    # Question 3 - Split samples into training- and testing sets.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    result = []
    for i in range(10):
        for p in np.arange(0.1, 1.01, .01):
            train_sample = train_X.sample(frac=p, replace=False)
            lr = LinearRegression()
            lr.fit(train_sample.values, train_y.loc[train_sample.index])
            result.append([p, lr._loss(test_X.values, test_y.values)])
    df = pd.DataFrame(result, columns=['Percentage', 'Loss']).groupby('Percentage')['Loss'].agg(['mean', 'std']).reset_index()
    percentages = df['Percentage']
    loss = df['mean']
    loss_std = df['std']
    fig = go.Figure([go.Scatter(x=percentages, y=loss, mode="markers+lines", name="Loss vs Percentage of data"),
                    go.Scatter(x=percentages, y=loss-2*loss_std, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                    go.Scatter(x=percentages, y=loss+2*loss_std, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(title=r"Loss of prediction by Percentage of data trained",
                                     xaxis={"title": "Percentage of trained data"},
                                     yaxis={"title": "Loss of predicted test (MSE)"}))
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2\loss.jpeg")


