import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=True, infer_datetime_format=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Year'].astype(str)
    df.drop(df[df['Temp'] < -40].index, inplace=True)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[(df['Country'] == 'Israel')].reset_index()
    fig = px.scatter(df_israel, x='DayOfYear', y='Temp', color='Year',
                     title=f"Daily temperature in Israel",
                     labels={'x': 'Day of Year', 'y': 'Temperature'})
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2\temp_vs_day.jpeg")
    # fig.show()

    grouped = df_israel.groupby('Month')['Temp'].std().reset_index()
    fig = px.bar(grouped, x='Month', y='Temp',
                 title=f"Standard deviation of temperature 1995-2020",
                 labels={'x': 'Month', 'y': 'Temperature STD'})
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2\temp_std.jpeg")
    # fig.show()

    # Question 3 - Exploring differences between countries
    grouped = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    fig = px.line(grouped, x='Month', y='mean', color='Country', error_y='std',
                  title=f"Mean temperature for each country per month",
                  labels={'x': 'Month', 'y': 'Mean Temperature'})
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2\mean_temp_per_country.jpeg")
    # fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_israel[['DayOfYear']], df_israel['Temp'])

    loss = []
    for k in range(1, 11):
        pf = PolynomialFitting(k).fit(train_X.values, train_y.values)
        loss.append(pf.loss(test_X.values, test_y.values))
    print({i+1: l for i, l in enumerate(loss)})
    fig = px.bar(x=range(1, 11), y=loss,
                 title=f"Loss of polynomial fitting of k degree",
                 labels={'x': 'K degree', 'y': 'Loss'})
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2\loss_by_k.jpeg")
    # fig.show()

    # Question 5 - Evaluating fitted model on different countries
    pf = PolynomialFitting(5).fit(df_israel[['DayOfYear']].values, df_israel['Temp'].values)
    loss_by_country = []
    for country in df['Country'].unique():
        df_country = df[df['Country'] == country]
        loss_by_country.append([country, pf.loss(df_country[['DayOfYear']].values, df_country['Temp'].values)])
    loss_by_country = pd.DataFrame(loss_by_country, columns=['Country', 'Loss'])
    fig = px.bar(loss_by_country, x='Country', y='Loss',
                 title=f"Loss of each country by Israel model",
                 labels={'x': 'Country', 'y': 'Loss'})
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex2\loss_by_country.jpeg")
