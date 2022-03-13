from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    univ_gauss = UnivariateGaussian().fit(samples)
    print(univ_gauss.mu_, univ_gauss.var_)

    # Question 2 - Empirically showing sample mean is consistent
    distances = []
    num_samples = np.arange(10, 1010, 10)
    for n in num_samples:
        expected = UnivariateGaussian().fit(np.random.choice(samples, n)).mu_
        distances.append(abs(expected - 10))
    # fig = px.scatter(x=num_samples, y=distances)
    fig = px.line(x=num_samples, y=distances, markers=True,
                  labels={'x': 'Number of Samples', 'y': 'Distance to Expected Value (10)'},
                  title='Distances of estimated and true expectation, by number of samples')
    # fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # ordered = np.sort(samples)
    pdfs = univ_gauss.pdf(samples)
    fig = px.scatter(x=samples, y=pdfs,
                     labels={'x': 'Sample Value', 'y': 'Probability Density'},
                     title='Probability Density Function')
    fig.show()


def multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    univariate_gaussian()
    # multivariate_gaussian()
