from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    univ_gauss = UnivariateGaussian().fit(samples)
    print(univ_gauss.mu_, univ_gauss.var_)

    # Question 2 - Empirically showing sample mean is consistent
    distances = []
    num_samples = np.arange(10, 1010, 10)
    for n in num_samples:
        expected = UnivariateGaussian().fit(samples[:n]).mu_
        distances.append(abs(expected - 10))
    fig = px.line(x=num_samples, y=distances, markers=True,
                  labels={'x': 'Number of Samples', 'y': 'Distance to Expected Value (10)'},
                  title='Distances of estimated and true expectation, by number of samples')
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex1\distances.jpeg")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = univ_gauss.pdf(samples)
    fig = px.scatter(x=samples, y=pdfs,
                     labels={'x': 'Sample Value', 'y': 'Probability Density'},
                     title='Probability Density Function')
    # fig.show()
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex1\pdf_univariate.jpeg")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                   [0.2, 2, 0, 0],
                   [0, 0, 1, 0],
                   [0.5, 0, 0, 1]]).T
    samples = np.random.multivariate_normal(mu, cov, 1000)
    multi_gauss = MultivariateGaussian().fit(samples)
    print(multi_gauss.mu_)
    print(multi_gauss.cov_)
    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_results = []
    for f1_val in f1:
        row = []
        for f3_val in f3:
            u = np.array([f1_val, 0, f3_val, 0])
            row.append(MultivariateGaussian.log_likelihood(u, cov, samples))
        log_results.append(row)
    fig = px.imshow(log_results, y=f1, x=f3,
                    labels={'x': 'f3 values', 'y': 'f1 value'},
                    title='Multivariate Log Likelihood Values')
    # fig.show()
    fig.write_image(r"C:\Users\Micha\Documents\HUJI\IML\Exercises\Ex1\multivariate_log_likelihood.jpeg")

    # Question 6 - Maximum likelihood
    log_results = np.array(log_results)
    idx = np.unravel_index(np.argmax(log_results, axis=None), log_results.shape)
    print(f1[idx[0]], f3[idx[1]])

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
