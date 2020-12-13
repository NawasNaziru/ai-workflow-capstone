import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Create an instance of a Gaussian Process model

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit the AAVAIL training data

gpr.fit(X, y)

# Make the prediction on the test data

y_predicted, sigma = gpr.predict(x, return_std=True)
