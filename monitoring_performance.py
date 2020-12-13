#!/usr/bin/env python
"""
Performance monitoring
"""

import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance

from cslib import fetch_ts
def get_latest_train_data(country):
    """
    load the data used in the latest training
    """

    data_dir = os.path.join("data","cs_train","data")
    ts_data = fetch_ts(data_dir)
        
    for item,df in ts_data.items():
        if(item==country):
            return df
def monitor_performance(data):

    
    pipe = Pipeline(steps=[('pca', PCA(2)),
                            ('clf', EllipticEnvelope(random_state=0,contamination=0.01))])
    data_piped = pipe.fit(data)
    
    bs_samples = 1000
    outliers_X = np.zeros(bs_samples)
    wasserstein_X = np.zeros(bs_samples)
    wasserstein_y = np.zeros(bs_samples)
    
    for b in range(bs_samples):
        n_samples = int(np.round(0.80 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,replace=True).astype(int)
        y_bs=y[subset_indices]
        X_bs=data_piped[subset_indices,:]
    
        test1 = data_piped.predict(X_bs)
        wasserstein_X[b] = wasserstein_distance(data_piped.flatten(),X_bs.flatten())
        wasserstein_y[b] = wasserstein_distance(y,y_bs.flatten())
        outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))

    ## determine thresholds as a function of the confidence intervals
    outliers_X.sort()
    outlier_X_threshold = outliers_X[int(0.975*bs_samples)] + outliers_X[int(0.025*bs_samples)]

    wasserstein_X.sort()
    wasserstein_X_threshold = wasserstein_X[int(0.975*bs_samples)] + wasserstein_X[int(0.025*bs_samples)]

    wasserstein_y.sort()
    wasserstein_y_threshold = wasserstein_y[int(0.975*bs_samples)] + wasserstein_y[int(0.025*bs_samples)]
    
    to_return = {"outlier_X": np.round(outlier_X_threshold,1),
                 "wasserstein_X":np.round(wasserstein_X_threshold,2),
                 "wasserstein_y":np.round(wasserstein_y_threshold,2),
                 "preprocessor":preprocessor,
                 "clf_X":pipe,
                 "X_source": data_piped,
                 "y_source":y,
                 "latest_X":X,
                 "latest_y":y}
    return(to_return)


if __name__ == "__main__":

    ## get latest training data
    data = get_latest_train_data("netherlands")
    print(monitor_performance(data))
