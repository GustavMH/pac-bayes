#!/usr/bin/env python3
import numpy as np
from sklearn.utils import check_random_state

from mvb import NeuralNetworkPostTrainClassifier as EnsembleClassifier

def optimize(dataset, m, smode, opt, reps, inpath, ensemble_path, write_files, indices=None, test_risk_indices=None, test_bound_indices=None, test_pred_file_name='test_predictions.pkl'):
    results = []

    X,Y = dataset
    trainX, valX, testX = X
    trainY, valY, testY = Y

    first_rho = None

    C = np.unique(trainY).shape[0]
    n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)

    rf = EnsembleClassifier(m, ensemble_path=ensemble_path, indices=indices, test_risk_indices=test_risk_indices, test_bound_indices=test_bound_indices, test_pred_file_name=test_pred_file_name)

    rhos = []
    rf.fit(trainX, trainY, valX, valY)

    _, mv_risk = rf.predict(testX,testY)
    stats = rf.stats()
    bounds, stats = rf.bounds(stats=stats)
    res_unf = (mv_risk, stats, bounds, -1, -1)

    # Optimize Lambda
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds, stats = rf.bounds(stats=stats)
    res_lam = (mv_risk, stats, bounds, bl, -1)
    rhos.append(rho)

    # Optimize TND
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':opt})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds, stats = rf.bounds(stats=stats)
    res_mv2 = (mv_risk, stats, bounds, bl, -1)
    rhos.append(rho)

    # Optimize DIS if binary
    if(C==2):
        (_, rho, bl, bg) = rf.optimize_rho('DIS', options={'optimizer':opt})
        _, mv_risk = rf.predict(testX,testY)
        stats = rf.aggregate_stats(stats)
        bounds, stats = rf.bounds(stats=stats)
        res_mv2u = (mv_risk, stats, bounds, bl, bg)
        rhos.append(rho)
    else:
        res_mv2u = (-1.0, dict(), dict(), -1, -1)
        rhos.append(-np.ones((m,)))

    return first_rho, results
