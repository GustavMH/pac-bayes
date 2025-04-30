#
# Implementation of the lambda bound and optimization procedure.
#
# Based on paper:
# [Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin.
#  A strongly quasiconvex385PAC-Bayesian bound. InAlgorithmic Learning Theory (ALT), 2017]
#
import numpy as np

from math import ceil, log, sqrt, exp


def uniform_distribution(m):
    return (1.0 / m) * np.ones((m,))


def kl(rho, pi):
    m = pi.shape[0]
    assert rho.shape[0] == m
    kl = 0.0
    for h in range(m):
        kl += rho[h] * log(rho[h] / pi[h]) if rho[h] > 10**-12 else 0
    return kl


def softmax(dist):
    dexp = np.exp(dist)
    return dexp / np.sum(dexp, axis=0)


def iRProp(grad, func, x0,
        max_iterations=1000,
        eps=10**-9,
        step_init=0.1,
        step_min=10^-20,
        step_max=10**5,
        inc_fact=1.1,
        dec_fact=0.5
           ):
    """Resilient backpropagation"""
    n    = x0.shape[0]
    dx   = np.zeros((max_iterations, n))
    x    = np.zeros((max_iterations, n))
    x[0] = x0
    step = np.ones(n)*step_init

    fx   = np.ones(max_iterations)
    fx[0]= func(x[0])
    tb   = 0

    t = 1
    while t < max_iterations:
        delta = fx[t-1]-fx[t-2] if t>1 else -1.0
        if t-tb > 10:
            break
        dx[t] = grad(x[t-1])

        # Update set size
        det = np.multiply(dx[t], dx[t-1])
        # Increase where det>0
        step[det>0] = step[det>0]*inc_fact
        # Decrease where det<0
        step[det<0] = step[det<0]*dec_fact
        # Upper/lower bound by min/max
        step        = step.clip(step_min, step_max)

        # Update w
        # If det >= 0, same as RProp
        x[t][det>=0] = x[t-1][det>=0] - np.multiply(np.sign(dx[t]), step)[det>=0]
        # If func(x[t-1])>func(x[t-2]) set x[t] to x[t-2] where det<0 (only happens if t>1, as det==0 for t=1)
        if delta>0:
            x[t][det<0] = x[t-2][det<0]
        else:
            x[t][det<0] = x[t-1][det<0] - np.multiply(np.sign(dx[t]), step)[det<0]
        # Reset dx[t] = 0 where det<0
        dx[t][det<0] = 0

        # Compute func value
        fx[t] = func(x[t])
        if fx[t] < fx[tb]:
            tb = t

        t += 1
    return x[tb]


def optimizeTND(tandem_risks, n, delta=0.05, max_iterations = 100, eps = 10**-9):
    m   = tandem_risks.shape[0]
    rho = uniform_distribution(m)
    pi  = uniform_distribution(m)

    # Some helper functions
    def _tndr(rho): # Compute tandem risk from tandem risk matrix and rho
        return np.average(np.average(tandem_risks, weights=rho, axis=0), weights=rho)

    def _bound(rho, lam=None): # Compute value of bound (also optimize lambda if None)
        rho  = softmax(rho)
        tndr = _tndr(rho)
        KL   = kl(rho,pi)
        if lam is None:
            lam = 2.0 / (sqrt((2.0*n*tndr)/(2.0*KL+log(2.0*sqrt(n)/delta)) + 1) + 1)
        bound = tndr / (1.0 - lam/2.0) + (2.0*KL+log(2.0*sqrt(n)/delta))/(lam*(1.0-lam/2.0)*n)
        return (bound, lam)

    # 1st order methods
    def _gradient(rho, lam): # Gradient (using rho = softmax(rho'))
        Srho = softmax(rho)
        # D_jS_i = S_i(1[i==j]-S_j)
        Smat = -np.outer(Srho, Srho)
        np.fill_diagonal(Smat, np.multiply(Srho,1.0-Srho))
        return np.dot(2*(np.dot(tandem_risks,Srho)+1.0/(lam*n)*(1+np.log(Srho/pi))),Smat)

    def _optRho(rho, lam):
        return iRProp(lambda x: _gradient(x,lam), lambda x: _bound(x,lam)[0], rho,\
                eps=eps, max_iterations=max_iterations)

    b, lam   = _bound(rho)
    bp       = b+1
    while abs(b-bp) > eps:
        bp = b
        # Optimize rho
        nrho = _optRho(rho, lam)
        b, nlam = _bound(nrho)
        if b > bp:
            b = bp
            break
        rho, lam = nrho, nlam

    return (min(1.0,4*b), softmax(rho), lam)


def tandem_risks(predictions, target):
    """
    given a (model, predictions) for all of list target,
    produce a n x n array of risks, where n = len(target)
    """
    n = len(predictions)
    risks = np.zeros((n,n))

    for i, p_a in enumerate(predictions):
        for j, p_b in enumerate(predictions):
            risks[i,j] += np.sum(
                np.logical_and(
                    p_a != target,
                    p_b != target
                )
            )

    print(risks.sum() / n)
    return risks / n


def optimize_rho(predictions, target):
    risks = tandem_risks(predictions, target)
    (bound, rho, lam) = optimizeTND(risks, len(target))
    return (bound, rho, lam)
