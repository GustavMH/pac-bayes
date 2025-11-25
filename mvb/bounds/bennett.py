#!/usr/bin/env python3
from ..util import kl, softmax, iRProp
import math as M
import numpy as np

def assert_shape(arr, n):
    if not len(arr.shape) == n:
        raise AssertionError(f"Expected dimension {n} but got {len(arr.shape)} for shape {arr.shape}")

def bennett(L_tnd, L, a, b, n1, n2, lam, pi, rho, delta = 0.05):
    assert_shape(L_tnd, 2)
    assert_shape(L, 1)
    assert_shape(pi, 1)
    assert_shape(rho, 1)

    Lp = (rho * L).sum()
    Lp_tnd = np.mean(L_tnd*np.outer(rho,rho))
    KL = kl(rho, pi)

    return (lam / (M.exp(lam/2) - 1))*(
        (1 / (1 - a/2)) * (
            Lp +
            (KL + M.log(4 * M.sqrt(n1) / delta)) /
            (a*n1)
        ) +
        ((M.exp(lam) - lam - 1) / lam) *
        (1 / (1 - b/2)) * (
            Lp_tnd +
            (2*KL + M.log(4 * M.sqrt(n2) / delta)) /
            (b*n2)
        )
    )


def optimizeBennett(L_tnd, L, a, b, n1, n2, lam, pi, delta = 0.05, eps=10**-9, max_iterations=1000):
    def opt_alpha(L, KL, n1, delta):
        return 2 / (
            M.sqrt(
                (2*n1*L) /
                (KL + M.log(4*M.sqrt(n1)/delta)) + 1
            ) + 1
        )

    def opt_beta(L_tnd, KL, n2, delta):
        return 2 / (
            M.sqrt(
                (2*n2*L_tnd) /
                (2*KL + M.log(4*M.sqrt(n2)/delta)) + 1
            ) + 1
        )

    def opt_lambda(L_tnd, L, a, b, n1, n2, delta, pi, rho):
        f = np.vectorize(lambda lam: bennett(L_tnd, L, a, b, n1, n2, lam, pi, rho, delta))
        x = np.linspace(10**-9,10,1000)
        return x[np.argmin(f(x))]

    def g_unconstrained(L_tnd, L, a, b, n1, n2, lam, pi, rho, delta):
        def g_kl(a,b):
            return 1 + np.log(np.where(a/b>10**-9, a/b, 10**-9))

        x  = 1/(1 - a/2)
        y  = 1/(1 - b/2)

        xp = x / (a * n1)
        yp = (2*y) / (b * n2)
        c = (np.exp(lam)-lam-1)/lam

        g_f = x*L + 2*c*y*rho@L_tnd + (xp+c*yp)*g_kl(rho, pi)

        factor = lam / (M.exp(lam/2)-1)

        return factor*g_f

    def gradient(L_tnd, L, a, b, n1, n2, lam, pi, rho, delta):
        def g_softmax(rho):
            # Jacobian matrix of softmax
            Srho = softmax(rho)
            return Srho*np.eye(len(rho)) - np.outer(Srho, Srho)

        return g_unconstrained(L_tnd, L, a, b, n1, n2, lam, pi, softmax(rho), delta)@g_softmax(rho)

    a_ = np.zeros(100)
    b_ = np.zeros(100)
    l_ = np.zeros(100)
    r_ = np.zeros((100,len(pi)))
    a_[0] = a
    b_[0] = b
    l_[0] = lam
    r_[0] = pi
    for i in range(1,100):
        Srho = softmax(r_[i])
        Lp = (Srho * L).sum()
        Lp_tnd = np.mean(L_tnd*np.outer(Srho,Srho))
        KL = kl(Srho, pi)

        a_[i] = opt_alpha(Lp, KL, n1, delta)
        b_[i] = opt_beta(Lp_tnd, KL, n2, delta)
        l_[i] = opt_lambda(L_tnd, L, a_[i], b_[i], n1, n2, delta, pi, Srho)
        r_[i] = iRProp(
            lambda rho: gradient(L_tnd, L, a_[i], b_[i], n1, n2, l_[i], pi, rho, delta),
            lambda rho: bennett(L_tnd, L, a_[i], b_[i], n1, n2, l_[i], pi, rho, delta),
            r_[i],
            eps=eps,
            max_iterations=max_iterations
        )

        #print(softmax(r_[i]), Lp, Lp_tnd, KL)
        if np.abs(r_[i-1] - r_[i]).sum() < eps:
            break

    return softmax(r_[i]), a_[i], b_[i], l_[i]
