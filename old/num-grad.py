#!/usr/bin/env python3

###################################
#                                 #
# NUMERICAL TESTS OF ANALITICALLY #
# DERIVED GRADIENTS               #
#                                 #
###################################

import math as M
import matplotlib.pyplot as plt
from functools import partial
import numpy as np


# TEST: tandem loss = 2rho*L_tnd

def g_tandem(L, rho):
    assert(len(rho.shape) == 1)
    assert(len(L.shape) == 2)

    return 2*rho@L

def tnd_loss(L, rho):
    assert(len(rho.shape) == 1)
    assert(len(L.shape) == 2)

    return np.sum(np.outer(rho,rho)*L)

def num_tandem():
    rng = np.random.default_rng()

    res = np.zeros((100, 4))
    for j in range(100):
        L = rng.uniform(0, 1, size=(4,4))
        L = (L+L.T)/(2*L.max())
        rho = rng.uniform(0, 1, size=4)
        rho /= rho.sum()

        g = np.zeros(4)
        p = 0.01
        for i in range(4):
            v = np.zeros(4)
            v[i] = p
            g[i] += (tnd_loss(L, rho+v) - tnd_loss(L, rho-v)) / (2*p)

        res[j] = g - g_tandem(L, rho)
    return res.sum()

# VISUAL INSPECTION: theorem 9 Andres et Al.

def theorem_9(L_tnd, kl, n, delta, lam):
    t1 = L_tnd/(1-lam/2)
    t2 = 2*kl+M.log(2*M.sqrt(n)/delta)
    t3 = lam*(1-(lam/2))*n
    return 4*(t1 + t2/t3)

def opt_theorem_9(L_tnd, kl, n, delta):
    t1 = 2*n*L_tnd
    t2 = 2*kl+M.log(2*M.sqrt(n)/delta)
    return 2/(M.sqrt((t1/t2)+1)+1)

def plot_tangent(ax, f, start = 10**-4, end = 10):
    X = np.linspace(start, end)
    Y = np.vectorize(f)(X)
    G = Y[1:] - Y[:-1]

    ax.plot(X,Y, c = "black")
    ax.plot(X[:-1],G, c = "red")

def plot_opt_theorem_9():
    plot_tangent(plt, partial(theorem_9, 1, 20, 100, 0.05), start = .2, end = 1.8)
    X = np.linspace(.2, 1.8)
    plt.scatter([opt_theorem_9(1, 20, 100, 0.05)], [0], c = "blue")

    plt.grid()
    plt.savefig("fig.png")
    plt.close()

# TEST: KL divergence, grad = 1 + np.log(a/b)

def kl(a, b):
    assert(len(a.shape) == 1)
    assert(len(b.shape) == 1)
    assert(np.count_nonzero(a) == len(a))
    assert(np.count_nonzero(b) == len(b))

    return (a*np.log(a/b)).sum()

def g_kl(a,b):
    assert(len(a.shape) == 1)
    assert(len(b.shape) == 1)
    assert(np.count_nonzero(a) == len(a))
    assert(np.count_nonzero(b) == len(b))

    return 1 + np.log(np.where(a/b>10**-9, a/b, 10**-9))

def num_kl():
    rng = np.random.default_rng()

    it = 10
    n = 5 #rng.integers(2,30)
    res = np.zeros((it, n))
    for j in range(it):
        a = rng.uniform(size=n)+0.02
        a /= a.sum()
        b = rng.uniform(size=n)+0.02
        b /= b.sum()

        g = np.zeros(n)
        p = 0.01
        for i in range(n):
            v = np.zeros(n)
            v[i] = p
            g[i] += (kl(a+v,b) - kl(a-v,b)) / (2*p)

        res[j] = g_kl(a,b) - g
    return res.round(3)

# TEST: Softmax

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def g_softmax(rho):
    # Jacobian matrix of softmax
    Srho = softmax(rho)
    return Srho*np.eye(len(rho)) - np.outer(Srho, Srho)

def num_softmax():
    rng = np.random.default_rng()

    it, n = 100, 30
    res = np.zeros((it, n, n))
    grd = np.zeros((it, n, n))
    for j in range(it):
        rho = rng.uniform(-10, 12, size=n)

        g = np.zeros((n, n))
        p = 0.0001
        for i in range(n):
            v = np.zeros(n)
            v[i] = p
            g[i] += (softmax(rho+v) - softmax(rho-v)) / (2*p)

        res[j] = g
        grd[j] = g_softmax(rho)
    return np.abs(grd-res).sum()

# Test: softmax + L_tnd

def num_softmax_tnd():
    rng = np.random.default_rng()

    it, n = 1, 5
    res = np.zeros((it, n))
    grd = np.zeros((it, n))
    for j in range(it):
        L = rng.uniform(0, 1, size=(n,n))
        L = (L+L.T)/(2*L.max())
        rho = rng.uniform(-2, 4, size=n)

        g = np.zeros(n)
        p = 0.01
        for i in range(n):
            v = np.zeros(n)
            v[i] = p
            g[i] += (tnd_loss(L, softmax(rho+v)) - tnd_loss(L, softmax(rho-v))) / (2*p)

        res[j] = g
        grd[j] = (2*softmax(rho)@L@g_softmax(rho))
    return grd.round(4), res.round(4), (grd-res).sum(), (grd+res).sum()

# TEST: eq. 34, new bound w.o. softmax

def eq_34(L_tnd, L, a, b, n1, n2, delta, lam, pi, rho):
    t1 = lam / (M.exp(lam/2) - 1)

    t2 = (rho * L).sum() / (1 - a/2)
    t3 = kl(rho, pi) + M.log(4*M.sqrt(n1)/delta)
    t4 = t3 / (a * (1 - a/2) * n1)

    t5 = (M.exp(lam) - lam - 1) / lam
    t6 = tnd_loss(L_tnd, rho) / (1 - b/2)
    t7 = 2*kl(rho, pi) + M.log(4*M.sqrt(n2)/delta)
    t8 = t7 / (b * (1 - b/2) * n2)

    return t1 * (t2 + t4 + t5 * (t6 + t8))

def eq_34(L_tnd, L, a, b, n1, n2, delta, lam, pi, rho):
    Lp = (rho * L).sum()
    Lp_tnd = tnd_loss(L_tnd, rho)
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

def g_eq_40(L_tnd, L, a, b, n1, n2, delta, lam, pi, rho):
    # Constants for the L term
    x  = 1/(1 - a/2)

    # Constants for the L_tnd term
    y  = 1/(1 - b/2)

    # Constants for the KL term
    xp = x / (a * n1)
    yp = (2*y) / (b * n2)
    c = (np.exp(lam)-lam-1)/lam

    g_f = x*L + 2*c*y*rho@L_tnd + (xp+c*yp)*g_kl(rho, pi)

    factor = lam / (M.exp(lam/2)-1)

    return factor*g_f

def num_bound_unconstrained():
    rng = np.random.default_rng()

    it = 100
    n = 40
    res = np.zeros((it, n))
    grd = np.zeros((it, n))
    for j in range(it):
        L_tnd = rng.uniform(0, 1, size=(n,n))
        L_tnd = (L_tnd+L_tnd.T)/(2*L_tnd.max())
        L = np.diag(L_tnd)
        rho = rng.uniform(.1, .9, size=n)
        rho /= rho.sum()
        pi = np.ones(n) / n
        n2 = rng.integers(10,500)
        n1 = n2 + rng.integers(10,500)
        a = rng.uniform(0.1,1.9)
        b = rng.uniform(0.1,1.9)
        delta = rng.uniform(0.01,0.2)
        lam = rng.uniform(0.1,0.9)

        g = np.zeros(n)
        p = 0.0001
        for i in range(n):
            v = np.zeros(n)
            v[i] = p
            f = partial(eq_34,L_tnd,L,a,b,n1,n2,delta,lam,pi)
            g[i] += (f(rho+v) - f(rho-v)) / (2*p)

        res[j] = g
        grd[j] = g_eq_40(L_tnd,L,a,b,n1,n2,delta,lam,pi,rho)
    return (res-grd).mean()

# TEST: Gradient w. softmax

def g_bound(L_tnd, L, a, b, n1, n2, delta, lam, pi, rho):
    return g_eq_40(L_tnd, L, a, b, n1, n2, delta, lam, pi, softmax(rho))@g_softmax(rho)

def num_bound():
    rng = np.random.default_rng()

    it = 100
    n = 40
    res = np.zeros((it, n))
    grd = np.zeros((it, n))
    for j in range(it):
        L_tnd = rng.uniform(0, 1, size=(n,n))
        L_tnd = (L_tnd+L_tnd.T)/(2*L_tnd.max())
        L = np.diag(L_tnd)
        rho = rng.uniform(.1, .9, size=n)
        rho /= rho.sum()
        pi = np.ones(n) / n
        n2 = rng.integers(10,500)
        n1 = n2 + rng.integers(10,500)
        a = rng.uniform(0.1,1.9)
        b = rng.uniform(0.1,1.9)
        delta = rng.uniform(0.01,0.2)
        lam = rng.uniform(0.1,0.9)

        g = np.zeros(n)
        p = 0.0001
        for i in range(n):
            v = np.zeros(n)
            v[i] = p
            f = partial(eq_34,L_tnd,L,a,b,n1,n2,delta,lam,pi)
            g[i] += (f(softmax(rho+v)) - f(softmax(rho-v))) / (2*p)

        res[j] = g
        grd[j] = g_bound(L_tnd,L,a,b,n1,n2,delta,lam,pi,rho)
    return (res-grd).mean()

# TEST: Derivative of One Shot Bennett inequality

def bennett_ineq(a,b,lam):
    return (
        lam*a + (M.exp(lam)-lam-1)*b
    ) / (M.exp(lam/2)-1)

def g_bennett_ineq(a,b,lam):
    return (
        (a + b*(M.exp(lam) - 1)) *
        (M.exp(lam/2) - 1) -
        (1/2)*(M.exp(lam/2)) *
        (a*lam + b*(M.exp(lam) - lam - 1))
    ) / (M.exp(lam/2)-1)**2

def num_bennett():
    rng = np.random.default_rng()
    it = 10
    p = 0.001
    res = np.zeros(it)
    grd = np.zeros(it)
    for j in range(it):
        a = rng.uniform(0.1,1.9)
        b = rng.uniform(0.1,1.9)
        lam = rng.uniform(0.1,0.9)
        f = partial(bennett_ineq,a,b)
        res[j] = (f(lam+p) - f(lam-p)) / (2*p)
        grd[j] = g_bennett_ineq(a,b,lam)
    return np.array((res.round(1), grd.round(1)))

print("\n".join(str(x) for x in num_bennett()))
