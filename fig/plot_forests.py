#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

a = np.load("random_forest.npz", allow_pickle=True)

def optimize_rho(params: dict = {}):
    L = params["gibbs_risks"]
    n1 = params["n1"]
    L_tnd = params["tandem_risks"]
    n2 = params["n2"]
    pi = np.ones(len(L)) / len(L)
    rho = None
    extra = {}

    def tandem_risk(rho):
        return (L_tnd*np.outer(rho,rho)).sum()

    def gibbs_risk(rho):
        return (L*rho).sum()

    (rho, alpha, beta, lam) = optimizeBennett(L_tnd, L, 1, 1, n1, n2, 1, pi)
    a_term, b_term = bennett_terms(L_tnd, L, alpha, beta, n1, n2, lam, pi, rho)

    return a_term, b_term, lam

def get_points(exp):
    return np.array([
        optimize_rho(p)
        for p in exp
    ])

from mvb.util import kl, softmax, iRProp
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
    Lp_tnd = np.sum(L_tnd*np.outer(rho,rho))
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

def bennett_terms(L_tnd, L, a, b, n1, n2, lam, pi, rho, delta = 0.05):
    assert_shape(L_tnd, 2)
    assert_shape(L, 1)
    assert_shape(pi, 1)
    assert_shape(rho, 1)

    Lp = (rho * L).sum()
    Lp_tnd = np.sum(L_tnd*np.outer(rho,rho))
    KL = kl(rho, pi)

    return (
        (1 / (1 - a/2)) * (
            Lp +
            (KL + M.log(4 * M.sqrt(n1) / delta)) /
            (a*n1)
        ),
        (1 / (1 - b/2)) * (
            Lp_tnd +
            (2*KL + M.log(4 * M.sqrt(n2) / delta)) /
            (b*n2)
        )
    )

def bennett_kl(L_tnd, L, a, b, n1, n2, lam, pi, rho, delta = 0.05):
    assert_shape(L_tnd, 2)
    assert_shape(L, 1)
    assert_shape(pi, 1)
    assert_shape(rho, 1)

    # risks drawn according to pi
    Lp = (pi * L).sum()
    Lp_tnd = np.mean(L_tnd*np.outer(pi,pi))
    KL = kl(rho, pi)

    return (
        (lam / (M.exp(lam/2) - 1)) * (
            KL + M.log(M.exp(Lp)/delta)
        ) +
        ((M.exp(lam) - lam - 1) / lam) * (
            KL + M.log(M.exp(Lp_tnd)/delta)
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def plot_fo_snd():
    fig, ax = plt.subplots(1, 2, figsize=(5.1, 1.5), width_ratios=[1, 3])

    xmax = 0
    ymax = 0
    for key in [
        "Mushroom",
        "Sensorless",
        "SVMGuide1",
        "Phishing",
        "Splice",
        "w1a",
        "Cod-RNA",
        "Adult",
        "Connect-4",
        "Shuttle",
        "Pendigits",
        "Letter",
        "SatImage",
        "MNIST",
        "Fashion-MNIST",
    ]:
        points = get_points(a[key])
        xmax = max(xmax, np.max(points.T[0]))
        ymax = max(ymax, np.max(points.T[1]))
        ax[0].plot(*points.T, linestyle="none", marker="o", label=key)
        ax[1].plot(*points.T, linestyle="none", marker="o", label=key)

    ax[1].set_xlabel("First moment")
    ax[1].set_ylabel("Second moment")
    ax[1].plot([0, 1.1 * xmax], [0, 0.55 * xmax], linestyle="dotted", c="grey")
    ax[1].set_xlim([-0.05, 1.1 * xmax])
    ax[1].set_ylim([-0.1, 1.1 * ymax])
    ax[1].yaxis.set_label_position("right")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[1].plot([-0.05,0],[1.1*ymax,0.03], linewidth=2, c="black")
    ax[1].plot([-0.05,0],[-0.1,0], linewidth=2, c="black")

    rect = Rectangle(
        (0, 0), 0.03, 0.03, linewidth=2, edgecolor="black", facecolor="none", zorder=10
    )
    ax[1].add_patch(rect)
    rect = Rectangle(
        (0, 0), 0.04, 0.03, linewidth=4, edgecolor="black", facecolor="none", zorder=10
    )
    ax[0].add_patch(rect)

    ax[0].annotate(
        "Sensorless",
        xy=(0.027, 0.008),
        xytext=(0.007, 0.001),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=3, headlength=3),
    )

    ax[0].set_xlim(0, 0.03)
    ax[0].set_ylim(0, 0.03)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].plot([0, 1.2 * xmax], [0, 0.6 * xmax], linestyle="dotted", c="grey")

    # from matplotlib.patches import Rectangle
    # from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    # from PIL import Image

    # img = Image.open('fig/zoom_in_icon.png').convert("RGBA")
    # imagebox = OffsetImage(img, zoom=0.04)
    # ab = AnnotationBbox(imagebox, (0.001, 0.029), frameon=False, box_alignment=(0,1))
    # ax[0].add_artist(ab)

    fig.subplots_adjust(wspace=0)

    plt.savefig("fig/sensorless_gap.pdf", dpi=300)
    plt.close()
