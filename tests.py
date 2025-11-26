#!/usr/bin/env python3

def test_oob_tandem_risk():
    calc_res, calc_n2 = oob_tandem_risks(
        np.ones((3,100)),
        np.block([
            [np.arange(100)],
            [np.arange(100)+30],
            [np.arange(100)+60]
        ]),
        np.block([
            np.zeros(100),
            np.ones(100)
        ])
    )
    res = np.array([[1.        , 1.        , 1.        ],
                    [1.        , 0.7       , 0.57142857],
                    [1.        , 0.57142857, 0.4       ]])
    n2 = 40
    return (calc_res-res).sum() , calc_n2 == n2

def reference_oob_tandem_risks(preds, targs):
    m = len(preds)
    tandem_risks  = np.zeros((m,m))
    n2            = np.zeros((m,m))

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i,M_j)
            tandem_risks[i,j] = np.sum(np.logical_and(P_i[M==1]!=targs[M==1], P_j[M==1]!=targs[M==1]))
            n2[i,j] = np.sum(M)

            if i != j:
                tandem_risks[j,i] = tandem_risks[i,j]
                n2[j,i]           = n2[i,j]

    return tandem_risks, n2

reference_oob_tandem_risks(
    np.block([
        [[np.ones(100),np.zeros(60)],
         [np.ones(100),np.zeros(60)]],
        [[np.zeros(30),np.ones(100),np.zeros(30)],
         [np.zeros(30),np.ones(100),np.zeros(30)]],
        [[np.zeros(60),np.ones(100)],
         [np.zeros(60),np.ones(100)]]
    ]),
    np.block([np.zeros(100), np.ones(60)])
)
