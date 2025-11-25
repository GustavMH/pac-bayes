#!/usr/bin/env python3

def test_oob_tandem_risk():
    oob = oob_tandem_risks(
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
    print(oob)
