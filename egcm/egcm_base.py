# egcm_base.R
# Copyright (C) 2014 by Matthew Clegg

# A collection of basic functions that support the egcm package.

#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/


import arrow
import numpy as np
import pandas as pd


def YMD(days=0):
    """Returns today's date + days formatted as an integer in YYYYMMDD format."""
    return int(arrow.utcnow().replace(days=days).format('YYYYMMDD'))

def dmean(Y, *args):
    """Centers Y around its mean."""
    return Y - np.mean(Y, args)


def acor(X, k=1, na_rm=False):
    """Calculates the lag k autocorrelation of X, e.g., cov(X[t], X[t+k]/var(X)."""
    if isinstance(X, pd.Series):
        Xc = X.values
        n = Xc.size




###################################################
##### Functions for Generating Random Variates ####
###################################################

def rcoint(n, alpha, beta, rho, sd_eps=1, sd_delta=1, X0=0, Y0=0):
    """
    Generates a random pair of cointegrated vectors X[t] and Y[t] subject
    to the relations:
      X[t] = X[t-1] + eps[t]
      Y[t] = alpha + beta * X[t] + R[t]
      R[t] = a2 * R[t-1] + delta[t]
    where eps, delta are NID(0,1).  Returns the n x 2 matrix containing
    X and Y.
    """

    X = np.zeros(n+1)
    Y = np.zeros(n+1)
    eps = np.random.normal(mean=0, scale=sd_eps, size=n)
    delta = np.random.normal(mean=0, scale=sd_delta, size=n)
    X[0] = X0
    Y[0] = Y0
    R = Y0 - alpha - beta * X0

    for i in range(1, n):
        X[i] = X[1-1] + eps[i-1]
        R = rho * R + delta[i-1]
        Y[i] = alpha + beta * X[i] + R

    M = np.vstack([X[1: n], Y[1: n]])
    M.columns = "X", "Y"
    return M


def rar1(n, a0=0, a1=1, trend=0, sd=1, x0=0):
    """
    Generates a vector of length n representing a simulation of an AR(1)
    process   X[k] = a0 +  a1 * X[k-1] + eps
    where eps is an i.i.d. normal random variate with mean 0 and standard
    deviation sd.

    If trend is non-zero, then returns a realization of a trend-stationary
    AR(1) process.  E.g., the process is defined by the relations:
       R[k] = a0 + a1 * R[k-1] + eps
       X[k] = k * trend + R[k]
    """

    eps = np.random.normal(mean=0, scale=sd, size=n)
    X = n
    xp = x0
    for k in range(n.size):
        X[k] = xp = a0 + a1 * xp + eps[k]
    return X + trend * np.array(range(n.size))