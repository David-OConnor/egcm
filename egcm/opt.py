# Contains optimized functions.
# Use these functions with arrays; not dataframes.

import bisect
import math

import numba
import numpy as np
import scipy.signal
import statsmodels.api as sm
from statsmodels.tsa import stattools as ts

from . import pgff

pgff_qtab = np.array(pgff.pgff_qtab)
pgff_qtab_detrended = np.array(pgff.pgff_qtab_detrended)


# qti is slighly slower (1microsec) with numba enabled than without.
@numba.jit
def quantile_table_interpolate_opt(qtab, sample_size, stat):
    i = bisect.bisect_right(qtab[0, 1:], sample_size)

    y1 = np.interp(stat, qtab[1:, i], qtab[1:, 0])
    if i < qtab.shape[1] - 1:
        y2 = np.interp(stat, qtab[1:, i+1], qtab[1:, 0])
        n1 = qtab[0, i]
        n2 = qtab[0, i+1]
        y = y1 * (n2 - sample_size) / (n2 - n1) + y2 * (sample_size - n1) / (n2 - n1)
    else:
        y = y1

    return y


@numba.jit
def pgff_(Y, detrend):
    M = Y.size

    y_squared_sum = 0.
    den2 = 0.

    if detrend:
        y = scipy.signal.detrend(Y)


        for i in range(M):
            y_squared_sum += y[i] ** 2
            den2 += (1 / M) * y[i] ** 2

    else:
        sum_ = 0.0
        len_ = 0
        for i in range(M):
            sum_ += Y[i]
            len_ += 1
        mean = sum_ / len_

        y = np.empty(M, dtype=np.float)
        for i in range(M):
            y[i] = Y[i] - mean
            y_squared_sum += y[i] ** 2
            den2 += (1 / M) * y[i] ** 2

    rho_ws_num = 0.
    for i in range(M - 1):
        rho_ws_num += (y[i] * y[i+1])

    den1 = 0.
    for i in range(1, M-1):
        den1 += y[i] ** 2

    rho_ws_den = den1 + den2
    rho = rho_ws_num / rho_ws_den

    # should I use the non-detrended qtab for a detrended test on R,
    # like in the R version under residual tests??
    qtab = pgff_qtab_detrended if detrend else pgff_qtab
    p_val = quantile_table_interpolate_opt(qtab, Y.size, rho)

    return rho, p_val


# todo numba's not helping atm.
@numba.jit
def egcm(S1, S2):
    """Optimized version of egcm.egcm.  Reference it for comments."""
    log_ = False
    normalize = False
    robust = False
    include_const = True
    p_val = .05

    # Detrend is true for urtest, false for i1test. Takes longer.

    if log_:
        # log_optimized performance is similar to np.log.
        S1 = log(S1)
        S2 = log(S2)
        # S1 = np.log(S1)
        # S2 = np.log(S2)

    # todo see if you can speed the glm
    L = sm.GLM(S2, ts.add_constant(S1, prepend=False)).fit()

    R = L.resid if robust else L.resid_pearson
    print(R[:10], 'resid opt')

    r_stat, r_pval = pgff_(R, True)

    return r_pval


@numba.jit
def log(s):
    M = s.size
    s_new = np.empty(M, dtype=np.float)
    for i in range(M):
        s_new[i] = math.log(s[i])
    return s_new