# Contains optimized functions.
# Use these functions with arrays; not dataframes.

import math

import numba
import numpy as np
import brisk
import scipy.signal
import statsmodels.api as sm
from statsmodels.tsa import stattools as ts

from . import data
from . import pgff

pgff_qtab = np.array(pgff.pgff_qtab)
pgff_qtab_detrended = np.array(pgff.pgff_qtab_detrended)
egc_pgff_qtab = np.array(data.egc_pgff_qtab)


# qti is slighly slower (1microsec) with numba enabled than without.
@numba.jit(nopython=True)
def quantile_table_interpolate(qtab, sample_size, stat):
    i = brisk.bisect(qtab[0, 1:], sample_size)

    y1 = brisk.interp_one(stat, qtab[1:, i], qtab[1:, 0])
    if i < qtab.shape[1] - 1:
        y2 = brisk.interp_one(stat, qtab[1:, i+1], qtab[1:, 0])
        n1 = qtab[0, i]
        n2 = qtab[0, i+1]
        y = y1 * (n2 - sample_size) / (n2 - n1) + y2 * (sample_size - n1) / (n2 - n1)
    else:
        y = y1

    return y


@numba.jit
def pgff_(Y, is_i1test):
    # If is_i1test, don't detrend the data, and use pgff's quantile table interp
    # instead of the egc version. If not, it's a unit root test on residuals;
    # in this case, detrend the data, and use egc_pgff_qtab.

    M = Y.size

    y_squared_sum = 0.
    den2 = 0.

    if not is_i1test:
        y = brisk.detrend(Y, 'l')

        for i in range(M):
            y_squared_sum += y[i] ** 2
            den2 += (1 / M) * y[i] ** 2

    else:
        mean = brisk.mean(Y)

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

    if rho_ws_den == 0:
        rho = 0
    else:
        rho = rho_ws_num / rho_ws_den

    qtab = pgff_qtab if is_i1test else egc_pgff_qtab
    p_val = quantile_table_interpolate(qtab, Y.size, rho)

    return rho, p_val


# todo numba's not helping atm.
# @numba.jit()
def egcm(S1, S2):
    """Optimized version of egcm.egcm.  Reference it for comments."""
    log_ = False
    normalize = False

    # Detrend is true for urtest, false for i1test. Takes longer.

    if log_:
        # log_optimized performance is similar to np.log.
        S1 = log(S1)
        S2 = log(S2)
        S1 = np.log(S1)
        S2 = np.log(S2)

    # todo see if you can implement a fast RLM. Much more accurate, takes MUCH longer than
    # todo your optimized OLS/GLM solution.

    slope, intercept = brisk.ols(S1, S2)
    R = brisk.lin_resids(S1, S2, slope, intercept)

    r_stat, r_pval = pgff_(R, False)

    return r_stat, r_pval


# todo implement your own log part, and move to brisk.
@numba.jit
def log(s):
    M = s.size
    s_new = np.empty(M, dtype=np.float)
    for i in range(M):
        s_new[i] = math.log(s[i])
    return s_new