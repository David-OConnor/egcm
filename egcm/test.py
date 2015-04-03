# egcm.R
# Copyright (C) 2014 by Matthew Clegg

# Engle-Granger Cointegration Models for Pairs Trading
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# A copy of the GNU General Public License is available at
# http://www.r-project.org/Licenses/
# The purpose of this module is to provide a simple implementation of
# the Engle Granger cointegration model that is convenient for use in
# the analysis of pairs trades.
#
# Given two time series Y[t] and X[t], the Engle Granger cointegration
# model in its simplest form looks for alpha, beta and rho such that
#
# Y[t] = alpha + beta * X[t] + R[t]
# R[t] = rho * R[t-1] + epsilon[t]
#
# where epsilon[t] is a series of independent and identically distributed
# innovations with mean zero. If alpha, beta and rho can be found such that
# -1 < rho < 1, then the series are said to be cointegrated. If abs(rho) = 1,
# then the residual series R[t] is said to have a unit root (or alternatively,
# it is said to follow a random walk).
#
# It should be noted that Engle and Granger's 1987 paper allows for a more general
# definition of cointegration. Namely, there can be multiple series included
# in the cointegrating relationship. Also, the residual series R[t] may
# be any ARMA process. If this greater generality is needed, the reader
# is referred to the urca package of Bernhard Pfaff.
#
# The major functions provided by this module are as follows:
#
# egcm(X,Y) -- Constructs an Engle-Granger cointegration model from X & Y
# summary.egcm(E) -- Prints various summary statistics on the Engle-Granger
# cointegration model constructed from X & Y
# plot.egcm(E) -- Creates a graph of the Engle-Granger cointegration model
#
# The following ancillary functions are also provided:
#
# rcoint(n, alpha, beta, rho) -- Generates a random pair of cointegrated vectors
# egc_test_specificity() -- Calculates the specificity of a cointegration (unit root) test
# egc_test_power() -- Calculates the power of a cointegration (unit root) test
# egc_test_power_table() -- Calculates a table of powers of a cointegration (unit root) test
# as.data.frame.egcm() -- Converts an egcm object to a single row data.frame
# residuals.egcm() -- Returns the residual series R[t] associated to an egcm object
# innovations.egcm() -- Returns the series epsilon[t] of innovations associated to an
# egcm object
# test.egcm.innovations() -- Tests the goodness of fit of a set of innovations
#
#
# References
#
# Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction:
# representation, estimation, and testing. Econometrica, (55) 2, 251-276.
# Pfaff, Bernhard (2008). Analysis of Integrated and Cointegrated Time
# Series with R. Springer.

import bisect
from collections import namedtuple

import arch.unitroot as ur
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa import stattools as ts
from statsmodels.stats import diagnostic


from . import base
from . import data
from . import pgff
from . import bvr
from . import johansen


################################################
##### Engle Granger Cointegration Test #####
################################################


# The EGCM package maintains several writable variables that are local to the
# package. These are saved in the following environment and accessed with getter
# and setter methods.

class EgcmEnv:
    def __init__(self):
        self.urtests = [
            'adfraw',  # Augmented Dickey-Fuller (adf.test{tseries}), using default p-values

            # All of the tests below use p-values that have been re-calibrated
            'adf',  # Augmented Dickey-Fuller test (adf.test{tseries})
            'pp',  # Phillips-Perron test (pp.test{tseries})
            'jo-e',  # Johansen's eigenvalue test (ca.jo{urca})
            'jo-t',  # Johansen's trace test (ca.jo{urca})
            'ers-p',  # Elliott, Rothenberg and Stock point optimal test (ur.ers{urca})
            'ers-d',  # Elliott, Rothenberg and Stock DF-GLS test (ur.ers{urca})
            'sp-r',  # Schmidt and Phillips rho statistics (ur.sp{urca})
            'hurst',  # Hurst exponent (aggvarFit{fArma})
            'bvr',  # Breitung's variance ratio
            'pgff'  # Pantula, Gonzales-Farias and Fuller Rho statistic
        ]

        # The following global variable enumerates the tests that are available for
        # checking if the input series are integrated. The tests from the urca library
        # have been omitted because these tests do not provide p-values.

        self.i1tests = [
            'adf',  # Augmented Dickey-Fuller (adf.test{tseries}), default p.values
            'pp',  # Phillips-Perron test (pp.test{tseries}), default p.values
            'bvr',  # Breitung's variance ratio
            'pgff'  # Pantula, Gonzales-Farias and Fuller Rho statistic
        ]
        
        self.urtest_default = 'pp'
        self.i1test_default = 'pp'
        self.pvalue_default = .05

    # The following getter/setter functions are used to control the
    # confidence level (p-value) that is used for the various statistical
    # tests.

    def set_default_pvalue(self, p):
        """Sets the default p-value used in the tests of cointegration."""
        if p < .001:
            raise AttributeError("p values less than 0.001 are not supported")
        self.pvalue_default = p


# include_investment_scenarios optional component from R module ommitted here.


egcm_env = EgcmEnv()


def test_egcm(EGCM, test_method=egcm_env.urtest_default):
    """
    Tests whether EGCM is cointegrated by performing a unit root test on the
    residual series. The choice of unit root test is determined by test.method:
    pp: Phillips-Perron test, as implemented by pp.test{tseries}.
    pgff: Weighted symmetric estimator rho of Pantula, Gonzales-Farias and Fuller.
    jo-e: Eigenvalue test from Johansen's VECM model as implemented by ca.jo{urca}.
    jo-t: Trace test from Johansen's VECM model as implemented by ca.jo{urca}.
    ers-p: Point optimal test of Elliott, Rothenberg and Stock as implemented by ur.ers{urca}.
    ers-d: DF-GLS test of Elliott, Rothenberg and Stock as implemented by ur.ers{urca}.
    sp-r: Rho statistic from Schmidt and Phillips, as implemented by ur.sp{urca}.
    bvr: Breitung's variance ratio test.
    hurst: Hurst exponent as calculated by aggvarFit{fArma}.

    EGCM can either be an egcm object returned by egcm(), or it can be a two-column
    matrix. In the latter case, a regression is performed of the first column on the
    second to obtain the residual series, which is tested for a unit root.

    Returns an object of type htest representing the results of the hypothesis test.
    A low p.value is interpreted as evidence that the null hypothesis
    of a unit root should be rejected. In other words, a low p.value is evidence
    for cointegration.
    """

    data_name = "Placeholder"

    # If EGCM is a result of EGCM(), find X and R from parsing it.
    if isinstance(EGCM, dict):
        R = EGCM['residuals']
        X = np.vstack([EGCM['S1'], EGCM['S2']])

    # If EGCM is a 2-column array or DataFrame, calculate R.
    else:
        # If input is a 2-column dataframe, handle as below. If a vstacked numpy
        # array, leave alone.
        X = EGCM.T.values if isinstance(EGCM, pd.DataFrame) else EGCM

        # R's cov() returns a single value.  numpy.cov() returns a 2x2 matrix,
        # where the values at [0][1] and [1][0] are the same as R's cov() value.
        beta = np.cov(X[1], X[0])[0][1] / np.var(X[0], ddof=1)
        alpha = np.mean(X[1]) - beta * np.mean(X[0])
        R = X[1] - beta * X[0] - alpha

    if test_method in ['jo-e', 'jo-t']:
        jo = johansen.coint_johansen(X, 0, 2)
        if test_method == 'jo-e':
            STAT = jo.lr2
            PVAL = base.quantile_table_interpolate(data.egc_joe_qtab, R.size, STAT)
            URTEST = "Johansen eigenvalue test"
        else:
            STAT = jo.lr1
            PVAL = base.quantile_table_interpolate(data.egc_jot_qtab, R.size, STAT)
            URTEST = "Johansen trace test."

        htest = {
            'statistic': STAT,
            'alternative': "cointegrated",
            'p_value': PVAL,
            'method': "",
            'urtest': URTEST,
            'data_name': data_name
        }

    else:
        htest = egc_residuals_test(R, test_method)
        htest['data_name'] = data_name

    htest['method'] = "Engle Granger two-step cointegration test {0}".format(test_method)
    return htest


def egc_residuals_test(R, test_method=None):
    """
    Tests whether the residual series from the Engle-Granger procedure
    contains a unit root.

    Input values:
    R: The residual series that is to be tested
    test.method: The method to be used for testing for a unit root.
    One of the following choices is permitted:

    adfraw: Augmented Dickey-Fuller test, as implemeneted by adf.test{tseries}.
    adf: Augmented Dickey-Fuller test, but with re-calibrated p-values.
    pp: Phillips-Perron test, as implemented by pp.test{tseries}.
    pgff: Weighted symmetric estimator rho of Pantula, Gonzales-Farias and Fuller.
    ers-p: Point optimal test of Elliott, Rothenberg and Stock as implemented by ur.ers{urca}.
    ers-d: DF-GLS test of Elliott, Rothenberg and Stock as implemented by ur.ers{urca}.
    sp-r: Rho statistic from Schmidt and Phillips, as implemented by ur.sp{urca}.
    bvr: Breitung's variance ratio test.
    hurst: Hurst exponent as calculated by aggvarFit{fArma}.

    Returns an object of type htest representing the results of the hypothesis test.
    In all cases, a low p.value is interpreted as evidence that the null hypothesis
    of a unit root should be rejected. In other words, a low p.value is evidence
    for cointegration.

    For all of the tests except adfraw, the p-values have been re-calibrated.
    """

    DNAME = "placeholder"
    METHOD = "Unit root test {0} of residuals in Engle Granger procedure".format(test_method)

    if test_method in ['adfraw', 'adf']:
        maxlag = np.floor((R.size-1)**(1/3))
        adf = ts.adfuller(R, regression='ct', maxlag=maxlag, autolag=None)
        STAT = adf[0]
        if test_method == 'adfraw':
            PVAL = adf[1]
            URTEST = "Augmented Dickey-Fuller test (Raw)"
        else:
            PVAL = base.quantile_table_interpolate(data.egc_adf_qtab, R.size, STAT)
            URTEST = "Augmented Dickey-Fuller test"

    # Uses egcm module's own implemention of pgff.
    elif test_method == 'pgff':
       STAT = pgff.rho_ws(R, detrend=True)
       PVAL = base.quantile_table_interpolate(data.egc_pgff_qtab, R.size, STAT)
       URTEST = "Pantula, Gonzales-Farias and Fuller Unit Root Test"

    elif test_method == 'pp':
        lags = int(np.floor(4*(R.size/100)**0.25))
        pp = ur.PhillipsPerron(R, trend='ct', lags=lags, test_type='rho')
        STAT = pp.stat
        PVAL = base.quantile_table_interpolate(data.egc_pp_qtab, R.size, STAT)
        URTEST = "Phillips-Perron test"

    # Can't find a non-GLS ERS test in python.
    # elif test_method == 'ers-p':
    #    ers = ur.ers(R, type='P-test', model='constant')
    #    STAT = ers@teststat
    #    names(STAT) = 'ERS P-test'
    #    PVAL = quantile_table_interpolate(egc_ersp_qtab, length(R), STAT)
    #    URTEST = ers@test.name

    elif test_method == 'ers-d':
        ers = ur.DFGLS(R, trend='c')
        STAT = ers.stat
        PVAL = base.quantile_table_interpolate(data.egc_ersd_qtab, R.size, STAT)
        URTEST = "Elliott, Rothenberg and Stock’s GLS version of the Dickey-Fuller test"

    elif test_method == 'sp-r':
        sp = ur.KPSS(R)
        STAT = sp.stat
        PVAL = base.quantile_table_interpolate(data.egc_spr_qtab, R.size, STAT)
        URTEST = "Kwiatkowski, Phillips, Schmidt and Shin (KPSS) stationarity test."

    # Uses egcm module's own implemention of pgff.
    elif test_method == 'bvr':
        STAT = bvr.bvr_rho(R)
        PVAL = base.quantile_table_interpolate(data.egc_bvr_qtab, R.size, STAT)
        URTEST = 'Breitung Variance Ratio Test'

    # Can't find hurst test for Python.
    # elif test_method == 'hurst':
    #     h = aggvarFit(R)
    #     STAT = h@hurst$H
    #     names(STAT) = 'H'
    #     PVAL = quantile_table_interpolate(egcm_data.egc_hurst_qtab, length(R), STAT)
    #     URTEST = h@title
    else:
        raise AttributeError('Unit root test_method {0} not implemented'.format(test_method))

    return {
        'statistic': STAT,
        'alternative': "cointegrated",
        'p_value': PVAL,
        'method': METHOD,
        'urtest': URTEST,
        'data_name': DNAME,
    }


def egc_test_power(test_method=egcm_env.urtest_default, rho=0.95, n=250, nrep=1000, p_value=0.05):
    """
    Uses simulation to estimate the power of a cointegration test.
    Power is defined to be the probability that the null hypothesis is rejected
    when the null hypothesis is false. Power is parameterized in terms of the
    mean reversion coefficient rho, the sample size n, and the p-value.
    The power is computed by generating random cointegrated
    pairs, and then counting the number of such pairs that are
    identified as cointegrated.
    """

    pvalues = np.repeat(test_egcm(base.rcoint(n, rho=rho)['p_value'], test_method), nrep)
    return sum(pv for pv in pvalues if pv < p_value) / pvalues.size



def egc_test_power_table(test_method=egcm_env.urtest_default, nrep=4000,
                         p_value=0.05, rho=(0.80, 0.90, 0.92, 0.94, 0.96, 0.98),
                         n=(60, 125, 250, 500, 1000)):
    """Constructs a table of power estimates for realistic values of rho."""
    def do_row(nv):
        return (egc_test_power(test_method, r, nv, nrep, p_value) for r in rho)


def egc_quantiles(test_method=egcm_env.urtest_default,
                  sample_size=100, nrep=40000,
                  q=(0.0001, 0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 0.80,
                     0.90, 0.95, 0.975, 0.99, 0.999, 0.9999)):
    """Calculates quantiles of the unit root test statistic under the assumption rho=1."""
    qvals = np.repeat(egcm(base.rcoint(sample_size, rho=1), urtest=test_method)['r_stat'], nrep)
    # return quantile(qvals, q)



# def egc_quantile_table(test_method=egcm_env.urtest_default, nrep=40000,
#                        q=(0.0001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 0.80,
#                           0.90, 0.95, 0.975, 0.99, 0.999, 0.9999),
#                        n=(25, 50, 100, 250, 500, 750, 1000, 1250, 2500)):
#
#     df = do.call("cbind", mclapply(n, function(nv) c(nv, egc_quantiles(test.method, nv, nrep, q))))
#     df = as.data_frame(df)
#     colnames(df) = n
#     df = cbind(data_frame(quantile=c(NA,q)), df)
#     return df


################################################
##### Engle Granger Cointegration Model #####
################################################

def egcm(X, Y=None, log=False, normalize=False, debias=True, robust=False,
         include_const=True, i1test=egcm_env.i1test_default,
         urtest=egcm_env.urtest_default, p_value=egcm_env.pvalue_default):
    """
    Performs the two-step Engle Granger cointegration analysis on the price
    series X and Y, and creates an object representing the results of the
    analysis.

    If X is the price series of the first security and Y is
    the price series of the second, then computes the fit:

    Y = alpha + beta * X + R
    R_t = rho * R_{t-1} + eps_t

    If log is TRUE, then the price series are logged before the analysis is
    performed. If Y is missing, then X is presumed to be a two-column
    matrix, and X[,1] and X[,2] are used in place of X and Y.
    """

    if Y is None:
        if len(X.columns) != 2:
            raise AttributeError("If Y is missing, X must be a two-column DataFrame.")
        series_names = X.columns
        S2 = X[1]
        S1 = X[0]

    elif Y.size != X.size:
        raise AttributeError("X and Y must be numeric vectors of the same length")
    else:
        S1 = X
        S2 = Y
        try:
            series_names = [S1.name, S2.name]
        except AttributeError:
            series_names = ['X', 'Y']

    if p_value < 0.001:
        raise ValueError("P-values less than 0.001 are not supported")

    if normalize:
        S1 = S1 / S1[0]
        S2 = S2 / S2[0]

    if log:
        S1 = np.log(S1)
        S2 = np.log(S2)

    # Necessary to prevent index alignment issues when calculating LR
    if isinstance(S1, pd.Series):
        S1 = S1.values
    if isinstance(S2, pd.Series):
        S2 = S2.values

    # Note: The robust regressions are returning slightly different results
    # from the R equivalent. Global are returning the same.

    # Global regressions seem to have the same results as running an OLS. Robusts are much slower,
    # But much more accurate. AFAIK, always use include_const, or the Y intercept is placed at 0,
    # regardless if that's appropriate.

    if robust and include_const:
        L = sm.RLM(S2, ts.add_constant(S1)).fit()
        alpha = L.params[0]
        beta = L.params[1]
    elif robust:
        L = sm.RLM(S2, S1).fit()
        alpha = 0
        beta = L.params[0]
    elif include_const:
        L = sm.GLM(S2, ts.add_constant(S1)).fit()
        alpha = L.params[0]
        beta = L.params[1]
    else:
        L = sm.GLM(S2, S1).fit()
        alpha = 0
        beta = L.params[0]

    # resid_deviance is alternative to pearson, but they seem to return the same results.
    R = L.resid if robust else L.resid_pearson
    FR = R[1:]
    BR = R[: -1]

    if not robust:
        LR = sm.GLM(FR, BR).fit()
        rho_raw = LR.params[0]
    else:
        LR = sm.RLM(FR, BR).fit()
        rho_raw = LR.params[0]

    rho = debias_rho(FR.size, rho_raw) if debias else rho_raw

    eps = FR - rho * BR

    if include_const:
        # alpha.se <- coef(L)[1,2]
        # The following works well in simulated data, but I am not sure if
        # it is correct. In any event, simply taking the standard error
        # generated by lm() is clearly incorrect.
        alpha_se = np.sqrt(L.bse[1]**2 + np.var(eps, ddof=1)/4)
        # Engle and Granger show that the convergence of beta is super-consistent,
        # however simulation shows that the standard errors computed by lm() seem
        # to be too small. Perhaps a correction based on the var(eps) needs to
        # be included here as well?
        beta_se = L.bse[0]
    else:
        alpha_se = 0
        beta_se = L.bse[0]

    rho_se = LR.bse[0]

    def i1testfunc(X, i1test):
        if i1test == 'adf':
            maxlag = np.floor((X.size-1)**(1/3))
            adf = ts.adfuller(X, regression='ct', maxlag=maxlag, autolag=None)
            stat, p_value_ = adf[0], adf[1]
        elif i1test == 'pgff':
            pgff_ = pgff.test(X)
            stat, p_value_ = pgff_['statistic'], pgff_['p_value']
        elif i1test == 'bvr':
            bvr_ = bvr.test(X)
            stat, p_value_ = bvr_['statistic'], bvr_['p_value']
        elif i1test == 'pp':
            lags = int(np.floor(4*(X.size/100)**0.25))
            pp = ur.PhillipsPerron(X, trend='ct', lags=lags, test_type='rho')
            stat, p_value_ = pp.stat, pp.pvalue

        return stat, p_value_

    S1i1_stat, S1i1_pvalue = i1testfunc(S1, i1test)
    S2i1_stat, S2i1_pvalue = i1testfunc(S2, i1test)

    if urtest in ['jo-t', 'jo-e']:
        R_test = test_egcm(np.vstack([S1, S2]), urtest)
    else:
        R_test = egc_residuals_test(R, urtest)

    # ljungbox doesn't support sample sizes smaller than 41.
    if eps.size > 40:
        lb = diagnostic.acorr_ljungbox(eps, lags=1)
        lb_value = float(lb[0])
        lb_pvalue = float(lb[1])
    else:
        lb_value = None
        lb_pvalue = None

    return {
        'S1': S1,
        'S2': S2,
        'residuals': R,
        'innovations': eps,
        'series_names': series_names,
        'index': S1.size,
        'i1test': i1test,
        'urtest': urtest,
        'pvalue': p_value,
        'log': log,
        'alpha': alpha,
        'alpha_se ': alpha_se,
        'beta': beta,
        'beta_se ': beta_se,
        'rho': rho,
        'rho_raw': rho_raw,
        'rho_se': rho_se,
        's1_i1_stat': S1i1_stat,
        's1_i1_p': S1i1_pvalue,
        's2_i1_stat': S2i1_stat,
        's2_i1_p': S2i1_pvalue,
        'r_stat': R_test['statistic'],
        'r_p': R_test['p_value'],
        'eps_ljungbox_stat': lb_value,
        'eps_ljungbox_p': lb_pvalue,
        's1_dsd': np.std(np.diff(S1), ddof=1),
        's2_dsd': np.std(np.diff(S2), ddof=1),
        'residuals_sd': np.std(R, ddof=1),
        'eps_sd': np.std(eps, ddof=1)
    }


def is_cointegrated(E):
    """Returns TRUE if the egcm model E appears to be cointegrated."""
    S1_I1 = E['s1_i1_p'] > E['pvalue']
    S2_I1 = E['s2_i1_p'] > E['pvalue']
    R_I0 = E['r_p'] < E['pvalue']

    return S1_I1 and S2_I1 and R_I0


def is_ar1(E):
    """Returns TRUE if the residuals in the egcm model E are adequately
       fit by an AR(1) model."""

    return E['eps_ljungbox.p'] > E['p_value']


################################################
##### De-biasing of Rho #####
################################################


class RhoBiasTable:
    """
    Calculates the bias of an estimator RHO_EST of rho in the Engle-Granger model
    Y[t] = alpha + beta * X[t] + R[t]
    R[t] = rho * R[t-1] + eps
    where eps is NID(0,1).

    N specifies the various vector lengths that are to be considered,
    and RHO specifies the actual values of RHO that are to be used in
    the simulation.
    """

def __init__(self,
             n=(60, 125, 250, 500, 750, 1000, 2500),
             rho=(0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1),
             nsamples=1000,
             print_=True,
             report_sd=False):

    self.n = n
    self.rho = rho
    self.nsamples = nsamples
    self.print_ = print_
    self.report_sd = report_sd


def rho_hat_sample(self):
    pass

# todo this won't work without rcoint.


def qtab_to_ltab(tab=data.rho_bias_qtab):
    pass


def generate_rho_bias_table():
    n = [25, 50, 100, 200, 400, 800, 1200, 1600, 2000]
    rho = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    qtab = RhoBiasTable(N=n, RHO=rho, nsamples=10000, print_=False)
    return qtab_to_ltab(qtab)


def debias_rho(sample_size, rho):
    """
    Given an estimate rho of the mean reversion parameter rho obtained through
    the Engle Granger procedure, calculates a de-biased estimate of rho.

    Several different approaches have been tried for debiasing rho. The
    first approach was to compute a table of values indexed by sample_size
    and rho. Each entry in the table contains the mean of the estimated
    value rho_hat for that pair (sample_size, rho). The table is then
    inverted using linear interpolation to obtain a debiased estimate of
    rho_hat. This approach was found to yield an improvement, but the
    estimates obtained in this way were still biased.

    It was then observed that for a given sample_size, the entries in
    the debias table are nearly perfectly linear in rho. Based upon this,
    the debias table was condensed down to a set of linear relations,
    indexed by sample_size. Interpolation is used for sample_sizes that
    are intermediate with respect to those represented in the table.

    Upon inspection of the debiasing table, it was found that the following
    relation seems to offer a reasonably good de-biasing estimate:
    rho_hat_debiased = rho_hat * (1 + 6/sample_size)

    Another approach that was tried was to choose random cointegrated
    vectors and then to use the Engle Granger procedure to estimate
    rho_hat. A table of values (rho, rho_hat) was constructed in this
    way, and then coefficients c_0 and c_1 were obtained which gave
    the best fit to rho_hat = c_0 + c_1 rho. These values were then
    inverted to obtain the debiasing relation, e.g.,
    rho_hat_debiased = -(c_0/c_1) + (1 / c_1) rho_hat
    This last approach seems to give the best fit, and so it is the
    method that is used.
    """

    # return(rho * (1 + 6/sample_size))
    # return(quantile_table_interpolate(rho_bias_qtab, sample_size, rho))

    # For a given sample size, the bias is well described as a linear
    # function of rho.
    table = data.rho_bias_ltab
    i = bisect.bisect_right(table['n'], sample_size) - 1

    if i == -1:
        y = table['c0'][0] + table['c1'][0] * rho
    elif i < len(table.index):
        y1 = table['c0'][i] + table['c1'][i] * rho
        y2 = table['c0'][i+1] + table['c1'][i+1] * rho
        n1 = table['n'][i]
        n2 = table['n'][i+1]
        w1 = (n2 - sample_size) / (n2 - n1)
        w2 = 1 - w1
        y = y1 * w1 + y2 * w2
        y = min(y, 1)
    else:
        y1 = table['c0'][i] + table['c1'][i] * rho
        w1 = np.exp(1 - sample_size / table['n'][i])
        w2 = 1 - w1
        y = y1 * w1 + rho * w2
        y = min(y, 1)

    return y


def cbtab_to_rhobtab(cbt):
    n = cbt['sample_size']
    c0r = cbt['Estimate_c0raw']
    c1r = cbt['Estimate_c1raw']
    c0 = -1 * c0r / c1r
    c1 = 1 / c1r
    return pd.DataFrame({'n': n, 'c0': c0, 'c1': c1})