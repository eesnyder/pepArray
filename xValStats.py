#!/usr/bin/env python3
'''
Pathology Prediction by Immunosignature
(c) Eric E. Snyder, Ph.D. (2017)
eesnyder@eesnyder.org
'''
# v0.96 Allow score_blinded_samples() to use input containing extraneous samples
# v0.97 Start working on t-tests recalculated for each kfold x-validation (not implemented)
#       removed short options to simplify/clarify usage information
#       .par file can now take the comment char, '#', at the beginning of the line to comment out the whole parameter or
#       in the middle of a line to comment out arguments or add a bonafied comment on a per-line basis.
#       Fixed problem with not providing blind factor name so it doesn't crash
# v0.98 Removed most commented lines for a clean start before implementing xval t-tests
# v1.00 recalculate t-tests on each kfold xvalidation iteration
# v1.01 confirm that t-test/kfold is working as intended
# v1.02 Migrate code to Python3.6
# v1.04 pruning obsolete options, documentation
# v1.05 continue plus rework initialization of PeptideBySubject class
# Wed, Oct 25, 2017  8:13:28 AM

from sklearn import preprocessing, model_selection, metrics
import scipy.stats as stats
import numpy as np
from math import sqrt
import createPepArray as cpa

TruthTableLength = 8           # tp, fp, fn, tn, sn, sp, acc, CC

class TruthTable():
    '''
    Calculate TruthTable values including raw counts and statistics (sn, sp, acc, CC)
    given TrueValList, PredictedValList.  Save information as object attributes for
    printing
    '''
    def __init__(self, myTrue, myPred):
        '''
        TruthTable(myTrue, myPred)
        Calculate confusion matrix values (tp, fp, fn, tn) plus analyses based on them:
        sensitivity (sn), specificity (sp), positive predictive value (ppv), negative
        predictive value (npv), accuracy (acc), and correlation coefficient (CC).  These
        values are returned as an ndarray of type=np.float.
        '''
        tn, fp, fn, tp = (0, 0, 0, 0)
        nfactors = len(myTrue)
        discrete_values = np.unique(np.concatenate((myTrue, myPred)))   # this code deals with a problem in
        if len(discrete_values) == 1:                                   # confusion_matrix which returns only one value
            if discrete_values[0] == 1:                                 # if the two inputs are homogeneous,
                tp = nfactors                                           # e.g., [1,1,1,1] and [1,1,1,1]
            elif discrete_values[0] == 0:
                tn = nfactors
            else:
                sys.exit('ERROR: unexpected value, {:3.1f}, for "discrete_values[0]" in TruthTable.__init__'.format(discrete_values[0]))
        else:
            tn, fp, fn, tp = metrics.confusion_matrix(myTrue, myPred).ravel()

        self.counts = [tp, fp, fn, tn]
        (tn, fp, fn, tp) = np.asarray((tn, fp, fn, tp), dtype=np.float)     # recast from int to float
        sn  = tp / (tp+fn)
        sp  = tn / (tn+fp)
        acc = (tp+tn) / (tp+fp+tn+fn)
        CC  = ( (tp*tn) - (fp*fn) ) / sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )
        self.acc = acc
        self.CC  = CC
        self.stats = [sn, sp, acc, CC]
        self.output_array = self.counts + self.stats    # self.output_array = [tp, fp, fn, tn, sn, sp, acc, CC]

    def tts(self):
        '''
        Return ndarray containing raw counts and statistics.
        '''
        return self.output_array

    def CC(self):
        '''
        Return correlation coefficient as scalar float
        '''
        return self.CC

class FormatTruthTable(TruthTable):
    '''
    Construct the basic shape of a Truth Table row, whether it part of a heading,
    consisting entirely of dashes or column labels, or a real row of data with
    proper formatting codes, etc.
    '''
    def __init__(self):
        TTFirstColsFLabels = ['f1', 'f2', 'nbest', 'validation']
        TTFirstColsHWidths = [5, 5, 12]
        TTFirstColsFWidths = [2, 2, 5, 12]
        TTFirstColsFFormats = ['<2d', '<2d', '5d', '12s']
        TTFirstColsSep = ' '
        TTCountLabels = ['tp', 'fp', 'fn', 'tn']
        TTCountWidths = [4, 4, 4, 4]
        TTCountFormats = ['4d', '4d', '4d', '4d']
        TTCountSep = '  '
        TTStatsLabels = ['sn', 'sp', 'acc', 'CC']
        TTStatsWidths = [6, 6, 6, 7]
        TTStatsFormats = ['6.4f', '6.4f', '6.4f', '7.4f']
        TTStatsSep = '  '
        self.HWidths = (TTFirstColsHWidths, TTCountWidths, TTStatsWidths)
        self.FWidths = (TTFirstColsFWidths, TTCountWidths, TTStatsWidths)
        self.FFormats = (TTFirstColsFFormats, TTCountFormats, TTStatsFormats)
        self.Sep = (TTFirstColsSep, TTCountSep, TTStatsSep)
        self.BlockSep = ('  ', '  ', '')
        self.xvStats = TTCountLabels + TTStatsLabels
        self.staticLabels = TTFirstColsFLabels[:3]          # f1, f2, nbest
        self.TTLabels = TTFirstColsFLabels + self.xvStats
        self.TTLength = len(self.TTLabels)

    def header(self):
        '''
        Create headings for master human-readable output file.
        '''
        TTHline = ''
        TTHstringFmt = ''
        TTDfieldFmt = ''
        for i in range(len(self.HWidths)):
            TTHline      += self.Sep[i].join(['-' * x              for x in self.HWidths[i]])  + self.BlockSep[i]
            TTHstringFmt += self.Sep[i].join(['{:' + str(x) + 's}' for x in self.FWidths[i]])  + self.BlockSep[i]
            TTDfieldFmt  += self.Sep[i].join(['{:' +     x  +  '}' for x in self.FFormats[i]]) + self.BlockSep[i]
        TTHlabels = TTHstringFmt.format(*self.TTLabels)
        self.TTHeader = TTHline + '\n' + TTHlabels + '\n' + TTHline
        self.TTHline = TTHline
        self.TTDfieldFmt = TTDfieldFmt

    def print_header(self, fh):
        '''
        Create and print the master output header; return vblock separator (a line containing a series
        of dashed lines mirroring columns) and TruthTable data field format string for formatting data
        gathered in main loop.
        '''
        self.header()
        print(self.TTHeader, file=fh)
        return(self.TTHline, self.TTDfieldFmt)
    
    def ftts(self, myTrue, myPred, xval_proc, i, j, nbest):
        myTT = TruthTable(myTrue, myPred)
        out_array =  myTT.output_array
        if type(nbest) != type(int):
            nbest = int(nbest)                  # a hack to support predictions on blinded, which forces everything to strings
        counts    =  myTT.counts
        datout    =  [i, j, nbest, xval_proc]
        datout    += [int(x) for x in counts]
        datout    += myTT.stats
        strout  = self.TTDfieldFmt.format(*datout)
        return strout, out_array
        
def xval_pct(model, X, t, test_size, replicates, args):
    '''
    xval_pct(classifier_model, training_data, targets, holdout fraction, replicates, cmdline_args)
    Perform cross-validation test based on withholding a percentage (fraction) of dataset,
    training on the remainder and testing on the holdouts.  It is assumed that the holdout
    fraction will be less than 0.5, more typically 0.2, 0.1, or 0.05 (depending on the
    number of subjects in the study)
    '''
    n_iterations = int(1 // test_size) + 1
    scoreA = np.zeros(n_iterations * replicates, dtype=np.float)
    for i in range(n_iterations * replicates):
        X_train, X_test, t_train, t_test = model_selection.train_test_split(X, t, test_size=test_size)
        model.fit(X_train, t_train)
        scoreA[i] = model.score(X_test, t_test)
        if args.debug:
            cpa.eprint("{:2d}/{:3d}   {:6.3f}".format(i, n_iterations, scoreA[i]))

    mean = np.mean(scoreA)
    median = np.median(scoreA)
    std = np.std(scoreA)
    return np.array([mean, median, std])

class kfold_stats:
    '''
    Class to accumulate k-fold cross validation statistics in a k-wise manner so that
    error bars, etc., can be calculated.
    '''
    def __init__(self, K):
        self.K = K
        self.kf_true = []
        self.kf_pred = []
        self.kf_TruthT   = np.zeros((K, TruthTableLength))     # 10 = number of raw parameters returned by truth_table()
        self.kcounter = 0

    def add_record(self, true_vec, pred_vec):
        '''
        Add data from one kfold iteration.  Each k-fold replicate is represented by
        a row in self.kf_TruthT, with the statistics aggregated by column in the order they
        occur in the TruthTableLabels list.
        '''
        self.kf_true.append(true_vec)
        self.kf_pred.append(pred_vec)
        raw_vals = TruthTable(true_vec, pred_vec).tts()
        self.kf_TruthT[self.kcounter]   = raw_vals
        self.kcounter += 1

    def summary(self):
        '''
        Calculate truth table with average and stdev on each metric
        '''
        self.min  = np.full(TruthTableLength, 9999, dtype=np.float)
        self.max  = np.full(TruthTableLength, -1,   dtype=np.float)
        self.mean = np.full(TruthTableLength, 9999, dtype=np.float)
        self.std  = np.full(TruthTableLength, 9999, dtype=np.float)
        for i in range(TruthTableLength):
            self.min[i]  = np.nanmin( self.kf_TruthT[:,i], axis=0)
            self.max[i]  = np.nanmax( self.kf_TruthT[:,i], axis=0)
            self.mean[i] = np.nanmean(self.kf_TruthT[:,i], axis=0)
            self.std[i]  = np.nanstd( self.kf_TruthT[:,i], axis=0)
        summary = np.asarray([self.mean, self.std, self.min, self.max], dtype=np.float)
        return(summary)

    def collected_values(self):
        '''
        Instead of calculating the summary values, return a matrix of truth_table values,
        each row representing one iteration of the k-fold cross validation.  This allows
        one to use the (stupid) box-and-whiskers plot in Excel (if you are so inclined).
        The data is structured so that each TT statistic has a column with the replicates
        in rows.
        '''
        return self.kf_TruthT

class XvalKfold():
    '''
    Calculate k-fold cross validation.
    '''
    def __init__(self, classifier, args):
        '''
        Initialize class instance with classifier model and command line arguments.
        '''
        self.k = args.kfold
        self.shuffle = args.kfold_shuffle
        self.kf = model_selection.KFold(n_splits=self.k, shuffle=self.shuffle)
        self.kfstats = kfold_stats(self.k)                              # initialize stats accumulator class
        self.classifier = classifier

    def kfold(self, X, t):
        '''
        Perform a k-fold cross validation using the features obtained by selecting
        the nbest peptides after sorting by p-value from the t-test conducted
        by PeptidesBySubject.calc_tTest at the beginning of the program (before
        the main loop).
        '''
        n = 0
        myTrue = np.asarray([], dtype=np.int)                           # required ?? : Yes
        myPred = np.asarray([], dtype=np.int)
        for itrain, itest in self.kf.split(X):
            cpa.eprint(".", end=''),
            if (len(set(t[itrain])) < 2):
                cpa.eprint("WARNING:  KFold cross-validation set {:d}/{:d} contains only a single target class".format(n,self.k))
                cpa.eprint("          Skipping this iteration.")
            else:
                self.classifier.fit(X[itrain], t[itrain])
                prediction = self.classifier.predict(X[itest])
                myPred = np.hstack((myPred, prediction))                # calc Truth Table at each iteration and calc stats
                myTrue = np.hstack((myTrue, t[itest]))
                self.kfstats.add_record(t[itest], prediction)           # add to accumulator
            n += 1
        cpa.eprint("Done.")
        summaryA = self.kfstats.summary()
        return myTrue, myPred, summaryA

    def kfold_ttest(self, S, t, nbest):
        '''
        Recalculate nbest peptides based on the specific samples chosen for the
        training set in cross validation.
        '''
        nfactors = np.empty((2), dtype=np.int)                  # S[n] are subsets of the master data matrix, each containing data
        nfeatures = len(S[0])                                   # for a single factor, n; shape = (nPeptides X nSamples for factor n)
        for n in [0,1]:                                         # for each factor, n, count the number of columns in S
            nfactors[n] = len(S[n][0])                          # count the number of samples
        S = np.hstack((S[0], S[1]))                             # join matricies such that factor rows are merged
        ivals = S.transpose()                                   # S shape = (nfac[0] + nfac[1], nfeatures)

        n = 0
        myTrue = np.asarray([], dtype=np.int)                   # required ?? : Yes
        myPred = np.asarray([], dtype=np.int)
        for itrain, itest in self.kf.split(ivals):              # kfold iterator, each loop produces a new training and test set
            itrainN          = len(itrain)                      # number of elements in array
            itestN           = len(itest)                       #  "
            training_sample  = ivals[itrain]                    # training samples: t-test will be calc'd using this
            training_targets = t[itrain]
            test_sample      = ivals[itest]                     # these will be used in kfold xval
            test_targets     = t[itest]
            bool_fac = np.empty((2,itrainN), dtype=np.bool)     # boolean array
            factor = []
            for x in [0,1]:
                bool_fac[x] = (t[itrain] == x)
                factor.append(training_sample[bool_fac[x]])     # factor[x] shape: nSamples X nFeatures

            if (len(set(t[itrain])) < 2):
                cpa.eprint("WARNING:  KFold cross-validation set {:d}/{:d} contains only a single target class".format(n,k))
                cpa.eprint("          Skipping this iteration.")
            else:
                ttests = stats.ttest_ind( a         = factor[0],
                                          b         = factor[1],
                                          axis      = 0,
                                          equal_var = False).pvalue
                ttestSort_indexArray = np.argsort(ttests)       # sort feature indices based on p-value
                self.classifier.fit(ivals[itrain][:,ttestSort_indexArray[:nbest]], t[itrain])
                prediction = self.classifier.predict(ivals[itest][:,ttestSort_indexArray[:nbest]])
                myPred = np.hstack((myPred, prediction))                # append predictions to myPred, used for overall stats
                myTrue = np.hstack((myTrue, t[itest]))
                self.kfstats.add_record(t[itest], prediction)           # add to accumulator for stats on CC
            n += 1
            cpa.eprint(".", end=''),
        cpa.eprint("Done.")
        summaryA = self.kfstats.summary()
        return myTrue, myPred, summaryA

def xval_lpo(model, X, t, p, args):
    '''
    Cross validation using leave-P-out protocol.  With medium to large datasets, P=1, is the
    only practical setting, doing one iteration for each element (E) in dataset. For P > 1,
    the number of iterations equals E choose P.
    '''
    lpo = model_selection.LeavePOut(p)
    myTrue = np.asarray([], dtype=np.int)
    myPred = np.asarray([], dtype=np.int)
    i = 0
    for itrain, itest in lpo.split(X):
        cpa.eprint(".", end='')
        model.fit(X[itrain], t[itrain])
        myPred = np.hstack((myPred, model.predict(X[itest])))
        myTrue = np.hstack((myTrue, t[itest]))
        i += 1
    cpa.eprint("Done.")
    return myTrue, myPred
