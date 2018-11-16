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
# v1.06 remove (nonfunctional) histogram function to avoid issues with plotting libraries
# Mon Aug 20 10:26:21 MST 2018

'''
pepArray.py is the executable program, requiring access to createPepArray.py and xValStats.py as supporting libriaries.
The program can be executed using the input and output file names followed by a long series of parameters.  In practice,
it is easier to use a parameter file or some combination there of:

    pepArray.py @parameter_file.par
    
Note: '@' is a flag at Python interprets as a signal that what follows is a file name that should be parsed like
a series of command line arguments.  It is not part of the file name.  Such a file might contain:

    BCdod12merGreenPT2.txt                                  # input matrix file
    BCdod12merGreenPT2_t7.txt                               # output file name
    --nbest_start  2                                        # evaluate classifier performance starting with top two peptides (sorted by t-test P-value)
    --nbest_nSteps 12                                       # evaluate classifier 12 times using an increasing number of peptides
    --nbest_logBase 2                                       # increase number of peptides in each step by 2-fold (default)
    --xval_methods recall kfold                             # cross-validation methods: kfold, loo (leave one out), recall (of training data)
    --kfold 5                                               # k, when using k-fold cross-validation; 5 => leave 20% out for testing and average the five cases
    --log_transform                                         # take natural log of all intensity data before processing
    --logT_pcount 0.01                                      # pseudocount for log transform (to prevent taking log of zero)
    --rescale_enMasse
    --rescale_axis 1
    --svm_kernel linear                                     # SVM kernel function: linear, polynomial, radial basis function
    --svm_C 1                                               # SVM's C parameter
    --peptidesNpvalues 1024
    --call_blinded BCdod12merGreenPT2_t7.blindCalls
    --omit_blinded
    --clobber
    --predictors control cancer
    --blind_factors blinded
    --dev_pickle BCdod12merGreenPT2_t6.pkl
    --random_seed 65

'''

import sys
import os as os
import numpy as np
import re
from sklearn import svm
import argparse
import createPepArray as cpa
import xValStats as xvs
import time
VERSION = 'v1.06'

def ln(arg, base):
    return np.log(arg)/np.log(base)

def write_datatable(dat, outfile):
    """
    """
    pass

def get_predictor_indices(predictors, dat):
    """
    Given an array of predictor strings (from args.predictors) consult dat and return the
    corresponding indices.
    """
    predIdx = []
    for predictor in predictors[0]:
        if predictor in dat.facName2num:
            predIdx.append(dat.facName2num[predictor])
        else:
            sys.exit('Unable to identify index for predictor factor {:s} in get_predictor_indices.'.format(predictor))
    return predIdx

def print_factor_list(dat):
    """
    print information on factor names, the factors they alias (if JSON), and counts for text output
    """
#   do headings first
    hline = "{:5s}  {:14s}  {:5s}".format("-----", "------", "-----")
    htext = "{:5s}  {:14s}  {:5s}".format("Index", "Factor", "Count")
    if dat.args.json:
        hline += "  {:12s}".format("------------")
        htext += "  {:12s}".format("Constituents")
    print("{:s}\n{:s}\n{:s}".format(hline, htext, hline), file=dat.txtout)

#   then data-- first the index, factor, count...
    for i in range(dat.nfactors):
        dtext = "{:5s}  {:14s}  {:5d}".format("[{:d}]".format(i), dat.factors[i],
                                                 len(dat.facIndex[i]))
#   then the factor's components, if any.
        if dat.args.json:
            # if dat.args.json['recode'][dat.factors[i]]:
            #     dtext += '  '
            #     for aFac in dat.args.json['recode'][dat.factors[i]]:
            #         dtext += aFac
            #         if aFac != dat.args.json['recode'][dat.factors[i]][-1]:     # if it's not the last one...
            #             dtext += ", "                                           # add a comma

            if dat.args.json[dat.factors[i]]:
                dtext += '  '
                for aFac in dat.args.json[dat.factors[i]]:
                    dtext += aFac
                    if aFac != dat.args.json[dat.factors[i]][-1]:     # if it's not the last one...
                        dtext += ", "                                           # add a comma
        print(dtext, file=dat.txtout)             # print the results with or without refactor data
    print(hline, file=dat.txtout)

def interpret_svm_gamma(args):
    """
    The svm_gamma parameter is unusual in that it can take on two variable types,
    string (for the default, "auto") and float (for user-provided values).  In order
    to provide the flexibility to use both types, non-default values must be validated
    as numeric and assigned to the parameter as a float.
    """
    if args.svm_gamma != 'auto':                                                    # if it is not the string default,
        if re.match(r'[+-]?(\d+(\.\d*)?|\.\d+)([Ee][+-]?\d+)?$',args.svm_gamma):    # check for valid number
            args.svm_gamma = np.float(args.svm_gamma)                               # then assign as float
        else:
            sys.exit("ERROR:  malformed number ({:s}) in --svm_gamma parameter.".format(args.svm_gamma))

def manage_data_sources(args):
    """
    On the initial run on a new dataset, the data is read directly from the correspond
    data matrix file.  By default, the data loading and processing up through creating
    peptides sorted by p-value (for all pairwise factor combinations) is "pickled" in
    a .pkl file from which that data will be read on subsequent invocations.
    """
    args.json = None
    dumpme = 0                                                                      # look for a reason to pickle the main data structure
    if args.dev_pickle and os.path.isfile(args.dev_pickle):                         # if there is a pickle file, read it in
        cpa.eprint('Reading --dev_pickle file, "{:s}".'.format(args.dev_pickle))
        dat = cpa.unpickle_file(args.dev_pickle)
        cpa.eprint('Done.')
    else:
        if args.recode:
            if args.recode[0] == '{':                                               # if --recode arg starts with '{' it looks like JSON...
                cpa.eprint('Parsing --recode argument, "{:s}", as JSON text.'.format(args.recode))
                args.json = cpa.read_json(args.recode)                              # read it as a JSON string
            elif os.path.isfile(args.recode):                                       # otherwise arg should be a filename, check first then...
                cpa.eprint('Reading --recode file, "{:s}".'.format(args.recode))
                args.json = cpa.read_json(args.recode)                              # read it as a JSON file
            else:
                cpa.eprint('ERROR: cannot open --recode argument, "{:s}", as a file for reading.'.format(args.recode))
                cpa.eprint('       It does\'t appear to be a valid JSON string either.  I give up...')
                sys.exit('Exiting.')
        dat = cpa.PeptideBySubject(args)                                            # begin processing master data matrix
        dumpme = 1                                                                  # now we have something to dump
    cpa.eprint("Data loading complete.")
    if dumpme:
        if args.dev_pickle:                                                         # don't pickle unless --dev_pickle + fname
            cpa.eprint("Creating dev_pickle file: {:s}...".format(args.dev_pickle))
            cpa.pickle_object(dat, args.dev_pickle)
            cpa.eprint("Done.")
            if args.dump_only:
                sys.exit('Exiting program by request.')
    return dat

def validate_nbestWRTnpeptides(args, dat):
    """
    Make sure that there are enough peptides to get nbest
    (mainly relevant to testing with toy datasets)
    """
    if(args.nbest > dat.npeptides):
        cpa.eprint("WARNING: nbest ({:d}) > number of peptides ({:d});".format(args.nbest, dat.npeptides))
        cpa.eprint("         nbest will be reset to match peptide count ({:d}).".format(dat.npeptides))
        args.nbest = dat.npeptides
    if(args.nbest_start > args.nbest):
        cpa.eprint("WARNING: nbest_start ({:d}) > nbest ({:d});".format(args.nbest_start, args.nbest))
        cpa.eprint("         nbest_start will be reset to match nbest ({:d}).".format(args.nbest))
        args.nbest_start = args.nbest

def calc_nbest_range(args, npeptides):
    """
    The main loop of the program tests predictor performance using a range of values
    for the top peptides ranked by p-value.  This function uses command-line parameters
    to determine whether the range should be linear or exponential ("log"), then picks
    values based on the distribution from nbest_start to nbest. ("nbest" has two meanings,
    the top n peptides at a given point and the largest n for the top n peptides.)
    If either nbest or nbest_start is omitted, nbest_nSteps will be used to calculate the
    opposing end point such that intermediate points will be "logical", i.e., multiples of
    nbest_nSteps (linear) or nbest_logBase raised to an integer power.
    """
#   calculate nbest_range
    if not args.nbest_start and not args.nbest:
        msg = 'ERROR: either --nbest_start <start> or --nbest <end> must be provided; '
        msg += 'type "{:s} --help" for more information.'
        sys.exit(msg.format(progname))
    if(args.nbest_linear):
        if args.nbest_start and args.nbest:
            pass
        elif not args.nbest_start:
            args.nbest_start = args.nbest - args.nbest_nSteps
        elif not args.nbest:
            args.nbest = args.nbest_nSteps - args.nbest_start
        else:
            sys.exit("ERROR: logical fall-through in nbest parameters in calc_nbest_range().")
        nbest_range = np.linspace(args.nbest_start, args.nbest, args.nbest_nSteps, dtype=np.int)
    else:
        if args.nbest_start and args.nbest:
            pass
        elif not args.nbest_start:
            args.nbest_start = args.nbest_logBase ** (ln(args.nbest, args.nbest_logBase) - args.nbest_nSteps + 1 )
        elif not args.nbest:
            args.nbest = args.nbest_logBase ** (args.nbest_nSteps - ln(args.nbest_start, args.nbest_logBase) + 1 )
        else:
            sys.exit("ERROR: logical fall-through in nbest parameters in calc_nbest_range().")

        nbest_range = np.logspace(ln(args.nbest_start, args.nbest_logBase),
                                  ln(args.nbest, args.nbest_logBase),
                                  args.nbest_nSteps,
                                  base=args.nbest_logBase, dtype=np.int)
#   then validate it
    nbest_range = np.unique(nbest_range)
    real_steps = len(nbest_range)
    if args.nbest_nSteps != real_steps:
        cpa.eprint("WARNING: requested nbest steps ({:d}) different from actual number of steps ({:d}).".format(
            args.nbest_nSteps, real_steps) )
    validated_nbest_range = []
    for r in nbest_range:
        if r < npeptides:
            validated_nbest_range.append(r)
        else:
            cpa.eprint('WARNING: nbest range step {:d} exceeds the total number of peptides ({:d})'.format(r, npeptides))
            cpa.eprint('         This step will be omitted from the range of feature counts evaluated.')
            break
    nbest_range = np.asarray(validated_nbest_range, dtype=np.int)        
    if args.debug:
        cpa.eprint("nbest_range: ", nbest_range)
    return nbest_range

def calc_dmatrix_rows(dat, args, nbest_range):
    """
    Calculate the total number of data rows in master output file based on the
    number of Factors, number of omitted blind factors, and number of nbest peptide
    values evaluated.
    """
    blind_factors = len(dat.blind_index)                    # number of blind factors (based on length of list containing their indices)
    skipped_factors = len(dat.skipFactor_index)
    used_factors = dat.nfactors                             # total number of factors in data structure
    if args.omit_blinded:                                   # if --omit_blinded, don't allocate memory in dmatrix for them
        used_factors -= blind_factors                       # (they will be skipped in main loops)
    used_factors -= skipped_factors
    drows = ( ( (used_factors * (used_factors-1) ) / 2)     # row count = number of half-matrix of pairwise factor combinations
        * len(nbest_range) )                                #               * actual nbest values evaluated
    return int(drows)

def prep_dmatrix(dat, args, fTruthT, nbest_range):
    """
    Create headings and allocate space for ndarray that will contain data for
    the master output file (.out.txt) and machine-readable equivalent (.txt).
    """
                                                            # create headings for dvec output (machine readable)
    dcol_headings_static = fTruthT.staticLabels             # these are the first three columns; they are only printed once per line
    xval_methods = args.xval_methods[0]                     # argparser returns a list of lists (why?)
    xval_methods.sort()                                     # order MUST correspond to the way the arrays are ordered in tvecC
    nxvals = len(xval_methods)                              # count the number of cross validation methods requested, incl recall (if used)
    dcol_headings = dcol_headings_static[:]                 # copy static headings to initialize list of all headings
    dcol_headings.extend([x + '_' + y for x in fTruthT.xvStats  # add the xval-specific headings, e.g.: tp_kfold, CC_lpo, acc_recall
                                      for y in xval_methods
                         ])
                                                            # calculate requirements for output file data struct (dmatrix)
    drows = calc_dmatrix_rows(dat, args, nbest_range)       # pairwise factor combinations * nbest values
    dcols = ( (len(fTruthT.xvStats) * nxvals )              # (tp, fp, fn, tn, Sn, Sp, acc, CC) * xval_method_count
        + len(dcol_headings_static) )                       # + f1, f2, nbest;  3 + (TruthTableLength(=8) * n_methods)
    if dcols != len(dcol_headings):                         # if column count != heading count...
        sys.exit("prep_dmatrix(): calculated dcols ({:d}) != actual size ({:d}).".format( dcols, len(dcol_headings) ))
    dmatrix = np.zeros((drows, dcols), dtype=np.float)      # pre-dimension dmatrix to avoid fuss with joining ndarrays later
    return dmatrix, dcol_headings

def prep_ccstat_matrix(dat, args, fTruthT, nbest_range):
    """
    Create, allocate memory and zero out ndarray to hold kfold CC statistics
    (consisting of mean, stdev, min and max of all the CCs produced in kfold
    cross validation run, plus i, j, nbest to indicate the specific run).
    Return ndarray along with list of ccStat file headings.
    """
    drows = calc_dmatrix_rows(dat, args, nbest_range)
    ccStat_headings  = fTruthT.staticLabels[:]              # these start the same: f12, f2, nbest
    ccStat_headings += ['mean', 'stdev', 'min', 'max']      # then add the stat labels
    ccStats = np.zeros((drows, len(ccStat_headings)))       # pre-dimension kfold correlation coefficient statistics:
                                                            # i, j, nbest + mean, std, min, max
    return ccStats, ccStat_headings

def make_predictions_on_blinded(predictive_parameters, unblinded):
    """
    Using a trained classifier model, make calls on blinded factors and merge with unblinded factor
    names, if available. Implemented w/o using numpy where possible.
    """
    ( args, dat, fitClassifier, scaler_model, nbest_range ) = predictive_parameters
    if args.call_blinded == None:                               # if args don't identify a file for blind calls
        return None                                             # there's nothing to do
    if len(dat.blind_index) == 0:                               # even if it is defined, there may be no data to support it
        return None                                             # and again there's nothing to do
    predictor_idx = get_predictor_indices(args.predictors, dat) # parse --predictors and return their indices
    predictor_idx.sort()                                        # the factor combination whose data was used to parameterize model
    (i, j) = predictor_idx
                                                                # loop over blind_index to produce separate predictions for each nbest value
    pvec = []                                                   # table of predicted values for blinded samples, incl. header info
    blindFacIndex = []                                          # indices of whatever samples are considered "blind"
    for k in dat.blind_index:                                   # loop over the indices of the blind things...
        blindFacIndex += list(dat.facIndex[k])                  # adding their facIndices to the list (needs to be a list for += to work line append)
    for rAttrib in list(dat.hRowDataDict.keys()):                            # row attributes
        rLabel = dat.rowAttrib2label[rAttrib]                # row label
        pv = np.insert(dat.hRowDataDict[rAttrib][blindFacIndex], 0, rLabel)
        pvec.append(pv)
    ivals = dat.getI4factor(dat.blind_index)                    # get intensity values for all blind factors (input: List)
    for nbest in nbest_range:                                   # loop through pre-calculated nbest points
        clf = fitClassifier[i][j][nbest]                        # retrieve saved classifier params, incl. support vectors & coefficients
        sbt = []
        for x in range(len(ivals)):                             # for each factor (there could be multiple "blinded" factors)
            S = ivals[x][dat.indexArray[:,i,j]]                 # ivals sorted according to t-tests for factor pair: i, j
            sbt.append(cpa.get_nbest(S, nbest).transpose())     # sbt[f1,f2][subj1..subjN][peptides]
        X = np.vstack((sbt[:]))                                 # input to classifier: intensity values for nbest peptides for blinded samples
        if args.rescale_perPair:                                # if doing intra-loop rescaling ...
            X = scaler_model[i][j][nbest].transform(X)          # apply the transformation corresponding to the position in the loop defined by factor i & j + nbest
        predictionsOnBlinded = np.asarray(clf.predict(X), dtype=np.int)     # apply the classifier to the input, then convert to ndarray
        pob_wNbestHeading = np.insert(predictionsOnBlinded, 0, nbest)       # then label the predictions with nbest
        pob_wNbestHeading = np.asarray(pob_wNbestHeading, dtype=np.str)     # convert to string to match labels (needed for homogeneous ndarray)
        pvec.append(pob_wNbestHeading)                                      # append the labeled row of predictions to output table
    ndpvec = np.asarray(pvec)                                   # convert table to ndarray
    tpvec = ndpvec.transpose()                                  # and transpose so labels are at top and Samples are in rows
    if type(unblinded) == type(None):                           # since unblinded is an array, need to check for data at variable type level
        cpa.writeTSVfile(tpvec, args.call_blinded)              # write out predicted affected status w/o unblinded factor info
        return None
    else:                                                       # if unblinded data is available ...
        fac_col_ub = np.where(unblinded[0] == 'Factor')[0]      # get the index of the column containing factor data
        samp_col_ub = np.where(unblinded[0] == 'Sample')[0]     # get the index of the column containing Subject IDs
        fac_col_tp = np.where(tpvec[0] == 'Factor')[0]          # col index containing 'Factor'
        tpvecd = np.delete(tpvec, fac_col_tp, axis=1)           # remove Factor column (which contains only "blind*" for all subjects;
                                                                # so there will be only one 'Factor' col when merged w/unblinded)
        unblinded[1:,:].sort(axis=0)                            # sort lines after header (in place)
        shift = np.where(tpvecd[0] == 'Sample')                 # get index of "Sample" column
        if len(shift) == 1:
            tpvecd = np.roll(tpvecd, len(tpvecd[0]) - shift[0], axis=1)  # rotate columns until "Sample" is in first column
        else:
            sys.exit("ERROR: non-unique index for 'Sample' in tpvecd; shift = {:s}.".format(repr(shift)))
        tpvecd[1:,:].sort(axis=0)                               # sort predictions by "Sample" column (in place), leaving header row alone
        fac_col_ub = np.hstack((fac_col_ub, samp_col_ub))       # add "sample" col from unblinded so we can see the IDs from both sources
        tpvec_unblinded = np.hstack((tpvecd, unblinded[:,fac_col_ub])) # merge header and calls with unblinded factors
        cpa.writeTSVfile(tpvec_unblinded, args.call_blinded)    # write out predicted affected status w/o unblinded factor info
        return(tpvec_unblinded)                                 # return merged arrays

def score_blinded_samples(pob, dat, nbest_range, args, fTruthT):
    """
    Given matrix consisting of transposed header from data matrix (input file), calls using feature counts
    listed in nbest_range, plus unblinded factors (if available), recode unblindeds according to JSON recode
    rules (if applicable), then score performance using truth_table().
    """
#   recode unblinded factors to the more general names used to define groups for classifier (if JSON file was used for remapping)
    facidx = np.where(pob[0] == 'Factor')                               # get index of Factor column in blindCalls matrix
    if hasattr(dat.args, 'json') and type(dat.args.json) != type(None): # if JSON arg, unblinded factors need to be recoded
        for newfac in dat.args.json['recode'].keys():               # loop over factor "alias", recoded names of Factors from .mat file
            for i in range(1,len(pob)):                                 # scan by index down blindCalls matrix, skipping row 0 (header)
                if pob[i][facidx] in dat.args.json['recode'][newfac]:   # if the unblinded factor is in list of names assoc w/current recoded Factor name
                    pob[i][facidx] = newfac                             # map original factor to its new name
    f2i = dict()                                                        # factor-to-index
    f2c = dict()                                                        # factor-to-call
    f2i = dat.facName2num
    args.predictors[0].sort()               # this ensures that the named predictors are in line with everything else
    predict = args.predictors[0]
    if f2i[predict[0]] < f2i[predict[1]]:
        f2c[predict[0]] = 0                 # originally assigned 0
        f2c[predict[1]] = 1                 # originally assigned 1
    else:
        cpa.eprint('ERROR: indices of --predictors are out of order, {:d} > {:d}.'.factor(f2i[predict[0]], f2i[predict[1]]))
    myTruth = np.squeeze(np.array(pob[1:, facidx], copy=True))
    for i in range(len(predict)):
        myTruth[np.where(myTruth==predict[i])] = f2c[predict[i]]
    fTruth = np.asarray(myTruth, dtype=np.float)
#   score calls
    pred = dict()
    i, j = f2i[predict[0]], f2i[predict[1]]
    fdata = dict()
    for nbest in np.array(nbest_range, dtype=np.str):
        nbestidx = np.where(pob[0] == nbest)
        myTarget = np.squeeze(np.asarray(pob[1:,nbestidx], dtype=np.float))
        pred[nbest], fdata[nbest] = fTruthT.ftts(fTruth, myTarget, 'pred_{:s}'.format(nbest), i, j, nbest)
        print(pred[nbest], file=dat.txtout)

def find_duplicates(myList):
    """
    Find duplicate items given a list or other iterable collection.
    If every item is unique, return the input list as a set.
    """
    mySet = set(myList)
    if len(mySet) < len(myList):
        cpa.eprint('WARNING: input list is not unique.')
        dups = [x for x in mySet if myList.count(x) > 1]
        cpa.eprint('         Duplicate entries:\n{:s}.'.format(repr(dups)))
        sys.exit('Exiting.')
    else:
        return mySet

def read_unblinded(unblind_file, blinded_samples):
    """
    Read tab-delimited file containing a two-column table consisting of header row,
    "Sample"<tab>"Factor", followed by rows containing Sample IDs and unblinded
    Factors representing actual (known) affected status, infectious agent, etc.,
    depending on study.  The file must contain IDs for all samples blinded in
    the original input matrix but may contain other samples (e.g., all samples
    and factors, or other extraneous data).  Sample IDs not among the blinded
    are simply ignored.
    The function returns a numpy ndarray consisting of the original header followed
    by the rows (containing Sample ID and unblinded Factor) corresponding to
    samples in which Factor == the argument of command-line parameter --blind_factors
    in the original master data file (.mat); this argument is typically "blinded".
    """
    if unblind_file:                                                        # if file of unblinded factors is provided
        all_unblinded = np.loadtxt(unblind_file, dtype=np.str, delimiter='\t')
    else:                                                                   # if there is no such file 
        return(None)                                                        # set to None
    header, body = np.vsplit(all_unblinded, [1])                            # separate header from body of table
    id_idx = np.where(header[0] == 'Sample')                                # find column containing Sample IDs
    file_unblinded_ids = np.squeeze(body[:,id_idx])
    all_unblinded_ids  = np.unique(file_unblinded_ids)
    blinded_ids = np.unique(blinded_samples)
    missing_ids = np.setdiff1d(blinded_ids, all_unblinded_ids)
    if len(missing_ids) > 0:
        cpa.eprint('WARNING: not all blinded samples (n={:d}) are found in file, "--unblind {:s}".'.format(len(missing_ids),unblind_file))
        cpa.eprint('         Missing sample IDs are:\n{:s}.'.format(repr(missing_ids)))
    unblinded = []                                                          # init list container for confirmed unblinded samples
    unblinded.append(header[0])                                             # add header to nascent list of confirmed unblinded
    sample_ids = []
    for file_row in body:                                                   # loop over body of input table a row at a time
        if file_row[id_idx] in blinded_samples:                             # if row's Sample ID is among the blinded samples
            unblinded.append(file_row)                                      # add the row to the unblined list
    np_unblinded = np.asarray(unblinded)                                    # convert list to numpy array and return
    return np_unblinded

def prep_blinded(args, dat):
    """
    Save index array for factors identified as "blinded" by --blind_factors in
    dat.blind_index.  Return blinded sample IDs.
    """
    dat.blind_index = []
    dat.notBlind_index = []
    dat.skipFactor_index = []
    blind_factor_count = 0
    myFactors = [x for x in dat.factors]                        # dat.factors is an ndarray
    if args.skip_factors:
        skip_factors = args.skip_factors[0]
    if args.blind_factors:                                      # if supplied with the names of the blind factors on cmdline
        blind_factor_names = args.blind_factors[0]              # use them
    else:
        blind_factor_names = []
    for fac in dat.factors:
        if fac in args.skip_factors[0]:
            dat.skipFactor_index.append(myFactors.index(fac))
        elif fac in args.blind_factors[0]:
            dat.blind_index.append(myFactors.index(fac))
        else:
            dat.notBlind_index.append(myFactors.index(fac))
    cpa.eprint("Blind_index      = {:s}.".format(repr(dat.blind_index)))
    cpa.eprint("notBlind_index   = {:s}.".format(repr(dat.notBlind_index)))
    cpa.eprint("skipFactor_index = {:s}.".format(repr(dat.skipFactor_index)))
    if len(blind_factor_names) == 0:
        return []
    blind_sample_index = []
    for fac in dat.blind_index:
        blind_sample_index += [x for x in dat.facIndex[fac]]
    blind_sampleIDs = dat.hRowDataDict['_sample'][blind_sample_index]
    return blind_sampleIDs
    
def write_txtout_preamble(txtout, args, cmdline):
    """
    Print a fancy header containing command-line arguments and parameters.
    """
    MAX_WIDTH = 45
    maxWidthKey = 0
    maxWidthVal = 0
    keys2print = []
    for k,v in sorted(vars(args).items()):
        if k != 'json':
            width = len(k)
            if width > maxWidthKey:
                maxWidthKey = width
            width = len(repr(v))
            if width > maxWidthVal:
                maxWidthVal = width
            keys2print.append(k)
    if maxWidthKey > MAX_WIDTH:
        cpa.eprint("maxWidthKey > {:d}: {:d}.".format(MAX_WIDTH, maxWidthKey))
    if maxWidthVal > MAX_WIDTH:
        cpa.eprint("maxWidthVal > {:d}: {:d}.".format(MAX_WIDTH, maxWidthVal))
    myline = '-' * (maxWidthKey + maxWidthVal)
    myFormat = '{{:<{:d}s}}{{:>{:d}s}}\n'.format(maxWidthKey, maxWidthVal)
    filling = myFormat.format(args.progname, args.timestamp)
    print_string = myline + '\n' + filling + myline
    print(print_string, file=txtout)
    myArgs = str()
    myFormat = '{{:<{:d}s}}{{:>{:d}s}}'.format(maxWidthKey, maxWidthVal)
    for k,v in sorted(vars(args).items()):
        if k != 'json':
            print(myFormat.format(k, repr(v)), file=txtout)
    print(myline + '\n', file=txtout)


####################################################################################################
class argParse(argparse.ArgumentParser):
    """
    Subclass of argparse.ArgumentParser which allows overloading of convert_arg_line_to_args()
    so that it can parse individual command line parameters with their arguments when each appears
    on a single line in a text/parameter file.  Provision is also made so that lines beginning
    with '#' are ignored as comment lines.
    """
    def convert_arg_line_to_args(self, arg_line):
        arg_line, sep, remainder = arg_line.partition('#')
        arglines = arg_line.split()
        if len(arglines) == 0:
            return []
        else:
            return arglines


####################################################################################################
# main()
# parse arguments from the command line or a parameter file prefixed with a "@" character
####################################################################################################
def main():
    """
    Command line argument parsing and main program loop happen here.
    SVM Parameters:
        C       trades off misclassification of training examples against simplicity of the decision surface.
                A low C makes the decision surface smooth, while a high C aims at classifying all training
                examples correctly.
        gamma   defines how much influence a single training example has. The larger gamma is, the
                closer other examples must be to be affected.
    """
    program_start = time.time()
    progname = sys.argv[0]
    qualified_progname = progname
    if progname.rfind('\\'):
        progname = progname[progname.rfind('\\')+1:]
    if progname.rfind('/'):
        progname = progname[progname.rfind('/')+1:]
    sys.argv[0] = progname
    json_file_format_description = '{"file_format":{"delimiter":"\\t","comments":"#"},"header":{"Factor":"_factor","Sample":"_sample"}}'
    command_line = str()
    for x in sys.argv:
        command_line += x + ' '
    del x
    
    parser = argParse(
        description             =   'Parse a tabular input file containing Peptides x Samples, ' +
                                    'optionally rescale and normalize intensities, calculate ' +
                                    't-tests for all factor pairs, rank by p-value, then use ' +
                                    'the top "nbest" peptides to train classifier(s) and evaluate ' +
                                    'performance by recall and cross validation. Blinded samples, ' +
                                    'if present, can be predicted using a classifier trained on a ' +
                                    'a specific predictor factor pair.',
        prog                    =   progname,
        usage                   =   '%(prog)s @parameter_file.par',
        fromfile_prefix_chars   =   '@'
    )
    parser.add_argument('infile',            nargs='?', type=str,
                                             help="input file" )
    parser.add_argument('outfile',           nargs='?', type=str,
                                             help="output file" )
    parser.add_argument('--version',         action='version', version=VERSION)
    parser.add_argument('--clobber',         action='store_true',
                                             help='allow existing outfile to be overwritten' )
    parser.add_argument('--file_format',     type=str, default=json_file_format_description,
                                             help='JSON string or name of JSON file ' +
                                                  'describing master data file format' )
    parser.add_argument('--recode',          type=str,
                                             help='JSON string or name of JSON file ' +
                                                  'describing how to recode ' +
                                                  'factor names (to rename or merge factors)')
    parser.add_argument('--random_seed',     type=int, default=program_start,
                                             help='integer to seed random number generator ' +
                                                  '(between 0 and 2**32-1, inclusive')

    # debugging-related
    parser.add_argument('--debug',           action='store_true',
                                             help='enable debugging output')
    parser.add_argument('--random_ttests',   action='store_true',
                                             help='populate t-tests table with random values rather ' +
                                                  'than actually calculating them')
    parser.add_argument('--dev_pickle',      type=str,
                                             help='save PeptideBySubject object containing samples, ' +
                                                  'peptides and unscaled intensities for quick restart')

    # data caching and dumping
    parser.add_argument('--dump_only',       action='store_true',
                                             help="stop after creating pickle file")
    parser.add_argument('--peptidesNpvalues',type=int, default=0,
                                             help='dump the indicated number of peptides ranked ' +
                                                  'by p-values for each factor permutation')
    parser.add_argument('--skip_factors',    nargs='+', action='append', default=None, required=False,
                                             help='factor(s) to ignore')

    # factor definition
    parser.add_argument('--add_factors',     nargs='+', action='append',
                                             help='append additional parameter(s) to primary ' +
                                                  'Factor name')

    # nbest parameters (number of features)
    parser.add_argument('--nbest',           type=int,
                                             help='number of top-scoring peptides to select for ' +
                                                  'use in classifier')
    parser.add_argument('--nbest_start',     type=int,
                                             help='starting point for the evaluation of multiple ' +
                                                  'values for number of peptides used in classifier')
    parser.add_argument('--nbest_logBase',   type=int, default=2,
                                             help='base for panning nbest')
    parser.add_argument('--nbest_nSteps',    type=int, default=5,
                                             help="pan nbest with --nbest_steps between start and end")
    parser.add_argument('--nbest_linear',    action='store_true',
                                             help='pan nbest over a linear (rather than log) scale')

    # preprocessing/rescaling
    parser.add_argument('--log_transform',   action='store_true',
                                             help='take ln_e of intensity values (prior to rescaling)')
    parser.add_argument('--logT_pcount',     type=np.float, default=0.01,
                                             help='(pseudocount) value added to intensity values ' +
                                                  'prior to log transformation')
    parser.add_argument('--rescale_enMasse', action='store_true',
                                             help='rescale features en masse to zero mean and ' +
                                                  'unit variance')
    parser.add_argument('--rescale_perPair', action='store_true',
                                             help='rescale features for each factor pair to ' +
                                                  'zero mean and unit variance')
    parser.add_argument('--rescale_axis',    type=int, choices=[0,1], default=1,
                                             help='rescale by sample (axis=0) or peptide (axis=1)')

    # cross validation methods
    parser.add_argument('--xval_methods',    nargs='+', choices=['lpo', 'kfold', 'recall'],
                                             action='append', required=False,
                                             help='select one or more methods for cross validation')
    # k-fold cross-validation parameters
    parser.add_argument('--kfold',           type=int, default=4,
                                             help='set "k" when using k-fold cross validation')
    parser.add_argument('--kfold_shuffle',   action='store_true', default=True,
                                             help='shuffle data before splitting into k-fold batches')
    parser.add_argument('--kfold_ttest',     action='store_true', default=False,
                                             help='calculate t-test and choose peptides on each kfold ' +
                                                  'iteration')
    # leave-p-out parameters
    parser.add_argument('--lpo_p',           type=int, default=1,
                                             help='set "p" when using "leave-p-out" cross validation')

    # choose classifier type(s)
    parser.add_argument('--classifier',      type=str, choices=['svm', 'nb'], required=False, default='svm',
                                             help='select classifier method' )
    # SVM classifier parameters
    parser.add_argument('--svm_C',           type=float, default=1.0,
                                             help='SVM learning parameter C')
    parser.add_argument('--svm_tol',         type=float, default=1e-3,
                                             help='tolerance for stopping criterion')
    parser.add_argument('--svm_maxIter',     type=int, default=-1,
                                             help='SVM maximum number of training iterations ' +
                                                  '(-1 => unlimited (default))')
    parser.add_argument('--svm_gamma',       type=str, default='auto',
                                             help='SVM learning parameter gamma')
    parser.add_argument('--svm_kernel',      type=str, default="linear",
                                             choices=['linear','poly','rbf','sigmoid','precomputed'],
                                             help='SVM kernel type')
    parser.add_argument('--svm_polyDegree',  type=int, default=3,
                                             help='polynomial degree for SVM kernel function ' +
                                                  '(ignored by other kernels)')
    # Naive Bayes classifier parameters (none yet)
    
    # blinded samples and factor pairs for blind prediction
    parser.add_argument('--omit_blinded',    action='store_true',
                                             help='omit "blinded" samples from training and testing')
    parser.add_argument('--call_blinded',    type=str,
                                             help='call blinded samples and write results to file')
    parser.add_argument('--predictors',      nargs='+', action='append', required=False,
                                             help='factor-pair used for prediction of blinded samples')
    parser.add_argument('--blind_factors',   nargs='+', action='append', default=None,
                                             help='factor name(s) associated with blinded samples ' +
                                                  '(e.g., "blinded")')
    parser.add_argument('--unblind',         type=str, default=None,
                                             help='file containing unblinded factors in ' +
                                                  'one-row-per-sample format ' +
                                                  '(with headings: "Sample" and "Factor")')
    parser.add_argument('--holdouts',        nargs='+', action='append', required=False,
                                             help='factors on which to do recall only, ' +
                                                  'no cross-validation')
    
    #   visualization
    # parser.add_argument('--plot_nbest',      nargs='+', action='append', default=None,
    #                                          help='plot histograms of intensities for n-best peptides ' +
    #                                               'using python range() semantics: provide up to ' +
    #                                               '3 values; populations are based on factor names ' +
    #                                               'provided in --predictors')
    args = parser.parse_args()
    if type(args.infile) is None:
        sys.exit("Positional argument 'infile' is required.")
    if type(args.outfile) is None:
        sys.exit("Positional argument 'outfile' is required.")

    if args.classifier == 'nb':
        from sklearn.naive_bayes import GaussianNB
    args.timestamp = cpa.get_date_string()
    (root, sep, ext) = args.outfile.partition('.')
    textout = root + sep + "out.txt"
    txtout = open(textout, 'w')

    if args.skip_factors == None:
        args.skip_factors = [[]]
    if args.blind_factors == None:
        args.blind_factors = [[]]
    interpret_svm_gamma(args)
    dat = manage_data_sources(args)
    dat.outfileRoot = root                              # save outfile root with PeptideBySubject so it can create fnames autonomously
    dat.txtout = txtout                                 # filehandle for .out.txt (main human-readable output table)
    args.progname = dat.progname = progname             # assign the program name to PeptideBySubject
    write_txtout_preamble(txtout, args, command_line)
    blind_subIDs = prep_blinded(args, dat)
    unblinded = read_unblinded(args.unblind, blind_subIDs)
    if args.log_transform:
        dat.ivals_logTransformed = dat.logTransform_ivals(dat.ivals_unscaled, args.logT_pcount)
        dat.ivals, mass_scaler = dat.rescale_ivals(dat.ivals_logTransformed, axis=args.rescale_axis)
    else:
        dat.ivals, mass_scaler = dat.rescale_ivals(dat.ivals_unscaled, axis=args.rescale_axis)
    dat.calc_tTest()
    dat.sortPeptides()
    if args.peptidesNpvalues:
        dat.dumpPeptidesNpvalues(args.peptidesNpvalues)                     # append ranked pvalues to .out.txt

    nbest_range = calc_nbest_range(args, dat.npeptides)
    validate_nbestWRTnpeptides(args, dat)                                   # ensure nbest does not exceed the number of peptides
    print_factor_list(dat)
    fTruthT = xvs.FormatTruthTable()
    dmatrix, dcol_headings   = prep_dmatrix(dat, args, fTruthT, nbest_range)
    ccStats, ccStat_headings = prep_ccstat_matrix(dat, args, fTruthT, nbest_range)
    col_lines, ttDFF = fTruthT.print_header(txtout)
    
    dmat_row = 0                                                            # index for dmatrix rows
    scaler_model = dict()
    fitClassifier = dict()
    cpa.eprint("Beginning main loops...")
    for i in range(dat.nfactors):
        if i in dat.blind_index and args.omit_blinded:
            continue
        if i in dat.skipFactor_index:
            continue
        scaler_model[i] = dict()
        fitClassifier[i] = dict()
        for j in range(i+1, dat.nfactors):
            if j in dat.blind_index and args.omit_blinded:
                continue
            if j in dat.skipFactor_index:
                continue
            scaler_model[i][j] = dict()
            fitClassifier[i][j] = dict()
            S = dat.getI4factor([i,j])                                      # S[f1,f2][peptides][subj1..subjN] = intensity
            # plot_histogram(S, args)
            for nbest in nbest_range:                                       # loop through pre-calculated nbest points
                dvec = np.asarray([i, j, nbest], dtype=np.float)            # float because performance data, added later, is float
                sbt = []; y = []
                for x in range(len(S)):
                    if x > 1:
                        sys.exit("There should never be more than two elements in S!  len(S) = {:d}.".format(len(S)))
                    sbt.append(cpa.get_nbest(S[x], nbest).transpose())      # sbt[f1,f2][subj1..subjN][peptides]
                    y.append(np.full(len(sbt[x]), x, dtype=np.float))       # y[f1,f2][subj1..subjN] = [0|1]
                X = np.vstack((sbt[0], sbt[1]))
                target = np.hstack((y[0], y[1]))
                if args.rescale_perPair:
                    X, scaler_model[i][j][nbest] = cpa.rescale_ivals(X)
                clf = svm.SVC(kernel=args.svm_kernel, gamma=args.svm_gamma, # default classifier method, SVM
                              C=args.svm_C, max_iter=args.svm_maxIter,
                              tol=args.svm_tol, degree=args.svm_polyDegree)
                if args.classifier == 'nb':
                    clf = GaussianNB()
                elif args.classifier == 'svm':
                    pass
                else:
                    cpa.eprint("classifier defaulting to SVM")
                    
                rarray = list()                                             # need to initialize for case when 'recall' *not* run
                if 'recall' in args.xval_methods[0]:
                    fitClassifier[i][j][nbest] = clf.fit(X, target)         # fit model to data & save model params
                    (rstr, rarray) = fTruthT.ftts(target, clf.predict(X), 'recall', i, j, nbest)
                    print(rstr, file=txtout)
                larray = list()
                if 'lpo' in args.xval_methods[0]:
                    cpa.eprint("LPO cross-validation for Factors[{:d}][{:d}], nbest={:d}: ".format(i, j, nbest),end='')
                    (myTrue, myPred) = xvs.xval_lpo(clf, X, target, args.lpo_p, args)
                    (lstr, larray) = fTruthT.ftts(myTrue, myPred, 'xval_lpo', i, j, nbest)
                    print(lstr, file=txtout)
                karray = list()
                if 'kfold' in args.xval_methods[0]:
                    cpa.eprint("{:d}-fold cross-validation for Factors[{:d}][{:d}], nbest={:d}: ".format(
                        args.kfold, i, j, nbest), end='')
                    xvkf = xvs.XvalKfold(clf, args)                     # get instance of kfold Xvalidator
                    if args.kfold_ttest:                                # kfold xval with t-test calculated before each iteration
                        myTrue, myPred, kfSumA = xvkf.kfold_ttest(S, target, nbest)
                    else:                                               # original kfold which uses the same features for each
                        myTrue, myPred, kfSumA = xvkf.kfold(X, target)
                    kstr, karray = fTruthT.ftts(myTrue, myPred, 'xval_kfold', i, j, nbest)
                    print(kstr, file=txtout)

                dvecC = np.concatenate((dvec, karray, larray, rarray))  # collect performance data
                dmatrix[dmat_row] = dvecC                               # for later file output to
                if args.xval_methods[0].count('kfold'):                 # machine-readable outfile
                    ccSrow = np.hstack((dvec, kfSumA[:,-1]))            # mean, std, min, max for kfold CC
                    ccStats[dmat_row] = ccSrow
                dmat_row += 1                                           
            print(col_lines, file=txtout)
    del(clf, S, X, sbt, y, dvec, dmat_row)
    cpa.writeTSVfile(dmatrix, args.outfile, header=dcol_headings)
    if args.xval_methods[0].count('kfold'):
        cpa.writeTSVfile(ccStats, args.outfile + 'CC', header=ccStat_headings)
    predictive_parameters = [args, dat, fitClassifier, scaler_model, nbest_range]
    pob = make_predictions_on_blinded(predictive_parameters, unblinded)
    if type(pob) != type(None):                                         # if we know the affected status of the blinded samples
        score_blinded_samples(pob, dat, nbest_range, args, fTruthT)     # calculate a truth_table-type score
    
    txtout.close()
    
    cpa.eprint("Done.")
    extimeSec = time.time() - program_start
    extimeMin = extimeSec // 60
    extimeSec = extimeSec %  60
    cpa.eprint('Execution time: {:4d} min {:5.2f} sec'.format(int(extimeMin), extimeSec))


####################################################################################################
if __name__ == "__main__":
    main()

