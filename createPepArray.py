#!/usr/bin/env python3
"""
Pathology Prediction by Immunosignature
(c) Eric E. Snyder, Ph.D. (2017)
eesnyder@eesnyder.org
"""
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

import sys
import os
import errno
import json
import scipy.stats as stats
import csv
import numpy as np
import pickle
import re
from sklearn import preprocessing
import xValStats as xvs
import datetime as dt
from pytz import timezone
import pytz

ARRAY_ALPHA16_str = 'ADEFGHKLNPQRSVWY'
ARRAY_ALPHA16 = list(ARRAY_ALPHA16_str)
ARRAY_LENGTHS = (5, 20)
rePeptide = re.compile('[{:s}]{{{:d},{:d}}}$'.format(ARRAY_ALPHA16_str, ARRAY_LENGTHS[0], ARRAY_LENGTHS[1]))

reNumber = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
__all__ = ["PeptideBySubject", "pickle_object", "unpickle_file", "get_nbest",
           "writeTSVfile", "readTSVfile", "eprint"]


def eprint(*args, **kwargs):
    print(file=sys.stderr, *args, **kwargs)


def readTSVfile(infile):
    """
    Read data from an "excel-style" tab-delimited file into a list-of-lists
    structure (i.e., a matrix) and returns it.
    """
    eprint('Reading input file, "{:s}".'.format(infile))
    if os.path.isfile(infile):
        with open(infile) as csvfile:  # open input file
            LoL = []  # init list of lists
            reader = csv.reader(csvfile, 'excel-tab')  # create csv.reader on file handle
            try:
                for row in reader:  # read a row at a time
                    LoL.append(row)  # and append to LoL
            except csv.Error as error:
                sys.exit('file %s, line %d: %s' % (infile, reader.line_num, error))
        return (LoL)
    else:
        sys.exit('ERROR: Unable to open input file, "{:s}", for reading.'.format(infile))


def writeTSVfile(iTable, outfile, header=[]):
    """
    writeTSVfile( LoL, outfile, header=header )
    Given a list of lists, the name of an output file and an optional list
    containing column headings, write the data structure to the output file
    using the MS Excel conventions for dumping a spreadsheet as a
    tab-delimited text file.
    """
    LoL = list()
    if type(iTable) == type(dict()):  # if table is actually a dictionary of lists
        for k, v in iTable.items():  # iterate over key-value pairs
            LoL.append(np.insert(v, 0, k))  # prepending the key to list/value
    else:
        LoL = iTable

    listi = LoL.__iter__()  # define list iterator on transposed LoLs
    eprint("Number of output lines for writeTSVfile({:s}): {:d}.".format(outfile, len(LoL)))
    with open(outfile, 'w') as fp:  # open output file
        writer = csv.writer(fp, dialect='excel-tab')  # create csv.writer on file handle
        try:
            if any(header):
                writer.writerow(header)
            for row in LoL:  # for every row in the LoL
                writer.writerow(next(listi))  # call the iterator and write to file
        except csv.Error as error:
            sys.exit('file {:s}, line {:d}: {:s}.'.format(outfile, writer.line_num, error))
    return True


def write_tabdelimited_file(iTable, outfile):
    """
    Write an ordinary tab-delimited file.
    """
    if os.path.isfile(outfile):
        fp = open(outfile, 'w')
        for i in range(len(iTable)):
            print('\t'.join(iTable[i]), file=fp)
    else:
        sys.exit('ERROR: Unable to open output file, "{:s}", for writing.'.format(outfile))


def get_nbest(sample, nbest):
    """
    Given an ndarray of shape [x, z], return the first nbest rows, where nbest < x.
    """
    if sample.shape[0] < nbest:
        warning = 'Warning: get_nbest() argument, nbest = {:d}, is greater than the total number of peptides in the sample. '
        warning += 'Resetting nbest to the total number of available peptides ({:d}).'

        eprint(warning.format(nbest, sample.shape[0]))
        return sample
    else:
        return sample[:nbest, :]


def read_json(input_string):
    """
    If input_string looks like JSON text, parse the text directly, otherwise
    assume it is a filename and return deserialized data structure.
    """
    if input_string[0] == '{':
        myJSON = json.loads(input_string)
        return myJSON
    fp = open(input_string, 'rb')
    try:
        myJSON = json.load(fp)
    except ValueError:
        eprint("WARNING:  Cannot open JSON file using default encoding; trying 'latin-1'...")
        try:
            myJSON = json.load(fp, 'latin-1')
        except ValueError:
            sys.exit('Sorry, that did not work either; exiting.')
        else:
            eprint("encoding 'latin-1' worked.")
            return myJSON
    else:
        return myJSON


def unpickle_file(infile):
    """
    Load a data structure into memory from a file containing a pickled version of that structure,
    potentially created with pickle_object().
    """
    if os.access(infile, os.R_OK):
        try:
            fp = open(infile, 'rb')
        except IOError as e:
            if e.errno == errno.EACCES:
                return ("Can't open pickled file: {:s}.".format(infile))
            raise
            eprint("What happens after an IOError exception is raised (post-raise)?")
        else:
            with fp:
                return pickle.load(fp)
    else:
        sys.exit('ERROR: file not found, "{:s}".'.format(infile))


def pickle_object(obj, outfile):
    """
    Given an object (any Python data structure) and a file name, pickle the object and write to file.
    This operation can be reversed using unpickle_file()
    """
    with open(outfile, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def get_date_string():
    """
    Get UNIX-like date string for timestamp
    """
    mydt = dt.datetime
    now = mydt.now()
    mytz = timezone('UTC')
    mytz.localize(now)
    return now.strftime('%a %b %d %I:%M:%S %Z %Y')


class PeptideBySubject:
    """
    This class reads data in the format:
    Sample      samp-1     samp-2 ...
    Factor      fact-1     fact-2 ...
    Feature1    val-s1,f1  val-s2,f1  ...
    Feature2    val-s1,f2  val-s2,f2  ...
    """

    def __init__(self, args):
        '''
        Initialization of a class instances reads data from the provided file, then extracts
        the subject IDs, factors, and peptides into separate arrays, leaving a numpy matrix    
        of intensity values with index arrays linking back to extracted rows and column.
        '''
        self.args = args
        self.timestamp = get_date_string()
        self.fformat = read_json(args.file_format)
        self.matrix = np.loadtxt(args.infile, dtype=np.str,
                                 comments=self.fformat['file_format']['comments'],
                                 delimiter=self.fformat['file_format']['delimiter']
                                 )
        self.rowLabel2attrib = self.fformat['header']  # dict mapping actual header row label to internal representation
        self.headerRowCount = len(self.fformat['header'])  # number of header rows
        self.rowAttrib2label = dict([(k, v) for (v, k) in self.rowLabel2attrib.items()])  # reverse mapping to row label
        self.hRowLabels = self.matrix[:self.headerRowCount, 0]  # header row labels in order (by row)
        self.hRowDataDict = dict()
        # self.headerRowAttrib = np.asarray(list(self.rowAttrib2label.keys()), dtype=np.str)
        self.hRowAttribs = np.empty((self.headerRowCount), dtype='<U50')
        for i in range(self.headerRowCount):
            self.hRowAttribs[i] = self.rowLabel2attrib[self.hRowLabels[i]]  # corresponding header row attribs in order
            self.hRowDataDict[self.hRowAttribs[i]] = self.matrix[i, 1:]  # column headings
        self.peptides = self.matrix[self.headerRowCount:, 0]  # all first column feature identifiers
        self.npeptides = len(self.peptides)  # count the peptides
        self.ivals_unscaled = np.asarray(self.matrix[self.headerRowCount:, 1:],
                                         dtype=np.float)  # make numpy float matrix contain entirely of intensity values
        self.add_factors()  # if needed, add additional header rows to form complete factor names
        self.check_samples()
        eprint("done."),
        if self.args.json:  # if JSON file provided as input ...
            self.make_facIndex_json()  # create factor index arrays based on factor aliases and names
        else:  # described therein.
            self.make_facIndex()  # otherwise, make factor index arrays based directly on matrix
            # file row labels and factor names

    def check_samples(self):
        """
        Check for duplicate samples and raise the alarm.
        """
        samples = self.hRowDataDict['_sample']
        uniq, inverse, counts = np.unique(samples, return_inverse=True, return_counts=True)
        duplicate_samples = uniq[np.where(counts > 1)]
        if len(duplicate_samples) > 0:
            eprint('\nWARNING: please ensure that sample IDs are unique; disambiguate before proceeding.')
            sys.exit('ERROR: the following sample IDs occur more than once: {:s}.'.format(repr(duplicate_samples)))

    def logTransform_ivals(self, myIvals, pcount):
        """
        Take the natural log of all intensity values in the dataset using a pseudocount to prevent
        division-by-zero errors.
        """
        eprint('Log-transforming intensity values...', end='')
        lt_iVals = np.empty_like(myIvals)  # predefine ndarray based on myIvals
        lt_iVals = np.log(myIvals + pcount)  # add pcount to myIvals element-wise, then take natural log
        del myIvals
        eprint('done.')
        return (lt_iVals)

    def rescale_ivals(self, myIvals, axis=None):
        """
        Rescale all feature data in original data matrix to zero mean and unit variance.
        Scaling is done on a per-subject basis (column-wise)a if axis=0, or
        on a per-feature (row-wise or per-peptide) basis if axis=1 (default, preferred)
        """
        eprint('Rescaling intensity values, axis={:d}...'.format(axis), end='')
        if axis == None:
            axis = self.args.rescale_axis
            eprint("rescale_ivals(axis not supplied, using self.args.rescale_axis={:d})".format(self.args.rescale_axis))
        myScaler = preprocessing.StandardScaler()
        myScaler.fit_transform(myIvals)
        scaled_ivals = preprocessing.scale(myIvals, axis=axis)
        del myIvals
        eprint('done.')
        return scaled_ivals, myScaler

    def calc_tTest(self):
        """
        calc_tTest(facRow=0, dsRow=0, ttestLoL=[])
        Given the ttestLoL data structure containing:
            Subject sub-1   sub-2 ...
            Factor  fac-1   fac-2 ...
            PEP1    val1,1  val1,2  ...
            PEP2    val2,1  val2,2  ...
        Calculate t-tests for each peptide for each factor combination.
        """
        eprint("Calculating t-test...", end='')
        sample = list()
        self.ttests = np.zeros((self.npeptides, self.nfactors - 1, self.nfactors),
                               dtype=np.float)  # initialize ttest zero matrix
        for i in range(self.nfactors):  # loop over factors
            sample.append(self.ivals[:, self.facIndex[i]])
        for f1 in range(self.nfactors):
            for f2 in range(f1 + 1, self.nfactors):
                self.ttests[:, f1, f2] = stats.ttest_ind(a=sample[f1],
                                                         b=sample[f2],
                                                         axis=1,
                                                         equal_var=False).pvalue
        eprint("done.")

    def sortPeptides(self):
        """
        For each factor combination, sort peptides according to t-test p-value.
        The goal is to return a ranked list of peptides with intensity values
        for each subject/factor for use in a classifier.
        """
        eprint("Sorting peptides by p-value...", end='')
        self.indexArray = np.zeros((self.npeptides, self.nfactors - 1, self.nfactors), dtype=np.int)
        for f1 in range(self.nfactors):
            for f2 in range(f1 + 1, self.nfactors):
                self.indexArray[:, f1, f2] = np.argsort(self.ttests[:, f1, f2],
                                                        axis=0)  # index arrays based on sorted p-values
                if self.args.debug:
                    for i in range(self.npeptides):
                        print('{:2d} {:s} {:7.5f} {:d}'.format(
                            i,
                            self.peptides[i],
                            self.ttests[i, f1, f2],
                            self.indexArray[i, f1, f2]))
                    print(self.peptides[self.indexArray[:, f1, f2]])
        eprint("done.")

    def add_factors(self):
        """
        Use data from addition matrix file header rows to add specificity to the original
        Factors.  For example, adding Wafer number to affected status can help assess the
        importance of batch effects on the ability to predict status.  The --add_factors
        cmdline parameter(s) are used to identify text to be added to each subject's factor
        string.  Additional factors are named using the header row labels found in column
        zero of the data matrix.
        """
        args = self.args  # use the command-line arguments assigned to object
        if args.add_factors:  # if --add_factors parameter used ...
            newFactorsL = []  # init list of validated header row labels
            for newfac in args.add_factors[0]:  # why does a single entry come rapped in its own list?
                newfacCount = self.hRowLabels.count(
                    newfac)  # count the number of previously-identified row labels matched by parameter
                if newfacCount == 1:  # if there is a single matching row label, use it
                    newFactorsL.append(newfac)  # add the row label to the validated list
                elif newfacCount == 0:  # if parameter does not match a known label
                    eprint('WARNING: --add_factors parameter: "{:s}" is not a valid header row label.'.format(newfac))
                else:  # if there are zero or > 1 matching row labels
                    eprint(
                        'WARNING: --add_factors parameter: "{:s}" does not identify a unique header row label.'.format(
                            newfac))
                if newfacCount != 1:
                    eprint('Note: the following row labels are valid --add_factors parameters: {:s}.'.format(
                        repr(self.hRowLabels)))
            for newfac in newFactorsL:  # foreach validated row label ...
                attrib = self.rowLabel2attribute[newfac]  # get row attribute for new factor component/row label
                myFactorList = [i + '_' + j for i, j in zip(self.hRowLabels['_factor'], self.hRowLabels[
                    attrib])]  # concatenate factor components subject-wise
                self.hRowLabels['_factor'] = np.asarray(myFactorList, dtype=np.str)  # make the list an np.array

    def getI4factor(self, myFactors):
        """
        Given one or more factor indices, return the peptide intensity values for the subjects
        with the corresponding factors.  The argument myFactors should be an iterable object,
        typically a list or array of factor indices. The method returns a LoLoL with axis=0
        corresponding to the factors in the order in which they were received.
        """
        facIvals = []  # initialize list to receive intensity values for each factor
        for i in myFactors:  # loop over input factor indices
            facIvals.append(
                self.ivals[:, self.facIndex[i]])  # using the index array of the named factor to identify data columns,
            # append matrix of intensity values to facIvals[]
        return facIvals  # return list of intensity matrices

    def dumpPeptidesNpvalues(self, topN=0):
        """
        Create a table of the most informative peptides and their p-values,
        ranked by p-value, for each factor pair.  Write the data to a file
        named using the output file name root + '.pep'.
        """
        if topN == 0:  # bail out if no peptides requested
            return
        eprint("Dumping peptides and p-values table...", end='')
        peptout = self.outfileRoot + '.' + 'pep'  # create output file name for peptides+pvalues table
        pepout = open(peptout, 'w')  # open output filehandle

        # for each peptide-pvalue pair, create a heading in the form:  factor1 / factor2
        for i in range(self.nfactors):
            for j in range(i + 1, self.nfactors):
                relation = "{:20s} / {:>12s}  ".format(self.factors[i], self.factors[j])
                print(relation, end='', file=pepout)
        print(end='\n', file=pepout)

        # then print the actual peptides and p-values
        for k in range(topN):
            for i in range(self.nfactors):
                for j in range(i + 1, self.nfactors):
                    pepNpv = "{:20s}   {:12.5e}  ".format(self.peptides[self.indexArray[k, i, j]],
                                                          self.ttests[self.indexArray[k, i, j], i, j])
                    print(pepNpv, end='', file=pepout)
            print(end='\n', file=pepout)
        pepout.close()  # close filehandle
        eprint("done.")

    def make_facIndex(self):
        """
        The method creates and returns facIndex, a look-up table which converts a
        factor name (e.g., "control") into its index in the self.factors array
        (e.g., 4) using the factor names from the original input file.
        Creates:
            self.factors        list of unique factors
            self.nfactors       unique factor count
            self.facIndex       list of index arrays for each factor (by factor index)
            self.facName2num    dictionary that translates factor name to factor index
        """
        uList = [x for x in
                 set(self.hRowDataDict['_factor'])]  # self.hRowLabels... is an ndarray: make unique and convert to list
        self.factors = sorted(uList)  # sort factors so the always come in same order
        self.nfactors = len(self.factors)  # number of unique factors
        facIndex = list()
        self.facName2num = dict()
        for i in range(self.nfactors):  # loop over factor indices
            facIndex.append(np.where(
                self.hRowDataDict['_factor'] == self.factors[i]))  # make array of column indices for that factor
            self.facName2num[self.factors[i]] = i  # assign effective factor number to dict keyed on factor name
        self.facIndex = np.squeeze(np.asarray(
            facIndex))  # squeeze out the single-dimensional entries that crept in when appending array tuples

    def make_facIndex_json(self):
        """
        The method creates a lookup table from the dictionary in self.args.json['recode']
        and uses it to remap factor names as they appear in the master data matrix to
        those encoded in the JSON (text or file) argument to the parameter '--recode'.
        The mapping may simply rename the factors or map multiple factors to a single
        value (e.g., 'benign' and 'control' to 'normal').
        Creates:
            self.factors        list of unique factors
            self.nfactors       unique factor count
            self.facIndex       list of index arrays for each factor (by factor index)
            self.facName2num    dictionary that translates factor name to factor index
        """
        recode = self.args.json                             # JSON file defines mapping of factors to aliases using dict
        self.factors = sorted(recode.keys())                # json keys are now aliases for original factors (phenotypes)
        self.nfactors = len(self.factors)
        self.factorName2num = dict()
        self.facIndex = dict()
        for i, factor in enumerate(self.factors):
            self.factorName2num[factor] = i                 # given factor name, return its index
        factor2alias = dict()                               # given original factor, return its new alias
        for factor, list in recode.items():
            for alias in list:
                factor2alias[alias] = factor
        for factor in self.factors:
            mlist = []
            for i, old_fac in enumerate(self.hRowDataDict['_factor']):
                if factor2alias[old_fac] == factor:
                    mlist.append(i)
            self.facIndex[self.factorName2num[factor]] = mlist
        for i in range(self.nfactors):  # check that all factors have corresponding subjects
            if len(self.facIndex[i]) == 0:
                eprint('ERROR: zero subjects are assigned to factor {:s}.'.format(self.factors[i]))
                eprint('This seems improbable; please check that all JSON keys have factors from\n' +
                       'the matrix file mapped to them.')
                sys.exit('Exiting.')


# Doctest
if __name__ == "__main__":
    import doctest

    doctest.testmod()
