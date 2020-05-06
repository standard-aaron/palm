import argparse
from argparse import ArgumentParser
import warnings
import numpy as np
import glob
from scipy.stats import chi2
import scipy.stats as stats
from scipy.special import logsumexp
from scipy.optimize import minimize
import progressbar
import sys
from numba import njit

from Bio import Phylo
from io import StringIO
import palm_utils.tree_utils as tree_utils

def locus_parse_coal_times(args):
	bedFile = args.treesFile
	derivedAllele = args.derivedAllele
	posn = args.posn
	sitesFile = args.sitesFile
	outFile = args.outFile
	timeScale = args.timeScale
	burnin = args.burnin
	thin = args.thin
	debug = args.debug

	if not args.sites:
		indLists = tree_utils._derived_carriers_from_haps(sitesFile,
								posn,
								args.offset,
								relate=args.relate)
	else:
		indLists = tree_utils._derived_carriers_from_sites(sitesFile,
								posn,
								relate=args.relate,
								derivedAllele=args.derivedAllele)	
	derInds = indLists[0]
	ancInds = indLists[1]
	ancHap = indLists[2]

	n = len(derInds)
	m = len(ancInds)
	
	f = open(bedFile,'r')
	lines = f.readlines()
	lines = [line for line in lines if line[0] != '#' and line[0] != 'R' and line[0] != 'N'][burnin::thin]
	
	numImportanceSamples = len(lines)


	derTimesList = []
	ancTimesList = []

	#if debug:
	#	print('Parsing trees...',file=sys.stderr)
	#	bar = progressbar.ProgressBar(maxval=numImportanceSamples, \
	#    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	#	bar.start()
	for (k,line) in enumerate(lines):
		nwk = line.rstrip().split()[-1]
		derTree =  Phylo.read(StringIO(nwk),'newick')
		ancTree = Phylo.read(StringIO(nwk),'newick')
		mixTree = Phylo.read(StringIO(nwk),'newick')

		derTimes,ancTimes,mixTimes = tree_utils._get_times_all_classes(derTree,ancTree,mixTree,
							derInds,ancInds,ancHap,n,m,sitesFile,
							timeScale=timeScale, prune=args.prune)
		derTimesList.append(derTimes)
		ancTimesList.append(ancTimes)

	#	if args.debug:
	#		bar.update(k+1)

	

	#if args.debug:
	#	bar.finish()
	times = -1 * np.ones((2,n+m,numImportanceSamples))
	for k in range(numImportanceSamples):
		times[0,:len(derTimesList[k]),k] = np.array(derTimesList[k])
		times[1,:len(ancTimesList[k]),k] = np.array(ancTimesList[k])
	return times

def _args_passed_to_locus(args):
	locusDir = args.locusDir
	passed_args = args

	# reach into args and add additional attributes
	d = vars(passed_args)
	d['treesFile'] = locusDir+args.locusTreeFile
	d['sitesFile'] = locusDir+args.locusSitesFile

	d['popFreq'] = 0.50

	d['posn'] = args.posn 
	d['derivedAllele'] = args.derivedAllele 
	return passed_args


def _args(super_parser,main=False):
	if not main:
		parser = super_parser.add_parser('snp_extract',description=
                'Parse/extract coalescence times in the derived & ancestral classes.')
	else:
		parser = super_parser
	# mandatory inputs:
	required = parser.add_argument_group('required arguments')
	required.add_argument('--locusDir',type=str)
	required.add_argument('--posn',type=int)
	required.add_argument('--derivedAllele',type=str)
	# options:
	parser.add_argument('-q','--quiet',action='store_true')
	parser.add_argument('-o','--output',dest='outFile',type=str,default=None)
	parser.add_argument('-debug','--debug',action='store_true')

	parser.add_argument('--locusTreeFile',type=str,default='mssel.tree')
	parser.add_argument('--locusSitesFile',type=str,default='relate.haps')
	parser.add_argument('--locusOutPrefix',type=str,default='mssel',help='prefix for outfiles (.ld, .der.npy, .anc.npy)')

	parser.add_argument('-timeScale','--timeScale',type=float,help='Multiply the coal times \
						 	in bedFile by this factor to get in terms of generations; e.g. use \
						 	this on trees in units of 4N gens (--timeScale <4*N>)',default=1)
	parser.add_argument('--relate',action='store_true')
	parser.add_argument('--sites',action='store_true')
	parser.add_argument('-thin','--thin',type=int,default=1)
	parser.add_argument('-burnin','--burnin',type=int,default=0)	
	parser.add_argument('--sep',type=str,default='\t')
	parser.add_argument('--offset',type=int,default=0)
	parser.add_argument('--prune',type=str,default=None)
	return parser

def freq(genoMat):
    n = genoMat.shape[1]
    return np.sum(genoMat,axis=1)/n

def r2(genoMat,posnFocal,posns,freqs):
    ifiltfocal = list(posns).index(posnFocal)
    genoMatFilt = genoMat[:,:]
    l = genoMatFilt.shape[0]
    r2vec = np.zeros(l)
    n = genoMatFilt.shape[1]
    rowa = genoMatFilt[ifiltfocal,:]
    for j,rowb in enumerate(genoMatFilt):
            pab = (rowa & rowb).sum()/n
            pa = rowa.sum()/n
            pb = rowb.sum()/n
            #print(pab,pa,pb)
            r2el = ((pab - pa*pb)/np.sqrt(pa*(1-pa)*pb*(1-pb)))
            r2vec[j] = r2el
    return np.array(r2vec)


def _parse_haps_file(haps,focalPosn):
        genoMat = []
        posns = []

        for line in open(haps,'r'):
                if line[0] == 'N' or line[0] == 'R':
                        continue

                cols = line.rstrip().split(' ')
                posn = int(cols[2])
                if posn == focalPosn:
                        iFocal = len(posns)
                alleles = ''.join(cols[5:])
                ancAllele = '0'
                derAllele = '1'
                if alleles == ancAllele*len(alleles) or alleles == derAllele*len(alleles):
                        continue
                genoMat.append([0 if char == ancAllele else 1 for char in alleles])
                posns.append(posn)
        genoMat = np.array(genoMat)

        freqs = freq(genoMat)
        freqs = np.array(freqs)
        posns = np.array(posns)	
        r2vector = r2(genoMat,focalPosn,posns,freqs)
        return posns,freqs,r2vector

def _write_ld_file(args,posns,freqs,r2vector,focalPosn,focalFreq,locusDir):
	out = open(locusDir+args.locusOutPrefix+'.ld','w')
	out.write('#posn\tfreq\tr\n')
	out.write('##%d\t%f\n'%(focalPosn,focalFreq))
	for (p,f,r) in zip(posns,freqs,r2vector):
		out.write('%d\t%f\t%f\n'%(p,f,r))
	out.close()
	return	


def _write_times_files(args,locusTimes):
	locusDir = args.locusDir
	i0 = np.argmax(locusTimes[0,:,0] < 0.0) 
	i1 = np.argmax(locusTimes[1,:,0] < 0.0)
	a1 = locusTimes[0,:i0,:]
	a2 = locusTimes[1,:i1,:]
	a1 = a1.transpose()	
	a2 = a2.transpose()
	np.save(locusDir+args.locusOutPrefix+'.der.npy',a1)
	np.save(locusDir+args.locusOutPrefix+'.anc.npy',a2)
	return 	

def _parse_locus_stats(args):
	passed_args = _args_passed_to_locus(args)
	locusTimes = locus_parse_coal_times(passed_args)
	_write_times_files(args,locusTimes)
	return

def _main(args):	
	_parse_locus_stats(args)
	
if True:
        super_parser = argparse.ArgumentParser()
        parser = _args(super_parser,main=True)
        args = parser.parse_args()
        _main(args)
