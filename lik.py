import argparse
import pandas as pd
import warnings
import numpy as np
import glob
from scipy.stats import chi2
import scipy.stats as stats
import progressbar
import sys
from numba import njit

from palm_utils.deterministic_utils import log_l_importance_sampler 
import gzip

def parse_clues(filename):
    with gzip.open(filename, 'rb') as fp:
        try:
            #parse file
            data = fp.read()
        except OSError:
            with open(filename, 'rb') as fp:
                try:
                    #parse file
                    data = fp.read()
                except OSError:
                    print('Error: Unable to open ' + filename)
                    exit(1)
           
        #get #mutations and #sampled trees per mutation
        filepos = 0
        num_muts, num_sampled_trees_per_mut = np.frombuffer(data[slice(filepos, filepos+8, 1)], dtype = np.int32)
        #print(num_muts, num_sampled_trees_per_mut)

        filepos += 8
        #iterate over mutations
        for m in range(0,num_muts):
            bp = np.frombuffer(data[slice(filepos, filepos+4, 1)], dtype = np.int32)
            filepos += 4
            anc, der = np.frombuffer(data[slice(filepos, filepos+2, 1)], dtype = 'c')
            filepos += 2
            daf, n = np.frombuffer(data[slice(filepos, filepos+8, 1)], dtype = np.int32)
            filepos += 8
            #print("BP: %d, anc: %s, der %s, DAF: %d, n: %d" % (bp, str(anc), str(der), daf, n))
            
            num_anctimes = 4*(n-daf-1)*num_sampled_trees_per_mut
            anctimes     = np.reshape(np.frombuffer(data[slice(filepos, filepos+num_anctimes, 1)], dtype = np.float32), (num_sampled_trees_per_mut, n-daf-1))
            filepos     += num_anctimes
            #print(anctimes)
            
            num_dertimes = 4*(daf-1)*num_sampled_trees_per_mut
            dertimes     = np.reshape(np.frombuffer(data[slice(filepos, filepos+num_dertimes, 1)], dtype = np.float32), (num_sampled_trees_per_mut, daf-1))
            filepos     += num_dertimes
         
    return dertimes,anctimes 

def _args(super_parser,main=False):
	if not main:
		parser = super_parser.add_parser('lik',description=
                'Locus selection likelihoods.')
	else:
		parser = super_parser
	# mandatory inputs:
	required = parser.add_argument_group('required arguments')
	required.add_argument('--times',type=str)
	# options:
	parser.add_argument('--popFreq',type=float,default=None)
	parser.add_argument('-q','--quiet',action='store_true')

	parser.add_argument('--locusAncientCounts',type=str,default=None)
	parser.add_argument('--out',type=str,default=None)
	#advanced options
	parser.add_argument('-N','--N',type=float,default=10**4)
	parser.add_argument('-coal','--coal',type=str,default=None,help='path to Relate .coal file. Negates --N option.')

	parser.add_argument('-w','--w',type=float,default=0.01)
	parser.add_argument('--sMax',type=float,default=0.1)
	parser.add_argument('-thin','--thin',type=int,default=1)
	parser.add_argument('-burnin','--burnin',type=int,default=0)
	parser.add_argument('--tCutoff',type=float,default=50000)
	parser.add_argument('--linspace',nargs=2,type=int,default=(50,1))
	parser.add_argument('--K',type=int,default=1,help='which epoch (bwd in time) selected started (e.g. K=1 & kappa=1 means selection started + ended in present day)')
	parser.add_argument('--kappa',type=int,default=1,help='# of epochs during which selection occurred, counting back from K')
	parser.add_argument('--timeScale',type=float,default=1.0)
	return parser

def _parse_locus_stats(args):
	locusDerTimes,locusAncTimes = parse_clues(args.times+'.palm')        

	if locusDerTimes.ndim == 0 or locusAncTimes.ndim == 0:
		raise ValueError	
	elif locusAncTimes.ndim == 1 and locusDerTimes.ndim == 1:
		M = 1
		locusDerTimes = np.transpose(np.array([locusDerTimes]))
		locusAncTimes = np.transpose(np.array([locusAncTimes]))
	elif locusAncTimes.ndim == 2 and locusDerTimes.ndim == 1:
		locusDerTimes = np.array([locusDerTimes])[:,::args.thin]
		locusAncTimes = np.transpose(locusAncTimes)[:,::args.thin]
		M = locusDerTimes.shape[1]	
	elif locusAncTimes.ndim == 1 and locusDerTimes.ndim == 2:
		locusAncTimes = np.array([locusAncTimes])[:,::args.thin]
		locusDerTimes = np.transpose(locusDerTimes)[:,::args.thin]
		M = locusDerTimes.shape[1]
	else:
		locusDerTimes = np.transpose(locusDerTimes)[:,::args.thin]
		locusAncTimes = np.transpose(locusAncTimes)[:,::args.thin]
		M = locusDerTimes.shape[1]
	n = locusDerTimes.shape[0] + 1
	m = locusAncTimes.shape[0] + 1
	ntot = n + m 
	row0 = -1.0 * np.ones((ntot,M))
	row0[:locusDerTimes.shape[0],:] = locusDerTimes
	row1 = -1.0 * np.ones((ntot,M))
	row1[:locusAncTimes.shape[0],:] = locusAncTimes
	locusTimes = np.array([row0,row1])* args.timeScale
	
	if args.popFreq == None:
		popFreq = n/ntot
	else:
		popFreq = args.popFreq
	return locusTimes,n,m,popFreq

def _print_sel_coeff_matrix(omega,args,epochs,se):
	print('\t'.join(['%d-%d'%(epochs[i],epochs[i+1]) for i in range(len(epochs[:-1]))]))
	O = omega.shape[0] 
	if True:
		sig = np.zeros((omega.shape[0],3))
		for level in range(3):
			c = stats.norm.ppf(1-0.05/(2*O)*10**-level)
			sig[:,level] = np.logical_not((omega - c*se <= 0) & (omega + c*se >= 0)) 
	print('\t'.join(['%.3f%s'%(omega[i],'*'*int(np.sum(sig[i,:]))) for i in range(omega.shape[0])]))		

	return

def _optimize_locus_likelihood(statistics,args):
	if args.coal != None:
		epochs = np.genfromtxt(args.coal,skip_header=1,skip_footer=1)
		N = 0.5/np.genfromtxt(args.coal,skip_header=2)[2:-1]
		N = np.array(list(N)+[N[-1]])
		K = args.K + args.kappa - 1
	else:
		epochs = np.linspace(0,args.linspace[0],args.linspace[1]+1) 
		N = args.N*np.ones(len(epochs))
		K = len(epochs)-1
	if not args.quiet: 
		print('Demographic model with diploid Ne:')
		print(N)
	icutoff = np.digitize(args.tCutoff,epochs)
	N = N[:icutoff]
	epochs = epochs[:icutoff]

	times,n,m,x0 = statistics
	tmp = np.swapaxes(times, 0, 2)	
	times = tmp
	I = len(epochs)-1
	if not args.quiet:
		print('Analyzing selection over %d time periods...'%(K))
		print('# importance samples: %d'%(times.shape[0]))
		print('Optimizing likelihood surface...')

	ns = np.array([n,m])
	logL0 = 0.0
	theta = np.zeros(len(epochs))	
	logL0 = log_l_importance_sampler(times,ns,epochs,theta,x0,N,tCutoff=args.tCutoff)
	
	S = np.linspace(-args.sMax,args.sMax,200)
	L = np.zeros(len(S))
	for i,s in enumerate(S):
		theta[0:args.kappa] = s
		logL1 = log_l_importance_sampler(times,ns,epochs,theta,x0,N,tCutoff=args.tCutoff)	
		L[i] = logL1 - logL0
		if args.out == None:
			print(s,logL1 - logL0)
	I = np.abs(S-S[np.argmax(L)]) < args.w 
	p = np.polyfit(S[I],L[I],deg=2)
	if args.out != None:
		np.save(args.out+'.quad_fit.npy',p)

	return

def _main(args):	
	statistics = _parse_locus_stats(args)
	_optimize_locus_likelihood(statistics,args)

if True:
	super_parser = argparse.ArgumentParser()
	parser = _args(super_parser,main=True)
	args = parser.parse_args()
	_main(args)




