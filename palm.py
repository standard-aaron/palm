import sys
import argparse
import pandas as pd
import warnings
import numpy as np
import glob
import scipy.stats as stats
import progressbar
from numba import njit

def _args(super_parser,main=False):
	if not main:
		parser = super_parser.add_parser('trait',description=
                'Trait selection tests.')
	else:
		parser = super_parser
	# mandatory inputs:
	required = parser.add_argument_group('required arguments')
	required.add_argument('--traitDir',type=str,help='A directory containing only directories, each representing a causal locus')
	required.add_argument('--metadata',type=str,help='A dataframe holding attributes for each SNP') 
	parser.add_argument('--traits',type=str,help='Traits to analyze, separated by commas; only specify if metadata has betas indexed by trait(s) (e.g., joint analysis of traits)',default='NULL')
	# options:
	parser.add_argument('-q','--quiet',action='store_true')
	parser.add_argument('-o','--output',dest='outFile',type=str,default=None)

	parser.add_argument('--quad',type=str,default=None,help='prefix for the quadratic likelihood fits from lik.py')
	parser.add_argument('--out',type=str,default=None)
	parser.add_argument('--B',type=int,default=250)
	#advanced options
	parser.add_argument('--maxp',type=float,default=1)
	parser.add_argument('--minmaf',type=float,default=0.005)
	parser.add_argument('--seed',default=None,type=int)
	return parser

def _parse_loci_stats(args):
	if args.seed != None:
		np.random.seed(args.seed)	
	
	coeffs = []
	betas = []
	mults = []
	pvals = []	
	df = pd.read_csv(args.metadata,sep='\t',index_col=(0,1),header=0)
		
	if args.traits == 'NULL':
		betaColumns = ['beta']
		pColumns = ['pval']
		seColumns = ['se']
		if not args.quiet:
			print()
			print('Analyzing trait...')
		traitNames = ['']
	else:
		pStr = 'pval@'	
		seStr = 'se@'
		betaStr = 'beta@'
		betaColumns = [col for col in df.columns if np.any([betaStr+trait in col for trait in (args.traits).split(',')])]
		pColumns = [col for col in df.columns if np.any([pStr+trait in col for trait in (args.traits).split(',')])]
		seColumns = [col for col in df.columns if np.any([seStr+trait in col for trait in (args.traits).split(',')])]
		traitNames = [col[len(betaStr):] for col in betaColumns]
		if not args.quiet:
			print()
			print('Analyzing traits: %s'%(', '.join(traitNames)))

	dfFiltered = df
	idxs = dfFiltered.index.values
	K = len(idxs)
	if not args.quiet:
		print('Loading likelihoods...',file=sys.stderr)
		bar = progressbar.ProgressBar(maxval=K, \
			widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

	for (k,idx) in enumerate(idxs):
		dfRow = df.loc[idx]
		mult = len(df.loc[idx[0]].index)
		variant = idx[1] 
		cols = variant.split(':')
		chrom = int(cols[0])
		bp = int(cols[1])
		ref = cols[2]
		alt = cols[3]
	
		ld_block = int(str(idx[0]).lstrip('ld_') )
		
		locusDir = args.traitDir+'ld_%d/'%(ld_block)	
		
		minor_allele = dfRow.minor_allele
		derived_allele = dfRow.derived_allele

		MAF = float(dfRow.minor_AF)
		if derived_allele != alt:
			flipper = -1.0
		else:
			flipper = 1.0

		loc_betas = list(flipper * np.array(dfRow[betaColumns]))
		loc_ses = list(np.array(dfRow[seColumns],dtype=float))
		loc_pvals = np.array(dfRow[pColumns],dtype=float)	
		if np.size(loc_ses) == 0:
			loc_ses = list(np.zeros(len(betaColumns)))
		if np.size(loc_pvals) == 0:
			loc_pvals = list(np.zeros(len(betaColumns)))
		if args.maxp < 1:
			if np.any(np.isnan(loc_betas)):
				continue	
			if np.logical_not(np.any(stats.chi2.sf((np.array(loc_betas)/np.array(loc_ses))**2,df=1) < args.maxp)):
				continue
				
		if MAF < args.minmaf:
			continue 

		try:	
			if args.quad != None:
				coeff = np.load(locusDir + args.quad + '.npy')	
			else:
				coeff = np.load(locusDir + 'bp%s.quad_fit.npy'%(bp))
		except:
			continue
		coeffs.append(coeff)
		betas.append(loc_betas)
		mults.append(1/mult)
		pvals.append(np.array(loc_pvals))
		if not args.quiet:
			bar.update(k)
	if not args.quiet:
		bar.finish()
	betas = np.array(betas)
	mults = np.array(mults)
	coeffs = np.array(coeffs)		
	pvals = np.array(pvals) 	
	return coeffs,betas,mults,pvals,traitNames

def _print_omega(omega,ses,traitNames,marg=None,T=None):
	if marg is None and T is None:
		print('Trait\t\t\tSel\t(SE)\t\tZ')
	else:
		print('Trait\t\t\tSel\t(SE)\t\tZ\tZmarg\tR')
	print('='*90)
	for j,trait in enumerate(traitNames):
		if len(trait) < 16:
			traitFmt = trait+' '*(16-len(trait))
		else:
			traitFmt = trait[:16]
		if marg is None and T is None:
			print('%s\t%.3f\t(%.4f)\t%.3f'%(traitFmt,omega[j],ses[j],omega[j]/ses[j]))
		else:
			print('%s\t%.3f\t(%.4f)\t%.3f\t%.3f\t%.3f'%(traitFmt,omega[j],ses[j],omega[j]/ses[j],marg[j],T[j]))
	print('='*90)
	return
			
def _out(omega,ses,L_byTrait,L,out,marg=None,T=None):
	np.save(out+'.est.npy',omega)
	np.save(out+'.se.npy',ses)
	np.save(out+'.L.npy',np.concatenate(([L],L_byTrait)))
	if marg is not None:
		np.save(out+'.margZ.npy',marg)
	if T is not None:
		np.save(out+'.T.npy',T)
	return

def _bootstrap(stats):
	coeffs,betas,mults,pvals,traitNames = stats
	I = np.random.choice(coeffs.shape[0],coeffs.shape[0],replace=True)
	return coeffs[I,:],betas[I,:],mults[I],pvals[I,:],traitNames

def _opt_omega(stats):
	coeffs,betas,mults,pvals,traitNames = stats
	J = betas.shape[1]
	L = betas.shape[0]
	
	A = np.zeros((J,J))
	b = np.zeros(J)
	
	for l in range(L):
		A += 2 * mults[l] * coeffs[l,0] * np.outer(betas[l,:],betas[l,:])
		b += -mults[l] * coeffs[l,1] * betas[l,:]
	Ainv = np.linalg.inv(A)
	omega = np.dot(Ainv,b)
	return omega	

def _nloci(stats,args):
	coeffs,betas,mults,pvals,traitNames = stats
	L = pvals.shape[0]
	J = pvals.shape[1]
	L_byTrait = np.sum(pvals < args.maxp,axis=0)
	return L_byTrait,L	

def _inference(statistics,args):
	omega = _opt_omega(statistics)	
	L = statistics[0].shape[0]
	J = len(omega)
	L_byTrait,L = _nloci(statistics,args)
	print('Analyzing %d loci...'%(L))
	B = args.B
	
	omegaJK = np.zeros((J,B))
	for b in range(B):	
		statsDK = _bootstrap(statistics) 
		omegaJK_b = _opt_omega(statsDK)	
		omegaJK[:,b] = omegaJK_b 
	ses = np.std(omegaJK,axis=1)
	
	return omega,ses

def _T_inference(statistics,args):
	coeffs,betas,mults,pvals,traitNames = statistics
	L_byTrait,L = _nloci(statistics,args)
	print('Analyzing %d loci...'%(L))
	omega = _opt_omega(statistics)
	J = betas.shape[1]
	margOmega = np.zeros(J)
	for j in range(J):
		Lj = L_byTrait[j]
		msig = pvals[:,j] < args.maxp
		mcoeffs = coeffs[msig,:]
		mbetas = np.reshape(betas[msig,j],(Lj,1))
		mmults = mults[msig]
		mpvals = np.reshape(pvals[msig,j],(Lj,1))
		mtraitNames = [traitNames[j]]	
		mstats = mcoeffs,mbetas,mmults,mpvals,mtraitNames
		momega = _opt_omega(mstats) 
		margOmega[j] = momega			

	## se estimation
	B = args.B
	omegaJK = np.zeros((J,B))
	margOmegaJK = np.zeros((J,B))
	for b in range(B):	
		statsDK = _bootstrap(statistics) 
		omegaJK_b = _opt_omega(statsDK)	
		omegaJK[:,b] = omegaJK_b 

		coeffs,betas,mults,pvals,traitNames = statsDK
		L_byTrait,L = _nloci(statsDK,args)

		for j in range(J):
			Lj = L_byTrait[j]
			msig = pvals[:,j] < args.maxp
			mcoeffs = coeffs[msig,:]
			mbetas = np.reshape(betas[msig,j],(Lj,1))
			mses = np.reshape(ses[msig,j],(Lj,1))
			mx0 = x0[msig]
			mmults = mults[msig]
			mpvals = np.reshape(pvals[msig,j],(Lj,1))
			mtraitNames = [traitNames[j]]	
			mstats = mcoeffs,mbetas,mses,mx0,mmults,mpvals,mtraitNames
			momega = _opt_omega(mstats) 
			margOmegaJK[j,b] = momega			
	ses = np.std(omegaJK,axis=1)
	Dses = np.std(omegaJK.transpose()/np.std(omegaJK,axis=1)-margOmegaJK.transpose()/np.std(margOmegaJK,axis=1),axis=0).transpose()
	Mses = np.std(margOmegaJK,axis=1)
	D = omega/ses - margOmega/Mses
	return omega,ses,margOmega,Mses,D,Dses

def _main(args):	
	statistics = _parse_loci_stats(args)
	L_byTrait,L = _nloci(statistics,args)
	coeffs,betas,mults,pvals,traitNames = statistics	
	J = betas.shape[1]

	if J > 1:
		# run test jointly
		omega,ses,margOmega,Mses,D,Dses = _T_inference(statistics,args)	
		Ts = D/Dses
		margZs = margOmega/Mses
	else:
		omega,ses = _inference(statistics,args)	

	if args.out != None:
		if J > 1:
			_out(omega,ses,L_byTrait,L,args.out,marg=margZs,T=Ts)
		else:
			_out(omega,ses,L_byTrait,L,args.out)	
	if not args.quiet:
		if J > 1:
			_print_omega(omega,ses,traitNames,marg=margZs,T=Ts)
		else:
			_print_omega(omega,ses,traitNames)
		print()
	return

if True:
	super_parser = argparse.ArgumentParser()
	parser = _args(super_parser,main=True)
	args = parser.parse_args()
	_main(args)
