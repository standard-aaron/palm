from Bio import Phylo
from io import StringIO
import numpy as np

'''
def _coal_times(clades):
        # get number of leaf nodes
        flatten = lambda l: [item for sublist in l for item in sublist]
        if len(clades) == 1:
            clades = clades[0].clades
        bls =  [float(clade.branch_length) for clade in clades]
        #print(bls)
        if len(bls) == 0:
            return []
        lenClades = [len(clade.clades) for clade in clades]
        #print lbl, rbl

        vals = []

        times =  [_coal_times(clade) for clade in clades if len(clade.clades) > 0 ]
        times = flatten(times)
        
        #print(times)
        
        
        #print(times)
        #print(bls,times)
        if len(times) == 0:
            return bls[:-1]
        else:
            return list(np.array(bls[:-1]) + times[0]) + times
'''
def _coal_times(clades):
        # get number of leaf nodes

        [left,right] = clades
        lbl =  float(left.branch_length)
        rbl =  float(right.branch_length)

        #print lbl, rbl
        if len(left.clades) == 0 and len(right.clades) == 0:
            return [rbl]

        elif len(left.clades) == 0:
            right_times =  _coal_times(right.clades)
            return [lbl] + right_times

        elif len(right.clades) == 0:
            left_times =  _coal_times(left.clades)
            return [rbl] + left_times

        else:
            left_times =  _coal_times(left)
            right_times =  _coal_times(right)

            if lbl < rbl:
                return [lbl + left_times[0]] + left_times + right_times
            else:
                return [rbl + right_times[0]] + left_times + right_times
#'''
def _branch_counts(coalTimes, timePts, eps=1):
	## return number of lineages at each time point
	n = len(coalTimes) + 1
	C = [n]
	
	for tp in timePts:
		i = 0
		for (j,ct) in enumerate(coalTimes[i:]):
			if ct >= tp + eps:
				i += j
				C.append( n-j )
				break
	return C

def _derived_carriers_from_haps(hapsFile,posn,offset,relate=False):
    f = open(hapsFile,'r')
    lines = f.readlines()
    for line in lines:
        posnLine = int(line.split()[2])
        if posnLine != posn:
            continue
        if posnLine == posn:
            alleles = ''.join(line.rstrip().split()[5:])
            hapsDer = [str(i+1-int(relate)+offset) for i in range(len(alleles)) if alleles[i] == '1']
            hapsAnc = [str(i+1-int(relate)+offset) for i in range(len(alleles)) if alleles[i] != '1'] 
            return [hapsDer,hapsAnc,[]]

def _derived_carriers_from_sites(sitesFile,posn,derivedAllele='G',ancientHap=None,relate=False,nDer=None,nAnc=None):
    '''
    Takes the sitesFile
    Returns a list of individuals (labels in 
    the header of sitesFile) who carry derived allele
    '''

    f = open(sitesFile,'r')
    lines = f.readlines()

    headerLine = lines[0]
    inds = headerLine.split()[1:]

    for line in lines:
        if line[0] == '#' or line[0] == 'N' or line[0] == 'R':
            continue
        cols = line.rstrip().split()
        thisPosn = int(cols[-2])
        alleles = cols[-1]
        if thisPosn < posn:
            continue
        elif thisPosn == posn:
            n = len(alleles)
            if not relate: 
                if ancientHap:
                    raise NotImplementedError 
                indsAnc = [str(_) for _ in range(1,nAnc+1)]
                indsDer = [str(_) for _ in range(nAnc+1,n+1)]
                return [indsDer,indsAnc,[]]
            if ancientHap != None:
                raise NotImplementedError
                idxsDerived = [i for (i,x) in enumerate(alleles) if x == derivedAllele and inds[i] != ancientHap]
                indsDerived = [inds[i] for i in idxsDerived]
                indsAnc = [ind for (i,ind) in enumerate(inds) if i not in idxsDerived and inds[i] != ancientHap]
                return [indsDerived,indsAnc,[ancientHap]]
            else:
                idxsDerived = [i for (i,x) in enumerate(alleles) if x == derivedAllele]
                indsDerived = [inds[i] for i in idxsDerived]
                indsAnc = [ind for (i,ind) in enumerate(inds) if i not in idxsDerived]
                return [indsDerived,indsAnc,[]]
        else:
            inds.remove(ancientHap) 
            return [[],inds,[ancientHap]]
            #raise ValueError('Specified posn not specified in sitesFile')

def _get_times_all_classes(derTree,ancTree,mixTree,derInds,ancInds,ancHap,n,m,sitesFile,timeScale=1,prune=None):
    
    indsToPrune = []
    if prune != None:
        for line in open(prune,'r'):
            indsToPrune += [line.rstrip()]
    #print(indsToPrune)
    if sitesFile == None:
        ### assume all individuals are fixed for the derived type!
        if ancHap != None:
            raise NotImplementedError
        else:
            derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
            ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
            mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))

    if ancHap == None:
        ancHap = []
    if n >= 2 and m >= 2:   
        for ind in set(ancInds + ancHap + indsToPrune):
            #print('der',ind)
            derTree.prune(ind)
        for ind in set(derInds + ancHap + indsToPrune):
            #print('anc',ind)
            ancTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
        ancTimes = timeScale *np.sort(_coal_times(ancTree.clade.clades))
        mixTimes = timeScale *np.sort(_coal_times(mixTree.clade.clades))


    elif n == 1 and m >= 2:
        for ind in set(derInds + ancHap + indsToPrune):
            ancTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
    
        ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
        mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))
        derTimes = np.array([])
    
    elif n >= 2 and m == 1:
        for ind in set(ancInds + ancHap + indsToPrune):
            derTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
    
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
        mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))
        ancTimes = np.array([])

    elif n == 0 and m >= 2:
        Cder = [0]
        for ind in set(ancHap + indsToPrune):
            ancTree.prune(ind)
        ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
        derTimes = np.array([])
        mixTimes = np.array([])

    elif n >= 2 and m == 0:
        Canc = [0]
        for ind in set(ancHap + indsToPrune):
            derTree.prune(ind)
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))
        ancTimes = np.array([])
        mixTimes = np.array([])
    return derTimes,ancTimes,mixTimes

def _get_branches_all_classes(derTree,ancTree,mixTree,derInds,ancInds,ancHap,n,m,sitesFile,times,timeScale=1,prune=None):
    
    indsToPrune = []
    if prune != None:
        for line in open(prune,'r'):
            indsToPrune += [line.rstrip()+'_1',line.rstrip()+'_2']
    if sitesFile == None:
        ### assume all individuals are fixed for the derived type!
        if ancHap != None:
            raise NotImplementedError
        else:
            derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
            ancTimes = timeScale *np.sort(_coal_times(ancTree.clade.clades))
            mixTimes = timeScale *np.sort(_coal_times(mixTree.clade.clades))
            Cder = _branch_counts(derTimes,times,eps=10**-10)[1:] + [1]
            Canc = _branch_counts(ancTimes,times,eps=10**-10)[1:] + [1]
            Cmix = _branch_counts(mixTimes,times,eps=10**-10)[1:] + [1]

    if ancHap == None:
        ancHap = []
    if n >= 2 and m >= 2:   
        for ind in set(ancInds + ancHap + indsToPrune):
            #print(ind)
            derTree.prune(ind)
        for ind in set(derInds + ancHap + indsToPrune):
            ancTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
        #print(derTimes[:20])
        #print(derTimes.astype(int)[:20])

        ancTimes = timeScale *np.sort(_coal_times(ancTree.clade.clades))
        #print(ancTimes.astype(int)[:20])
        mixTimes = timeScale *np.sort(_coal_times(mixTree.clade.clades))
        #print(mixTimes.astype(int)[:20])
        Cder = _branch_counts(derTimes,times,eps=10**-10)[1:] + [1]
        Canc = _branch_counts(ancTimes,times,eps=10**-10)[1:] + [1]
        Cmix = _branch_counts(mixTimes,times,eps=10**-10)[1:] + [1]
        print(np.array(Cder).astype(int)[:20])
        print(np.array(Canc).astype(int)[:20])
        print(np.array(Cmix).astype(int)[:20])
        #print(Canc,Cmix)

    elif n == 1 and m >= 2:
        for ind in set(derInds + ancHap + indsToPrune):
            ancTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
    
        ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
        mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))
        #print(n,m)
        Cder = [1]
        Canc = _branch_counts(ancTimes,times,eps=10**-10)[1:] + [1]
        Cmix = _branch_counts(mixTimes,times,eps=10**-10)[1:] + [1]
        #print(Canc,Cmix)
    
    elif n >= 2 and m == 1:
        for ind in set(ancInds + ancHap + indsToPrune):
            derTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
    
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
        mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))
        #print(n,m)
        Cder = _branch_counts(derTimes,times,eps=10**-10)[1:] + [1]
        Canc = [1]
        Cmix = _branch_counts(mixTimes,times,eps=10**-10)[1:] + [1]
    elif n == 0 and m >= 2:
        Cder = [0]
        for ind in set(ancHap + indsToPrune):
            ancTree.prune(ind)
        ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
        Canc = _branch_counts(ancTimes,times,eps=10**-10)[1:] + [1]
        Cmix = Canc

    elif n >= 2 and m == 0:
        Canc = [0]
        for ind in set(ancHap + indsToPrune):
            derTree.prune(ind)
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))
        Cder = _branch_counts(derTimes,times,eps=10**-10)[1:] + [1]  
        Cmix = [1]
    
    return Cder,Canc,Cmix
