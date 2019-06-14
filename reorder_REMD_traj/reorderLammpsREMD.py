#!/usr/bin/env python

"""
LAMMPS Replica Exchange Molecular Dynamics (REMD) trajectories are arranged by
replica, i.e., each traj is a continuous replica that records all the 
ups and downs in temperature. However, often the requirement is trajectories
that are continuous in temperature. This requires the LAMMPS REMD trajectories
to be re-ordered. This script achieves that in parallel using MPI.

Author: Tanmoy Sanyal, Shell lab, Chemical Engineering, UC Santa Barbara


Usage
-----
To get detailed information about the arguments, flags, etc use:
python reorderLammpsREMD.py -h or 
python reorderLammpsREMD.py --help

Features of this script
-----------------------
a) reorder LAMMPS REMD trajectories by temperature keeping only desired frames.
Note: this only handles LAMMPS format trajectories (i.e. .lammpstrj format)
Trajectories can be gzipped or bz2-compressed.

b) (optionally) calculate configurational weights for each frame at each
temperature if potential energies are supplied.

Dependencies
------------
mpi4py
pymbar (for getting configurational weights)
tqdm (for printing pretty progress bars)
StringIO (or io if in Python 3.x)

"""


import os, sys, numpy as np, argparse, time, pickle

from mpi4py import MPI
from tqdm import tqdm, trange
import gzip
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#import pymbar, whamlib


# init MPI
# (note that all output on screen will be printed only on the ROOT proc)
ROOT = 0
comm = MPI.COMM_WORLD
me = comm.rank # my proc id
nproc = comm.size


def _get_nearest_temp(temps, query_temp):
    """
    Helper function to get the nearest temp in a list
    from a given query_temp
    
    :param temps: list of temps.
    
    :param query_temp: query temp
    
    Returns:
    idx: index of nearest temp in the list
    
    out_temp: nearest temp from the list
    """
    
    if isinstance(temps, list): temps = np.array(temps)
    idx = np.argmin(abs(temps - query_temp))
    out_temp = temps[idx]
    return idx, out_temp


def readwrite(trajfn, mode = "rb"):
    """ 
    Helper function for input/output LAMMPS traj files.
    Trajectories may be plain text, .gz or .bz2 compressed.
        
    :param trajfn: name of LAMMPS traj
    
    :param mode: "r" ("w") and "rb" ("wb") depending on read or write
        
    Returns: file pointer
    """
    
    if trajfn.endswith(".gz"):
        return gzip.GzipFile(trajfn, mode)
    elif trajfn.endswith(".bz2"):
        return bz2.BZ2File(trajfn, mode)
    else:
        return file(trajfn, mode)


def get_replica_frames(logfn, temps, nswap, writefreq):
    """
    Get a list of frames from each replica that is 
    at a particular temp. Do this for all temps.
    
    :param logfn: master LAMMPS log file that contains the temp
                  swap history of all replicas
    
    :param temps: list of all temps used in the REMD simulation.
    
    :param nswap: swap frequency of the REMD simulation
    
    :param writefreq: traj dump frequency in LAMMPS
    
    Returns: master_frame_dict: dict containing a tuple (replica #, frame #) 
                                for each temperature
    """
    
    n_rep = len(temps)
    swap_history = np.loadtxt(logfn, skiprows = 3)
    master_frame_dict = dict( (n, []) for idx in range(n_rep) ) 
    
    # walk through the replicas
    print("Getting frames from all replicas at temperature:")
    for n in range(n_rep):
        print("%3.2f K" % temps[n])
        rep_inds = [np.where(x[1:] == n)[0][0] for x in swap_history]
        
        # case-1: when frames are dumped faster than temp. swaps
        if writefreq <= nswap:
            for ii, i in enumerate(rep_inds[:-1]):
                start = int(ii * nswap / writefreq)
                stop = int( ii+1) * nswap / writefreq)
                [master_frame_dict[idx].append( (i,x) ) for x in range(start, stop)]
        
        # case-2: when temps. are swapped faster than dumping frames
        else:
            nskip = int(writefreq / nswap)
            [master_frame_dict[n].append( (i,ii) ) for ii, i in enumerate(rep_inds[0::nskip])]
    
    return master_frame_dict


def get_byte_index(rep_inds, byteindfns, intrajfns):
    """
    Get byte indices from (un-ordered) trajectories.
    
    :param rep_inds: indices of replicas to process on this proc
    
    :param byteindsfns: list of filenames that will contain the byte indices
    
    :param intrajfns: list of (unordered) input traj filenames
    """
    for n in rep_inds:
        # check if the byte indices for this traj has aleady been computed
        if os.path.isfile(byteindfns[n]): continue
        
        # extract bytes
        fobj = readwrite(intrajfns[n]) 
        byteinds = [ [0,0] ]
        
        # place file pointer at first line
        nframe = 0
        first_line = fobj.readline()
        cur_pos = fobj.tell()
        #TODO: print the log from ROOT proc only
        pb = tqdm(desc = "Reading replica %d" % n, leave = True, position = ROOT + 2*me,
                  unit = "B", unit_scale = True, unit_divisor = 1024)
        
        # start crawling through the bytes
        while True:
            next_line = fobj.readline()
            if len(next_line) == 0: break
            # this will only work with lammpstrj traj format.
            # this condition essentially checks periodic recurrences of the token TIMESTEP.
            # each time it is found, we have crawled through a frame (snapshot)
            if next_line == firstline:
                nframe += 1
                byteinds.append( [nframe, cur_pos] )
                pb.update()
            cur_pos = fobj.tell()
            pb.update(0)
        pb.close()
        
        # take care of the EOF
        cur_pos = fobj.tell()
        byteinds.append( [nframe+1, cur_pos] ) # dummy index for the EOF
        print("\n") #TODO: do this only from the ROOT proc
        
        # write to file
        np.savetxt(byteindfns[n], np.array(byteinds), fmt = "%d")
        
        # close the trajfile object
        fobj.close()
        
        return


def write_reordered_traj(temp_inds, outtemps, temps, 
                         frame_dict, nprod, writefreq,
                         outtrajfns, infobjs):
    """
    Reorders trajectories by temperature and writes them to disk
    
    :param temp_inds: list index of temps (in the list of all temps) for which 
                       reordered trajs will be produced on this proc.
    
    :param outtemps: list of all temps for which to produce reordered trajs.
    
    :param temps: list of all temps used in the REMD simulation.
    
    :param outtrajfns: list of filenames for output (ordered) trajs.
    
    :param frame_dict: dict containing a tuple (replica #, frame #) 
                       for each temperature
    
    :param nprod: number of production timesteps. Last (nprod / writefreq) number of 
                  frames from the end will be written to disk.
    
    :param writefreq: traj dump frequency in LAMMPS
    
    :param infobjs: list of file pointers to input (unordered) trajs.
    """
    
    for n in temp_inds:
        # open string-buffer and file
        buf = StringIO()
        of = readwrite(outtrajfns[n], mode = "wb")
        
        # get frames
        abs_temp_ind = np.argmin( abs(temps - outtemps[n]) )
        this_frames = frame_dict[abs_temp_ind]
        # retain only production frames
        if nprod:
            # write only from ROOT proc
            if me == ROOT: print("\nRetaining last %d frames" % int(nprod / writefreq) )
            this_frames = this_framelist[-int(nprod/writefreq) : ]
        
        # write frames to buffer
        #TODO: print this only from root
        pb = tqdm(this_frames, desc = ("Buffering traj at %3.2f K") % temps[abs_temp_ind],
                  leave = True, position = ROOT + 2*me,
                  unit = ' frame', unit_scale = True)
        for i, (rep, frame) in enumerate(pb):
            start_ptr = int(byteinds[rep][frame,1])
            stop_ptr = int(byteinds[rep][frame+1,1])
            byte_len = stop_ptr - start_ptr
            infobs[rep].seek(start_ptr)
            buf.write(infobjlist[rep].read(byte_len))
        pb.close()
        
        # write buffer to disk
        print("\nWriting buffer to file")
        of.write(buf.getvalue())
        of.close()
        buf.close()
    
    for i in infobjs: i.close()
    
    return
        

def write_reordered_energies():
    """
    """
    pass


def get_logw():
    """
    """
    pass















# accept user inputs
parser = argparse.ArgumentParser(description = __doc__,
                         formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument('prefix', help = 'Prefix of REMD trajectories (supply full path if not running inside Traj-dir)')
parser.add_argument('nswap', type = int, help = 'Swap frequency')
parser.add_argument('nwrite', type = int, help = 'Dump file writing frequency')
parser.add_argument('-p', '--nprod', type = int)
parser.add_argument('-t', '--temps', nargs = '+', type = float, help = 'Produce trajectories at these temperatures')
parser.add_argument('-logw', '--getlogw', action = 'store_true', help = 'calculate log weights?')
parser.add_argument('-e', '--enefile', help = 'file containing NFrame X NRep array of energies')
parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Verbosity level')

# parse inputs
args = parser.parse_args()
TrajPrefix = os.path.abspath(args.prefix)
NStepsSwap = args.nswap
WriteFreq = args.nwrite
NStepsProd = args.nprod
ReorderTemps = args.temps
EneFile = args.enefile
getlogw = args.getlogw
Verbose = args.verbose


# file formats
REPLICAFMT = "%s.lammpstrj.%d.gz"
TRAJFMT = "%s.%3.2f.lammpstrj.gz"
LOGFMT = "%slammps.log"
ENEFMT = "%s.%3.2f.ene.dat.gz"




# set file paths and default values
TrajDir = os.path.dirname(TrajPrefix)
TempFile = os.path.join(TrajDir, 'temps.txt')
Temps = np.loadtxt(TempFile)
NRep = len(Temps)
if ReorderTemps is None: ReorderTemps = Temps
NReorder = len(ReorderTemps)
LogFn = LOGFMT % TrajPrefix
ReplicaFnList = [REPLICAFMT % (TrajPrefix, i) for i in range(NRep)]
LogFnList = ['%s.%d' % (LogFn, i) for i in range(NRep)]

# output files
NearestTemp = lambda T: Temps[ np.argmin(abs(Temps-T)) ]
OutFnList = [TRAJFMT % (TrajPrefix, NearestTemp(i)) for i in ReorderTemps]
OutEneFnList = [ENEFMT % (TrajPrefix, NearestTemp(i)) for i in ReorderTemps]
ByteIndFnList = [os.path.join(TrajDir, '.byteind%d.gz' % i) for i in range(NRep)]
if getlogw: LogWFn = TrajPrefix + '.logw.pickle'


#########################################
######## REORDERING TRAJECTORIES ########
#########################################

# get frames for each replica visited by a particular temp and do this for all temps
# pretty darn fast so run from ROOT
MasterRepInds = np.loadtxt(LogFn, skiprows = 3)
FrameList = {}
if me == ROOT:
    # walk through replicas
    if Verbose: print ('\nGetting replica indices at: ',)
    for TempInd in range(NRep):
        if Verbose: print ('%3.2fK' % Temps[TempInd], )
        RepInds = [np.where(x[1:] == TempInd)[0][0] for x in MasterRepInds]
        this_FrameList = []
        # case1: 
        if WriteFreq <= NStepsSwap:
            RepInds = RepInds[:-1] # can calculate stop for last frame
            for ii, i in enumerate(RepInds):
                start = int (ii * NStepsSwap / WriteFreq)
                stop = int ( (ii+1) * NStepsSwap / WriteFreq)
                [this_FrameList.append( (i, x) ) for x in range(start, stop)]
        # case2: 
        else:
            NSkip = int(WriteFreq / NStepsSwap) 
            for ii, i in enumerate(RepInds[0::NSkip]):
                this_FrameList.append( (i,ii) )
        # store
        FrameList[TempInd] = this_FrameList
        
# broadcast to all procs
FrameList = comm.bcast(FrameList, root = ROOT)
if me == ROOT and Verbose: print ('\n')

# define a chunk of replicas for each proc
ChunkSize = int(NRep / nproc) # usually less nproc <= NRep
if me < nproc-1: myRepInds = range( (me*ChunkSize), ((me+1)*ChunkSize) )
else: myRepInds = range( (me*ChunkSize), NRep )

# get byte indices from replica trajectories in parallel
for RepInd in myRepInds:
    # check if data already present
    if os.path.isfile(ByteIndFnList[RepInd]): continue
    # else extract byte indices
    fobj = FileOpen(ReplicaFnList[RepInd])
    byteinds = [ [0, 0] ]
    FirstLine = fobj.readline()
    CurPos = fobj.tell()
    NFrame = 0
    if Verbose:
        pb = tqdm(desc = 'Reading replica %d' % RepInd, leave = True,
                  position = ROOT + 2*me, unit = 'B', unit_scale = True, unit_divisor = 1024)
    while True:
        NextLine = fobj.readline()
        if len(NextLine) == 0: break
        if NextLine == FirstLine:
            NFrame += 1
            byteinds.append( [NFrame, CurPos] )
            if Verbose: pb.update()
        CurPos = fobj.tell()
        if Verbose: pb.update(0)
    if Verbose: pb.close()
    # dummy index pointing to the EOF
    CurPos = fobj.tell()
    byteinds.append( [NFrame+1, CurPos] )   
    if Verbose: print ('\n')
    # write to file
    byteinds = np.array(byteinds)
    np.savetxt(ByteIndFnList[RepInd], byteinds, fmt = '%d')
    # cleanup
    fobj.close()

# block until all procs have finished
comm.barrier()
if Verbose: print ('\n')

# open all replica files for reading
fobjList = [FileOpen(i) for i in ReplicaFnList]

# open all byteind files
byteinds = {}
for ii, i in enumerate(ByteIndFnList): byteinds[ii] = np.loadtxt(i)

# define a chunk of output trajectories for each proc
# number of temps to reorder at, may be lesser than available procs
ChunkSize = int(NReorder / nproc)
if ChunkSize == 0:
    nproc_effective = NReorder
    ChunkSize = 1
    if me == ROOT: print ('\nExcess procs found; releasing %d procs' % (nproc - nproc_effective))
else:
    nproc_effective = nproc
if me < nproc_effective-1: myTempInds = range( (me*ChunkSize), ((me+1)*ChunkSize) )
else: myTempInds = range( (me*ChunkSize), NReorder )

# retire excess procs
if me >= nproc_effective:
    for i in fobjList: i.close()
    exit()

# keep necessary procs active in reordering in parallel
else:
    for TempInd in myTempInds:
        # open string-buffer and file
        buf = StringIO.StringIO()
        of = FileOpen(OutFnList[TempInd], 'wb')
        # get frames
        this_Temp = ReorderTemps[TempInd]
        AbsTempInd = np.argmin(abs(Temps - this_Temp))
        this_FrameList = FrameList[AbsTempInd]
        # retain only prodsteps
        if not NStepsProd is None:
            if me == ROOT and Verbose: print ('\nRetaining last %d steps') % int(NStepsProd / WriteFreq)
            this_FrameList = this_FrameList[-int(NStepsProd/WriteFreq) : ]
        # write frames
        if Verbose:
            pb = tqdm(this_FrameList, desc = ('Buffering traj at %3.2fK') % Temps[AbsTempInd], 
                      leave = True, position = ROOT + 2*me, 
                      unit = ' frame', unit_scale = True)
        else: pb = this_FrameList
        #for i, (Rep, Frame) in enumerate(this_FrameList):
        for i, (Rep, Frame) in enumerate(pb):
            StartPtr = int(byteinds[Rep][Frame, 1])
            StopPtr = int(byteinds[Rep][Frame+1, 1])
            ByteLen = StopPtr - StartPtr
            fobjList[Rep].seek(StartPtr)
            buf.write(fobjList[Rep].read(ByteLen))
        if Verbose: pb.close()
        # write buffer data to file
        print ('\nWriting buffer file')
        of.write(buf.getvalue())
        # clear stuff
        of.close()
        buf.close()

    # cleanup
    for i in fobjList: i.close()


#########################################
######## CALCULATING LOG WEIGHTS ########
#########################################

# usually working with numpy arrays at this point
# so everything is fast. retire all but the root proc
if not me == ROOT or EneFile is None: exit()

# reorder energies
ene = np.loadtxt(EneFile)
if Verbose: print ('\nReordering energies from supplied energy file (on root proc)')
K = len(Temps)
if NStepsProd is not None:
    n = int(NStepsProd/WriteFreq)
    print ('\nRetaining last %d steps' % n)
else:
    n = len(FrameList[0]) # assume all trajectories have same nframes
u_kn = np.zeros([K, n])
for TempInd in range(K):
    if Verbose: print ('%3.2f K' % Temps[TempInd],)
    this_FrameList = FrameList[TempInd]
    if not NStepsProd is None:
        this_FrameList = this_FrameList[-int(NStepsProd/WriteFreq) : ]
    for i, (Rep, Frame) in enumerate(this_FrameList):
        u_kn[TempInd, i] = ene[Rep, Frame]

# write reordered energies to file at requested temps
if Verbose: print ('\nWriting re-ordered energies to file')
for i, t in enumerate(ReorderTemps):
    print ('%3.2f K' % NearestTemp(t))
    this_ind = [j for j, t_ in enumerate(Temps) if '%3.2f' % t_ == '%3.2f' % NearestTemp(t)][0]
    print (this_ind)
    this_outfile = OutEneFnList[i]
    this_ene = u_kn[this_ind, :]
    np.savetxt(this_outfile, this_ene, fmt = '%14.7e')
    
# see if log-weight calculation is requested
if not getlogw: exit()

# get reduced energies
kB = 0.001987
beta_k = 1.0 / (kB * Temps)
N = u_kn.shape[1]
N_k = N * np.ones(K, np.uint8)
u_kln = np.zeros([K,K,N], float)
for k in range(K):
    for l in range(K):
        u_kln[k, l, 0:N_k[k]] = beta_k[l] * u_kn[k, 0:N_k[k]]

# initialize mbar object
if Verbose: print ('Calculating config weights...')
mbar = pymbar.mbar.MBAR(u_kln, N_k, verbose = Verbose)
f_k = mbar.f_k

# get logweights
logw_kn = {}
for k in range(K):
    logw_kn[k] = whamlib.log_weight(ekn = u_kn, betak = beta_k, fk = f_k,
                                    targetbeta = beta_k[k])

# store these in a zipped txt file so that numpy can read
with open(LogWFn, 'w') as of:
    pickle.dump(logw_kn, of)
    
