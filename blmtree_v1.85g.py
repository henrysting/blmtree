#! /usr/bin/python
# coding = utf-8 

# Blooming Tree Hierarchical Structure Analysis Script (2025.03 v1.85g)
# Script to detect structures with spectroscopic redshifts of galaxies.
# Taking the projected binding energy as the linking length.
# Developed by Yu Heng @ Beijing Normal University 2015-2025 
# with Prof. Antonaldo Diaferio of Turin University, 
# and his Caustic Group: http://www.dfg.unito.it/ricerca/caustic/
# ---------------------------------------------------------------
# Usage: 
# blmtree.py list test    # check and summarize the input catalog
# blmtree.py list dEta=20 # cut the tree with the blooming tree algorithm, with a given threshold 20
# blmtree.py list dEta=50 mem=20 # set 20 as the minimal number of members in a group, 10 by default.  
# 
# Other cutting approaches are also supported
#
# blmtree.py list bEne=-1 # cut the tree with the binding energy
# blmtree.py list vdis=50 # cut the tree with the velocity disperion
# blmtree.py list mem=10  # cut the tree with the sigma plateau, valid for fields with one main object 
# --------------------------------------------------------------
# 
# 2015.10.24: EdS binding energy
# 2015.10.31: redshift space plot
# 2016.01.21: calculate binding energy with RW metric; unify the usage of H0
# 2016.06.06: improve for individual cluster analysis 
# 2017.05.03: remove critical binding energy
# 2017.05.19: fixed distance bug
# 2017.05.22: add circle plot
# 2017.06.12: minor updates.
# 2017.06.24: precheck selection, redshift diagram
# 2017.07.02: redshift independent, work with zrange when neccessary
# 2017.12.07: uniform structure detection
# 2018.03.06: first release 
# 2018.03.21: module astropy removed
# 2018.05.19: add status bar for energy calculation, support caustic data input
# 2018.05.24: auto-range redshift distribution, fix frw distance 
# 2018.05.27: optimize header filter
# 2018.10.31: save pairwise energy into a binary file
# 2018.11.11: move member limitation filter to the plot stage
# 2019.03.28: synchronize with the star cluster version
# 2019.05.28: update 2D projection map
# 2019.06.04: add substructures plots with velocity dispersion filter
# 2019.06.06: minor correction on data filter.
# 2019.06.08: add substructure illustration
# 2019.07.08: remove nearby false detections
# 2019.07.10: add sector plot and adjust basic figure output.
# 2019.09.23: norm.pdf is replaced by the scipy.stats.norm.pdf since matplotlib 3.1.0
# 2019.09.30: add labels to zoomed plot
# 2019.10.05: calculate d_avg with celestial distantce instead of Euclidean
# 2019.11.20: fix field radius calculation 
# 2020.05.30: spatial plot with weighted the depth of potential well
# 2020.06.03: clarify 1D and 2D plots
# 2020.09.07: skip galaxies within 100 Mpc (z < 0.024)  
# 2020.09.26: transfer the geometrical longtitude of basemap plots into the RA format
# 2021.02.03: replace the structure centroid calculation algorithm --smallest enclosing circle with median position.
# 2021.02.18: update xticklabels of simplified hierarchical structure plot, fixed bugs
# 2021.04.10: remove astropy completely, and fix inconsistence in zl calculation.
# 2021.06.12: update output format, keep consistent with v1.7g release
# 2021.11.23: add low limit of velocity dispersion (vdisLim)
# 2021.12.18: save nodes properties of simplified tree
# 2021.12.28: add log file for analysis, add the parameter vdisPlot
# 2022.03.22: add support of simplified tree ploting for other trimming threshold  
# 2022.12.15:    minor adjustments
# 2023.07.18: fix a bug of find parents
# 2023.10.20: fix a bug of radius calculation strucrad
# 2023.11.23: keep SDSSID correct, save eta and delta_eta in float instead of int
# 2024.01.06: support multiple separators of the input file
# 2024.01.20: optimize name of output files
# 2025.03.19: Fix the possible mismatch between gsub and glis/gsum when the number of members is equal

import matplotlib as mpl
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import scipy.cluster.hierarchy as hac
from scipy.stats import norm

import struct
import numpy as np
import sys,os

##http://stackoverflow.com/questions/10204173/plotting-dendrogram-in-scipy-error-for-large-dataset
sys.setrecursionlimit(50000)   # to enable dendrogram to deal large data

### http://www.w3schools.com/html/html_colornames.asp
### the first 9 colors are compatible with ds9
color_cl = ["blue","red","green","yellow","cyan","magenta","orange","purple",'brown',"SlateGray","Lime","Crimson","Coral","Darkblue","Darkred","Darkgreen","Darkcyan","sienna"]

light_speed = 2.99792458E+5  #speed of light, in km/s, z = v/c
Mpc =  3.08567802e+22 # in m
G = 6.67408E-11 # gravitational constant : in m^3 /kg /s^2
msun = 1.9885E+30  # solar mass, in kg
H0 = 67.4 # hubble constant, Planck18
mgal= 1.E+11 * msun / (H0/100) # 2.76E+41 kg,   galaxy mass unit
energy_c = 0 # energy threshold, 0 by default

def precheck(fname):
    """precheck if the input data contains any incomplete rows"""
    lines = open(fname).readlines()
    nmag = 0  #  indicator of the mag columns
    ### check delimiter 
    possible_separators = [" ", ',', ';', '\t']  # common separators
    delim = ""
    for separator in possible_separators:
        data = lines[3].strip().split(separator)
        if len(data) > 1:
            delim = separator
            break    
    ### check header 
    nh = 0    # indicator of the head row
    if lines[0][0]!="#" and lines[0].strip()!="":
        data = lines[0].strip().split(delim)
        #~ print(data)
        if float(data[0])==len(lines)-1: 
            print("There is a summary line in the input catalog.")
            nh = 1  

                
    for line in lines:
        if line[0]!="#" and line[0].strip()!="":
            data = lines[3].strip().split(delim)
            if float(data[2])<=0:
                print("Warning: non-positive redshift: {0}".format(line))
            if len(data) < 3: 
                #~ print("Warning: This line does not have enough columns:", line)              
                sys.exit("Error: This line does not contain enough data: \n{0}".format(line))
            if len(data) > 4:
                nmag = 1
    
    print("The format of "+fname+" seems fine.")
    return nh,nmag,delim

def circle_r(data):
    """calculate the center and radius of celestial objects"""
    """outputs are in radian"""
    data = np.radians(data)
    centroid = np.median(data[:,:2],axis=0)
    ## this formula will lose numerical accuracy very quickly when theta << 1, Haversine formula, or Vincenty's formulae instead
    #### the largest radius 
    theta = np.max(np.arccos(np.cos(centroid[1])*np.cos(data[:,1])*np.cos(centroid[0]-data[:,0]) + np.sin(centroid[1])*np.sin(data[:,1])))
    #~ print(centroid,theta)
    return np.r_[centroid,theta]
    
def harmonic_r(data)    :
    """harmonic radius from 1993AJ....105.2035D"""
    data = np.radians(data)
    #~ centroid = np.median(data[:,:2],axis=0)
    centroid = np.mean(data[:,:2],axis=0)
    theta = 0
    num = len(data)
    for i in range(num-1):
        theta = theta + np.sum(1/np.arccos(np.cos(data[i,1])*np.cos(data[i+1:,1])*np.cos(data[i,0]-data[i+1:,0]) + np.sin(data[i,1])*np.sin(data[i+1:,1])))
    
    hr = np.pi/4 * num *(num-1)/theta
    return np.r_[centroid,hr]
    
def filterdata2(fname,zrange="",nfile="test.g00",nhead=0,sep=None):
    """filter data with given redshift range, work with RA(deg) DEC(deg) cz(km/s) """
    sig_rad = "degree"
    sig_v = "velocity"
    ### summary ###
    print('>>>>>>>>>>>> catalog summary  >>>>>>>>>>')
    if sep== " ":
        sep = None
    data = np.loadtxt(fname,skiprows=nhead,delimiter=sep)   ###  TOD
    
    if np.std(data[:,2]) < 10:   # in z
        sig_v = "redshift"
        ls = 1 
    else:
        sig_v = "cz"
        ls = light_speed
    
    print('{0} records in the {3} range of ({1}, {2})'.format(len(data),np.round(np.min(data[:,2]),2),np.round(np.max(data[:,2]),2),sig_v))
        
    ## for fields less than 20 x 20 degree
    print("Catalog centroid: {}".format(np.median(data[:,:3],axis=0)))
    print("RA range: {:.4f} ~ {:.4f}".format(np.min(data[:,0]),np.max(data[:,0])))
    print("DEC range : {:.4f} ~ {:.4f}".format(np.min(data[:,1]),np.max(data[:,1])))
    print('Spatial coordiantes are in the range of {0:.2f} x {1:.2f} deg'.format(np.max(data[:,0])-np.min(data[:,0]), np.max(data[:,1])-np.min(data[:,1])))
    ### for the real radius
    centroid = circle_r(data[:,:2])*180/np.pi
    print('Field radius of the sample is {0:.3f} deg, or {1:.2f} arcmin centered on ({2:.6f},{3:.6f})'.format(centroid[2],centroid[2]*60,centroid[0],centroid[1]))    #
    
    lines = open(fname).readlines()[nhead:]
    nf = open(nfile,"w")
    ndata = 0
    if zrange == "":
        zrange=[0,np.max(data[:,2])/ls]
        
    for line in lines:
        if line[0] != "#" :
            sdata = line.strip().split(sep)
            #~ print(sdata)
            if float(sdata[2])>=zrange[0]*ls and float(sdata[2])<=zrange[1]*ls:       # cz within the range cz+[-zrange,zrange]
                sdata[2] = str(float(sdata[2])*light_speed/ls)                        # alwasys save in cz(km/s)
                nf.write(" ".join(sdata)+"\n")
                ndata = ndata +1

    print("There are {:.0f} galaxies in the redshift range of ({:.4f}, {:.4f}) ".format(ndata, zrange[0],zrange[1]))
    print("They are withdrawn and saved in {0}".format(nfile))
    return sig_rad, sig_v


def vpdict(fname):
    """original positions and velocity dictionary"""
    data = np.loadtxt(fname)
    vpdict = dict()
    if data[0,2] < 10:
        data[:,2] = data[:,2] * light_speed
    for i in range(len(data)):
        vpdict[i]=data[i,:]
    return vpdict

def flatten(lst):
    return sum( ([x] if not isinstance(x, list) else flatten(x)
            for x in lst), [] )

def strucrad(points):
    """calculate average distance between structure members, both input and output are radians """
    data = np.array(points[:,:2])
    data = np.radians(data)
    #~ print(data[:5,:])
    n = len(data)
    dis_sum = 0
    if n >1:
        for i in range(n-1):
            theta = np.arccos(np.cos(data[i,1])*np.cos(data[i+1:,1])*np.cos(data[i,0]-data[i+1:,0]) + np.sin(data[i,1])*np.sin(data[i+1:,1]))
            dis_sum = dis_sum + np.sum(theta)
        davg = dis_sum/n/(n-1)
    else:
        davg = -1
        
    return davg

def avmin(ratio,thr=20):
    """minimum of average density"""
    
    r_jump = ratio[:-1]-ratio[1:]   # neighbour jump, more compact, more reliable membership
    node_str_jump = np.flatnonzero(r_jump > thr)

    if len(node_str_jump) > 0:
        node_str = node_str_jump[-1]     # the last large jump down
        vjump = r_jump[node_str]
    else:
        node_str = -1
        vjump = -1

    return [node_str,vjump]


def find_children(data,node):
    """return nodes id (begin from 0) hanging below the given Node"""
    """work with *.gtre data"""
    node_set = []  # nodes
    leaf_set = []  # leaves
    n_leaf = len(data)+1
    k = int(node)
    if k > n_leaf-1:
        node_set.append(k)
            
        for i in [0,1]:
            #~ print(i,k-n_leaf,data[k-n_leaf,i])
            nset = find_children(data,data[k-n_leaf,i])
            node_set = node_set + nset[0]
            leaf_set = leaf_set + nset[1]
    else:
        leaf_set.append(k)        
        
    return [node_set,leaf_set]

def minode(fname,thr=20,memLim=9,vdisLim=200):
    """trace local minimums of binding energy profile of the tree,
    return the minimum nodes of these branches
    Leaves ID from 0 ~ N-1   ; Nodes ID from N ~ 2N-2, 
    """
    from scipy.signal import argrelextrema
    
    data = np.loadtxt(fname+".gtre")
    mdata= np.loadtxt(fname+".gepr")  # tree profile
    n_leaf = len(data)+1
    
    min_loc = argrelextrema(mdata[:,2], np.less, order=2)[0]  ## minimum index
    
    min_ind= min_loc[mdata[min_loc,2] < energy_c]
    print("{0} local minimums are found within {1} nodes.".format(len(min_ind), n_leaf))
    node_str = []
    nodes = []  # the branches contain nodes
    
    jump_set = {}
                   
    fpath = "./branches"   # figure path
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    
    min_g = 1E+9
    max_n = 0
    
    ### trace each minimum
    print("start tracing minima with threshold {0}".format(thr))

    fbrch=open(fname+"_dEta_thr"+str(thr)+"_M"+str(memLim)+".gbrc",'w')
    feta=open(fname+"_dEta_thr"+str(thr)+"_M"+str(memLim)+".getap",'w')
    for i in range(len(min_ind)):
        v_set = []  # velocity dispersion set 
        n_set = []  # number of leaves
        d_set = []  # average distance set
        r_set = []  # structure raidus set
        E_set = []  #  binding energy
        da_set = []  # branch distance set 
        id_set = [] # set of sequence, begin from 0
        k = mdata[min_ind[i],0]     # particles ID, begin from 0
        #~ ind = np.where(data[:,1:3]==k)[0][0]   # line number, begin from 0
        ######## search nodes above given node
        while (k < 2*n_leaf-2): # and (data[ind,6] < 4000):
            #~ print(k,np.where(data[:,1:3]==k))
            ind = np.where(data[:,1:3]==k)[0][0]
            id_set.append(ind)
            E_set.append(data[ind,3]) 
            v_set.append(data[ind,6])    # NOTICE: not the value of node k 
            n_set.append(data[ind,4])
            d_set.append(data[ind,7])
            r_set.append(data[ind,8])
            da_set.append(data[ind,9])
            k = ind + n_leaf
            
        if len(id_set) >0:
            id_set = np.array(id_set)
            v_set = np.array(v_set)
            n_set = np.array(n_set)
            d_set = np.array(d_set)
            r_set = np.array(r_set)
            E_set = np.array(E_set)
            ratio = v_set / np.sqrt(n_set) / (d_set+1E-9)   ####  eta
            r_ang = r_set * 180 / np.pi / da_set  ## real size of the branch
                    
            lim_id = int(mdata[min_ind[i],0])
            minset = avmin(ratio[E_set<energy_c],thr)
            #~ plt.figure()
            fig, ax1 = plt.subplots()
            #~ ax2 = ax1.twinx()

            ###plt.plot(v_set[ind_b[0]:],ratio[ind_b[0]:],'-o')
            ax1.plot(range(len(n_set)),n_set,'g--',lw=2,label="$n$")
            ax1.plot(range(len(d_set)),100*d_set,'b-.',lw=2,label="$100d_{avg}$")
            ax1.plot(range(len(r_set)),10*r_set,'m-.',lw=2,label="$10R_{prj}$")
            ax1.plot(range(len(v_set)),v_set/2,'c:',lw=2,label="$\sigma_v/2$")
            
            ax1.plot(range(len(ratio)),ratio,'r-',lw=2,label="$\eta$")
            ax1.plot(range(len(ratio)-1),ratio[:-1]-ratio[1:],'-',color="orange",lw=4,label="$\Delta \eta$")
            
            ax1.plot(range(len(ratio)),ratio,'k.',markersize=0.5)
            ax1.set_ylim([0,500])
            
            if np.sum(E_set<energy_c)>0:
                ax1.axvline(x=np.where(E_set<energy_c)[0][-1],color="k")
            else:
                ax1.axvline(x=0,color="k",label="E_b=0")
            ax1.axhline(y=50,color="k")
            # find at least one minimum
            if (minset[0]>0):
                # contain enough members
                nmem = len(find_children(data[:,1:],id_set[minset[0]]+n_leaf)[1])
                #~ print(nmem,d_set[minset[0]])
                if  (nmem> memLim) and (v_set[minset[0]] >vdisLim): # and (r_ang[minset[0]]/r_ang[-1] < 0.9):   #### structures should be smaller than 90% of the field
                    # if it is not included in the set
                    if (id_set[minset[0]] not in node_str):  
                        node_str.append(id_set[minset[0]])
                        jump_set[id_set[minset[0]]]=[minset[1],ratio[minset[0]+1]]  ### value of jump and base
                        
                ####   bottom leaves, sub thr, cl thr    
                print("branch:", lim_id, "contains",nmem,"leaves, key str id:",id_set[minset[0]]+n_leaf)
                #~ print(r_ang[minset[0]:])
                fbrch.write(" ".join(map(str,[lim_id,nmem,id_set[minset[0]]+n_leaf, r_set[minset[0]]]))+"\n")
                
                ax1.plot(minset[0],ratio[minset[0]],'ys',markersize=5,label="str node",alpha=0.6)
                ax1.axvline(x=minset[0],color="y")
                    
            ### plt.gca().invert_xaxis()
            plt.title("Branch:"+str(lim_id))
            ax1.set_ylabel("$\eta$")
            ax1.set_xlabel("Node Sequence")
            l = ax1.legend(numpoints=1,ncol=2)            
            l.set_zorder(20) # put the legend on the top
            
            plt.savefig(fpath+"/"+fname+"_"+str(lim_id)+"_thr"+str(thr)+".pdf")
            plt.close(fig)
            feta.write("\t".join(map(str,[lim_id]+list(ratio)+list(n_set)))+"\n")
            
    fbrch.close()
    feta.close()
    #~ # remove nodes on the upper branches
    str_u= unode(node_str,fname,jump_set,thr=thr,memLim=memLim,vdisLim=vdisLim)
    #~ print(node_str,str_u)
    ######## begining of branches, key min nodes
    return  [min_ind, str_u]

def minode3(fname,thr=0,indx=6,memLim=9):
    """trim the tree with a given v_disp (indx=6)/binding energy(indx=3) threshold,
    check branches directly.
    return the minimum nodes of these branches
    Leaves ID from 0 ~ N-1   ; Nodes ID from N ~ 2N-2, 
    """
    
    data = np.loadtxt(fname+".gtre")
    linkage_matrix = np.c_[data[:,1],data[:,2],data[:,indx],data[:,4]]
    n_leaf = len(data)+1
    
    #############
    dict_node = dict(zip(data[:,0],data[:,indx])) 
    dict_leaf = dict(zip(range(n_leaf),[0]*n_leaf))
    dict_vdis = dict_node.copy()
    dict_vdis.update(dict_leaf)
    
    #############
    dict_node2 = dict(zip(data[:,0],data[:,4])) 
    dict_mem = dict_node2.copy()
    dict_mem.update(dict_leaf)
                    
    tmp = np.array([dict_vdis[int(x)] for x in data[:,1]])
    tmp2 = np.array([dict_mem[int(x)] for x in data[:,1]])
    branches_l = np.c_[data[:,0],data[:,1],data[:,indx],tmp,tmp2]
    tmp = np.array([dict_vdis[int(x)] for x in data[:,2]])
    tmp2 = np.array([dict_mem[int(x)] for x in data[:,2]])
    branches_r = np.c_[data[:,0],data[:,2],data[:,indx],tmp,tmp2]
    branches = np.r_[branches_l,branches_r]
    
    knode1 = branches[(branches[:,2]>= thr) *(branches[:,3] < thr) * (branches[:,4] > memLim)]
    knode2 = branches[(branches[:,2]< thr) *(branches[:,3] >= thr) * (branches[:,4] > memLim)]
    knode = list(knode1[:,1])+list(knode2[:,1])
    ######### remove sub-nodes
    knode_s = np.array(sorted(list(map(int,knode)),reverse=True))
    #~ print(knode_s)
    mask = knode_s > 0
    knode_f = []
    for i in range(len(knode_s)):
        [nodes,leaves] = find_children(linkage_matrix,knode_s[i])
        knode_f.append([knode_s[i],len(leaves)])
        for j in range(i+1,len(knode_s)):
            if knode_s[j] in nodes:
                mask[j] = False
    
    ###########  sort with member number #########

    knode_f = np.array(knode_f)[mask]
    #~ print(mask,knode_f)
    return sorted(knode_f,key=lambda x:(x[1], x[0]),reverse=True)
    
def unode(nodes,fname,jump_set,thr=10,memLim=9,vdisLim=200):
    """check unique key node which does not contain any other nodes"""
    data = np.loadtxt(fname+".gtre")
    n_leaf = len(data)+1
    node_subu = [] ## unique substrucuture node
    for node in set(nodes):
        ### the number of substructure nodes hanging below node
        node_below = find_children(data[:,1:],node+n_leaf)
        node_id = list(np.array(nodes)+n_leaf)
        # check a node contain others or not
        n_dup = len(set(node_below[0]))+len(set(nodes))-len(set(node_below[0]+node_id))
        n_leaves = len(node_below[1])
        vjump = jump_set[node]
        # do not contain other key nodes
        if n_dup < 2:
            node_subu.append([node,n_leaves,vjump[0],vjump[1]]) 
    node_subf = np.array(sorted(node_subu,key=lambda x: (x[1], x[0]),reverse=True))   
    
    pid_set = np.loadtxt(fname+"_dEta_thr"+str(thr)+"_M"+str(memLim)+".gbrc",ndmin=2)
    pid_dict = {}
    if len(node_subf) > 0:
        node_subid = node_subf[:,0].astype("int")
        
        ## generate source id dict

        for k in range(len(pid_set)):
            pid_dict[pid_set[k,2]]=int(pid_set[k,0])

        # save structure list
        fsub = open(fname +"_dEta_thr"+str(thr)+"_vdisL"+str(vdisLim)+"_M"+str(memLim)+".gsub","w")
        fsub.write("#sID, minID, nodeID, lkid, rkid, E_bind, N_mem, z, Vdis, r_avg, r_cl(Mpc), dist(Mpc), centroid_x, centroid_y, Deta, base\n")

        for i in range(len(node_subf)):
            data_node = data[data[:,0]==(node_subf[i,0]+n_leaf),:][0]
            data_node[5]=np.round(data_node[5]/light_speed,4)
            output = ("{:3d}"+" {:6.0f}"*4+" {:7.1f} {:4.0f} {:8.4f}"+" {:9.4f}"*8).format(i+1,pid_dict[int(data_node[0])],*data_node,*node_subf[i,2:4])
            #~ print "sub:",output
            fsub.write(output+"\n")
        fsub.close()
    else:
        node_subid = []
        
    return node_subid
    
def cdistance2(z,model="frw"):
    """the comoving distance of redshift z within certain metric
    input should be numpy array, redshift in radian"""
    from astropy.cosmology import WMAP9 as cosmo
    z = np.array(z)
    dc_set = []
    for zi in z:
        if model=="eds":
            # in the EdS universe         
            dc = 2*(1-1/np.sqrt(1+zi))*light_speed*(1E+3)/(H0 * 1000/Mpc) # in m
        elif model=="frw":
            dc = cosmo.comoving_distance(zi).value*Mpc
        dc_set.append(dc)
    
    return np.array(dc_set)

def cdistance3(x, type="z"):
    """calculate comoving_distance with interpolation table Dm_LCDM.dat"""    
    """loading file from disk is slow"""
    from scipy import interpolate
    ### generated with Planck15 of astropy.cosmology 
    data = np.array([0.00000,0.00000,0.00010,0.44260,0.00020,0.88517,0.00030,1.32773,0.00040,1.77027,0.00050,2.21278,0.00060,2.65528,0.00070,3.09775,0.00080,3.54020,0.00090,3.98264,0.00100,4.42505,0.00110,4.86744,0.00120,5.30981,0.00130,5.75217,0.00140,6.19450,0.00150,6.63681,0.00160,7.07910,0.00170,7.52136,0.00180,7.96361,0.00190,8.40584,0.00200,8.84805,0.00210,9.29024,0.00220,9.73240,0.00230,10.17455,0.00240,10.61667,0.00250,11.05878,0.00260,11.50086,0.00270,11.94292,0.00280,12.38497,0.00290,12.82699,0.00300,13.26899,0.00310,13.71097,0.00320,14.15293,0.00330,14.59487,0.00340,15.03679,0.00350,15.47869,0.00360,15.92057,0.00370,16.36243,0.00380,16.80427,0.00390,17.24608,0.00400,17.68788,0.00410,18.12966,0.00420,18.57141,0.00430,19.01314,0.00440,19.45486,0.00450,19.89655,0.00460,20.33822,0.00470,20.77988,0.00480,21.22151,0.00490,21.66312,0.00500,22.10471,0.00510,22.54628,0.00520,22.98783,0.00530,23.42936,0.00540,23.87087,0.00550,24.31235,0.00560,24.75382,0.00570,25.19527,0.00580,25.63669,0.00590,26.07810,0.00600,26.51948,0.00610,26.96085,0.00620,27.40219,0.00630,27.84351,0.00640,28.28482,0.00650,28.72610,0.00660,29.16736,0.00670,29.60860,0.00680,30.04982,0.00690,30.49102,0.00700,30.93220,0.00710,31.37336,0.00720,31.81449,0.00730,32.25561,0.00740,32.69671,0.00750,33.13778,0.00760,33.57884,0.00770,34.01987,0.00780,34.46089,0.00790,34.90188,0.00800,35.34285,0.00810,35.78380,0.00820,36.22473,0.00830,36.66564,0.00840,37.10654,0.00850,37.54740,0.00860,37.98825,0.00870,38.42908,0.00880,38.86989,0.00890,39.31068,0.00900,39.75144,0.00910,40.19219,0.00920,40.63291,0.00930,41.07362,0.00940,41.51430,0.00950,41.95497,0.00960,42.39561,0.00970,42.83623,0.00980,43.27683,0.00990,43.71741,0.01000,44.15797,0.01010,44.59851,0.01020,45.03903,0.01030,45.47953,0.01040,45.92001,0.01050,46.36046,0.01060,46.80090,0.01070,47.24131,0.01080,47.68171,0.01090,48.12208,0.01100,48.56244,0.01110,49.00277,0.01120,49.44308,0.01130,49.88337,0.01140,50.32364,0.01150,50.76389,0.01160,51.20412,0.01170,51.64433,0.01180,52.08452,0.01190,52.52469,0.01200,52.96484,0.01210,53.40496,0.01220,53.84507,0.01230,54.28515,0.01240,54.72522,0.01250,55.16526,0.01260,55.60528,0.01270,56.04529,0.01280,56.48527,0.01290,56.92523,0.01300,57.36517,0.01310,57.80509,0.01320,58.24499,0.01330,58.68487,0.01340,59.12472,0.01350,59.56456,0.01360,60.00438,0.01370,60.44417,0.01380,60.88395,0.01390,61.32370,0.01400,61.76344,0.01410,62.20315,0.01420,62.64284,0.01430,63.08251,0.01440,63.52216,0.01450,63.96179,0.01460,64.40140,0.01470,64.84099,0.01480,65.28056,0.01490,65.72011,0.01500,66.15963,0.01510,66.59914,0.01520,67.03862,0.01530,67.47809,0.01540,67.91753,0.01550,68.35696,0.01560,68.79636,0.01570,69.23574,0.01580,69.67510,0.01590,70.11444,0.01600,70.55376,0.01610,70.99306,0.01620,71.43234,0.01630,71.87160,0.01640,72.31083,0.01650,72.75005,0.01660,73.18924,0.01670,73.62842,0.01680,74.06757,0.01690,74.50671,0.01700,74.94582,0.01710,75.38491,0.01720,75.82398,0.01730,76.26303,0.01740,76.70206,0.01750,77.14107,0.01760,77.58006,0.01770,78.01902,0.01780,78.45797,0.01790,78.89690,0.01800,79.33580,0.01810,79.77469,0.01820,80.21355,0.01830,80.65239,0.01840,81.09122,0.01850,81.53002,0.01860,81.96880,0.01870,82.40756,0.01880,82.84630,0.01890,83.28502,0.01900,83.72371,0.01910,84.16239,0.01920,84.60105,0.01930,85.03968,0.01940,85.47830,0.01950,85.91689,0.01960,86.35546,0.01970,86.79402,0.01980,87.23255,0.01990,87.67106,0.02000,88.10955,0.02010,88.54802,0.02020,88.98647,0.02030,89.42490,0.02040,89.86330,0.02050,90.30169,0.02060,90.74006,0.02070,91.17840,0.02080,91.61672,0.02090,92.05503,0.02100,92.49331,0.02110,92.93157,0.02120,93.36981,0.02130,93.80803,0.02140,94.24623,0.02150,94.68441,0.02160,95.12257,0.02170,95.56071,0.02180,95.99882,0.02190,96.43692,0.02200,96.87499,0.02210,97.31305,0.02220,97.75108,0.02230,98.18910,0.02240,98.62709,0.02250,99.06506,0.02260,99.50301,0.02270,99.94094,0.02280,100.37885,0.02290,100.81673,0.02300,101.25460,0.02310,101.69245,0.02320,102.13027,0.02330,102.56808,0.02340,103.00586,0.02350,103.44363,0.02360,103.88137,0.02370,104.31909,0.02380,104.75679,0.02390,105.19447,0.02400,105.63213,0.02410,106.06977,0.02420,106.50739,0.02430,106.94498,0.02440,107.38256,0.02450,107.82011,0.02460,108.25765,0.02470,108.69516,0.02480,109.13265,0.02490,109.57013,0.02500,110.00758,0.02510,110.44501,0.02520,110.88242,0.02530,111.31981,0.02540,111.75717,0.02550,112.19452,0.02560,112.63185,0.02570,113.06915,0.02580,113.50644,0.02590,113.94370,0.02600,114.38094,0.02610,114.81817,0.02620,115.25537,0.02630,115.69255,0.02640,116.12971,0.02650,116.56685,0.02660,117.00396,0.02670,117.44106,0.02680,117.87814,0.02690,118.31519,0.02700,118.75223,0.02710,119.18924,0.02720,119.62624,0.02730,120.06321,0.02740,120.50016,0.02750,120.93709,0.02760,121.37400,0.02770,121.81089,0.02780,122.24776,0.02790,122.68460,0.02800,123.12143,0.02810,123.55823,0.02820,123.99502,0.02830,124.43178,0.02840,124.86853,0.02850,125.30525,0.02860,125.74195,0.02870,126.17863,0.02880,126.61529,0.02890,127.05193,0.02900,127.48855,0.02910,127.92514,0.02920,128.36172,0.02930,128.79827,0.02940,129.23481,0.02950,129.67132,0.02960,130.10782,0.02970,130.54429,0.02980,130.98074,0.02990,131.41717,0.03000,131.85358,0.03010,132.28997,0.03020,132.72633,0.03030,133.16268,0.03040,133.59901,0.03050,134.03531,0.03060,134.47160,0.03070,134.90786,0.03080,135.34410,0.03090,135.78032,0.03100,136.21652,0.03110,136.65270,0.03120,137.08886,0.03130,137.52500,0.03140,137.96112,0.03150,138.39721,0.03160,138.83329,0.03170,139.26934,0.03180,139.70538,0.03190,140.14139,0.03200,140.57738,0.03210,141.01335,0.03220,141.44930,0.03230,141.88523,0.03240,142.32114,0.03250,142.75703,0.03260,143.19289,0.03270,143.62874,0.03280,144.06456,0.03290,144.50037,0.03300,144.93615,0.03310,145.37191,0.03320,145.80765,0.03330,146.24337,0.03340,146.67907,0.03350,147.11475,0.03360,147.55041,0.03370,147.98604,0.03380,148.42166,0.03390,148.85725,0.03400,149.29283,0.03410,149.72838,0.03420,150.16391,0.03430,150.59942,0.03440,151.03491,0.03450,151.47038,0.03460,151.90583,0.03470,152.34126,0.03480,152.77667,0.03490,153.21205,0.03500,153.64742,0.03510,154.08276,0.03520,154.51808,0.03530,154.95339,0.03540,155.38867,0.03550,155.82393,0.03560,156.25917,0.03570,156.69438,0.03580,157.12958,0.03590,157.56476,0.03600,157.99991,0.03610,158.43505,0.03620,158.87016,0.03630,159.30526,0.03640,159.74033,0.03650,160.17538,0.03660,160.61041,0.03670,161.04542,0.03680,161.48041,0.03690,161.91537,0.03700,162.35032,0.03710,162.78524,0.03720,163.22015,0.03730,163.65503,0.03740,164.08989,0.03750,164.52474,0.03760,164.95956,0.03770,165.39436,0.03780,165.82913,0.03790,166.26389,0.03800,166.69863,0.03810,167.13334,0.03820,167.56804,0.03830,168.00271,0.03840,168.43737,0.03850,168.87200,0.03860,169.30661,0.03870,169.74120,0.03880,170.17577,0.03890,170.61032,0.03900,171.04484,0.03910,171.47935,0.03920,171.91384,0.03930,172.34830,0.03940,172.78274,0.03950,173.21717,0.03960,173.65157,0.03970,174.08595,0.03980,174.52031,0.03990,174.95465,0.04000,175.38897,0.04010,175.82326,0.04020,176.25754,0.04030,176.69179,0.04040,177.12603,0.04050,177.56024,0.04060,177.99443,0.04070,178.42860,0.04080,178.86275,0.04090,179.29688,0.04100,179.73099,0.04110,180.16508,0.04120,180.59914,0.04130,181.03319,0.04140,181.46721,0.04150,181.90121,0.04160,182.33520,0.04170,182.76916,0.04180,183.20310,0.04190,183.63702,0.04200,184.07091,0.04210,184.50479,0.04220,184.93865,0.04230,185.37248,0.04240,185.80630,0.04250,186.24009,0.04260,186.67386,0.04270,187.10761,0.04280,187.54134,0.04290,187.97505,0.04300,188.40874,0.04310,188.84241,0.04320,189.27605,0.04330,189.70968,0.04340,190.14328,0.04350,190.57687,0.04360,191.01043,0.04370,191.44397,0.04380,191.87749,0.04390,192.31099,0.04400,192.74447,0.04410,193.17793,0.04420,193.61136,0.04430,194.04478,0.04440,194.47817,0.04450,194.91154,0.04460,195.34490,0.04470,195.77823,0.04480,196.21154,0.04490,196.64483,0.04500,197.07809,0.04510,197.51134,0.04520,197.94457,0.04530,198.37777,0.04540,198.81096,0.04550,199.24412,0.04560,199.67726,0.04570,200.11038,0.04580,200.54348,0.04590,200.97656,0.04600,201.40962,0.04610,201.84266,0.04620,202.27567,0.04630,202.70867,0.04640,203.14164,0.04650,203.57459,0.04660,204.00753,0.04670,204.44044,0.04680,204.87333,0.04690,205.30619,0.04700,205.73904,0.04710,206.17187,0.04720,206.60467,0.04730,207.03746,0.04740,207.47022,0.04750,207.90296,0.04760,208.33569,0.04770,208.76839,0.04780,209.20106,0.04790,209.63372,0.04800,210.06636,0.04810,210.49898,0.04820,210.93157,0.04830,211.36415,0.04840,211.79670,0.04850,212.22923,0.04860,212.66174,0.04870,213.09423,0.04880,213.52670,0.04890,213.95915,0.04900,214.39157,0.04910,214.82398,0.04920,215.25636,0.04930,215.68873,0.04940,216.12107,0.04950,216.55339,0.04960,216.98569,0.04970,217.41797,0.04980,217.85023,0.04990,218.28247,0.05001,218.71468,0.05011,219.14688,0.05021,219.57905,0.05031,220.01121,0.05041,220.44334,0.05051,220.87545,0.05061,221.30754,0.05071,221.73961,0.05081,222.17165,0.05091,222.60368,0.05101,223.03569,0.05111,223.46767,0.05121,223.89963,0.05131,224.33158,0.05141,224.76350,0.05151,225.19540,0.05161,225.62728,0.05171,226.05913,0.05181,226.49097,0.05191,226.92279,0.05201,227.35458,0.05211,227.78635,0.05221,228.21811,0.05231,228.64984,0.05241,229.08155,0.05251,229.51324,0.05261,229.94491,0.05271,230.37655,0.05281,230.80818,0.05291,231.23978,0.05301,231.67137,0.05311,232.10293,0.05321,232.53447,0.05331,232.96599,0.05341,233.39749,0.05351,233.82897,0.05361,234.26043,0.05371,234.69186,0.05381,235.12328,0.05391,235.55467,0.05401,235.98604,0.05411,236.41740,0.05421,236.84873,0.05431,237.28004,0.05441,237.71133,0.05451,238.14259,0.05461,238.57384,0.05471,239.00506,0.05481,239.43627,0.05491,239.86745,0.05501,240.29861,0.05511,240.72975,0.05521,241.16087,0.05531,241.59197,0.05541,242.02305,0.05551,242.45410,0.05561,242.88514,0.05571,243.31615,0.05581,243.74715,0.05591,244.17812,0.05601,244.60907,0.05611,245.04000,0.05621,245.47091,0.05631,245.90179,0.05641,246.33266,0.05651,246.76351,0.05661,247.19433,0.05671,247.62513,0.05681,248.05591,0.05691,248.48667,0.05701,248.91741,0.05711,249.34813,0.05721,249.77883,0.05731,250.20950,0.05741,250.64016,0.05751,251.07079,0.05761,251.50141,0.05771,251.93200,0.05781,252.36257,0.05791,252.79312,0.05801,253.22365,0.05811,253.65415,0.05821,254.08464,0.05831,254.51510,0.05841,254.94555,0.05851,255.37597,0.05861,255.80637,0.05871,256.23675,0.05881,256.66711,0.05891,257.09745,0.05901,257.52776,0.05911,257.95806,0.05921,258.38833,0.05931,258.81859,0.05941,259.24882,0.05951,259.67903,0.05961,260.10922,0.05971,260.53939,0.05981,260.96954,0.05991,261.39966,0.06001,261.82977,0.06011,262.25985,0.06021,262.68991,0.06031,263.11996,0.06041,263.54998,0.06051,263.97998,0.06061,264.40995,0.06071,264.83991,0.06081,265.26985,0.06091,265.69976,0.06101,266.12966,0.06111,266.55953,0.06121,266.98938,0.06131,267.41921,0.06141,267.84902,0.06151,268.27881,0.06161,268.70857,0.06171,269.13832,0.06181,269.56804,0.06191,269.99775,0.06201,270.42743,0.06211,270.85709,0.06221,271.28673,0.06231,271.71635,0.06241,272.14594,0.06251,272.57552,0.06261,273.00507,0.06271,273.43461,0.06281,273.86412,0.06291,274.29361,0.06301,274.72308,0.06311,275.15253,0.06321,275.58196,0.06331,276.01137,0.06341,276.44075,0.06351,276.87012,0.06361,277.29946,0.06371,277.72878,0.06381,278.15808,0.06391,278.58736,0.06401,279.01662,0.06411,279.44586,0.06421,279.87507,0.06431,280.30427,0.06441,280.73344,0.06451,281.16259,0.06461,281.59172,0.06471,282.02083,0.06481,282.44992,0.06491,282.87899,0.06501,283.30804,0.06511,283.73706,0.06521,284.16607,0.06531,284.59505,0.06541,285.02401,0.06551,285.45295,0.06561,285.88187,0.06571,286.31077,0.06581,286.73965,0.06591,287.16850,0.06601,287.59734,0.06611,288.02615,0.06621,288.45494,0.06631,288.88371,0.06641,289.31246,0.06651,289.74119,0.06661,290.16990,0.06671,290.59858,0.06681,291.02725,0.06691,291.45589,0.06701,291.88451,0.06711,292.31311,0.06721,292.74169,0.06731,293.17025,0.06741,293.59879,0.06751,294.02731,0.06761,294.45580,0.06771,294.88428,0.06781,295.31273,0.06791,295.74116,0.06801,296.16957,0.06811,296.59796,0.06821,297.02633,0.06831,297.45467,0.06841,297.88300,0.06851,298.31130,0.06861,298.73958,0.06871,299.16785,0.06881,299.59609,0.06891,300.02431,0.06901,300.45250,0.06911,300.88068,0.06921,301.30884,0.06931,301.73697,0.06941,302.16508,0.06951,302.59317,0.06961,303.02124,0.06971,303.44929,0.06981,303.87732,0.06991,304.30533,0.07001,304.73331,0.07011,305.16128,0.07021,305.58922,0.07031,306.01714,0.07041,306.44504,0.07051,306.87292,0.07061,307.30078,0.07071,307.72862,0.07081,308.15643,0.07091,308.58423,0.07101,309.01200,0.07111,309.43975,0.07121,309.86748,0.07131,310.29519,0.07141,310.72288,0.07151,311.15055,0.07161,311.57819,0.07171,312.00582,0.07181,312.43342,0.07191,312.86100,0.07201,313.28856,0.07211,313.71610,0.07221,314.14362,0.07231,314.57112,0.07241,314.99859,0.07251,315.42605,0.07261,315.85348,0.07271,316.28089,0.07281,316.70828,0.07291,317.13565,0.07301,317.56300,0.07311,317.99033,0.07321,318.41763,0.07331,318.84492,0.07341,319.27218,0.07351,319.69942,0.07361,320.12664,0.07371,320.55384,0.07381,320.98102,0.07391,321.40818,0.07401,321.83531,0.07411,322.26243,0.07421,322.68952,0.07431,323.11659,0.07441,323.54364,0.07451,323.97067,0.07461,324.39768,0.07471,324.82467,0.07481,325.25163,0.07491,325.67857,0.07501,326.10550,0.07511,326.53240,0.07521,326.95928,0.07531,327.38614,0.07541,327.81298,0.07551,328.23979,0.07561,328.66659,0.07571,329.09336,0.07581,329.52011,0.07591,329.94684,0.07601,330.37355,0.07611,330.80024,0.07621,331.22691,0.07631,331.65356,0.07641,332.08018,0.07651,332.50678,0.07661,332.93337,0.07671,333.35993,0.07681,333.78647,0.07691,334.21298,0.07701,334.63948,0.07711,335.06596,0.07721,335.49241,0.07731,335.91884,0.07741,336.34526,0.07751,336.77165,0.07761,337.19802,0.07771,337.62436,0.07781,338.05069,0.07791,338.47700,0.07801,338.90328,0.07811,339.32954,0.07821,339.75578,0.07831,340.18200,0.07841,340.60820,0.07851,341.03438,0.07861,341.46054,0.07871,341.88667,0.07881,342.31278,0.07891,342.73888,0.07901,343.16495,0.07911,343.59100,0.07921,344.01703,0.07931,344.44303,0.07941,344.86902,0.07951,345.29498,0.07961,345.72093,0.07971,346.14685,0.07981,346.57275,0.07991,346.99863,0.08001,347.42449,0.08011,347.85032,0.08021,348.27614,0.08031,348.70193,0.08041,349.12770,0.08051,349.55345,0.08061,349.97918,0.08071,350.40489,0.08081,350.83058,0.08091,351.25625,0.08101,351.68189,0.08111,352.10751,0.08121,352.53312,0.08131,352.95870,0.08141,353.38426,0.08151,353.80979,0.08161,354.23531,0.08171,354.66081,0.08181,355.08628,0.08191,355.51173,0.08201,355.93716,0.08211,356.36257,0.08221,356.78796,0.08231,357.21333,0.08241,357.63868,0.08251,358.06400,0.08261,358.48931,0.08271,358.91459,0.08281,359.33985,0.08291,359.76509,0.08301,360.19031,0.08311,360.61550,0.08321,361.04068,0.08331,361.46583,0.08341,361.89096,0.08351,362.31608,0.08361,362.74117,0.08371,363.16623,0.08381,363.59128,0.08391,364.01631,0.08401,364.44131,0.08411,364.86630,0.08421,365.29126,0.08431,365.71620,0.08441,366.14112,0.08451,366.56601,0.08461,366.99089,0.08471,367.41575,0.08481,367.84058,0.08491,368.26539,0.08501,368.69018,0.08511,369.11495,0.08521,369.53970,0.08531,369.96443,0.08541,370.38913,0.08551,370.81382,0.08561,371.23848,0.08571,371.66312,0.08581,372.08774,0.08591,372.51234,0.08601,372.93692,0.08611,373.36148,0.08621,373.78601,0.08631,374.21053,0.08641,374.63502,0.08651,375.05949,0.08661,375.48394,0.08671,375.90837,0.08681,376.33277,0.08691,376.75716,0.08701,377.18152,0.08711,377.60586,0.08721,378.03019,0.08731,378.45449,0.08741,378.87876,0.08751,379.30302,0.08761,379.72726,0.08771,380.15147,0.08781,380.57566,0.08791,380.99984,0.08801,381.42399,0.08811,381.84811,0.08821,382.27222,0.08831,382.69631,0.08841,383.12037,0.08851,383.54442,0.08861,383.96844,0.08871,384.39244,0.08881,384.81642,0.08891,385.24038,0.08901,385.66431,0.08911,386.08823,0.08921,386.51212,0.08931,386.93599,0.08941,387.35984,0.08951,387.78367,0.08961,388.20748,0.08971,388.63127,0.08981,389.05503,0.08991,389.47878,0.09001,389.90250,0.09011,390.32620,0.09021,390.74988,0.09031,391.17354,0.09041,391.59718,0.09051,392.02079,0.09061,392.44439,0.09071,392.86796,0.09081,393.29151,0.09091,393.71504,0.09101,394.13855,0.09111,394.56204,0.09121,394.98550,0.09131,395.40895,0.09141,395.83237,0.09151,396.25577,0.09161,396.67915,0.09171,397.10251,0.09181,397.52585,0.09191,397.94917,0.09201,398.37246,0.09211,398.79574,0.09221,399.21899,0.09231,399.64222,0.09241,400.06543,0.09251,400.48861,0.09261,400.91178,0.09271,401.33493,0.09281,401.75805,0.09291,402.18115,0.09301,402.60423,0.09311,403.02729,0.09321,403.45033,0.09331,403.87335,0.09341,404.29634,0.09351,404.71931,0.09361,405.14227,0.09371,405.56520,0.09381,405.98811,0.09391,406.41100,0.09401,406.83386,0.09411,407.25671,0.09421,407.67953,0.09431,408.10233,0.09441,408.52511,0.09451,408.94787,0.09461,409.37061,0.09471,409.79333,0.09481,410.21602,0.09491,410.63870,0.09501,411.06135,0.09511,411.48398,0.09521,411.90659,0.09531,412.32918,0.09541,412.75175,0.09551,413.17429,0.09561,413.59681,0.09571,414.01932,0.09581,414.44180,0.09591,414.86426,0.09601,415.28670,0.09611,415.70911,0.09621,416.13151,0.09631,416.55388,0.09641,416.97623,0.09651,417.39857,0.09661,417.82087,0.09671,418.24316,0.09681,418.66543,0.09691,419.08768,0.09701,419.50990,0.09711,419.93210,0.09721,420.35428,0.09731,420.77644,0.09741,421.19858,0.09751,421.62070,0.09761,422.04279,0.09771,422.46487,0.09781,422.88692,0.09791,423.30895,0.09801,423.73096,0.09811,424.15295,0.09821,424.57491,0.09831,424.99686,0.09841,425.41878,0.09851,425.84068,0.09861,426.26257,0.09871,426.68442,0.09881,427.10626,0.09891,427.52808,0.09901,427.94987,0.09911,428.37165,0.09921,428.79340,0.09931,429.21513,0.09941,429.63684,0.09951,430.05853,0.09961,430.48019,0.09971,430.90184,0.09981,431.32346,0.09991,431.74506,0.10001,432.16665,0.10011,432.58820,0.10021,433.00974,0.10031,433.43126,0.10041,433.85275,0.10051,434.27423,0.10061,434.69568,0.10071,435.11711,0.10081,435.53852,0.10091,435.95991,0.10101,436.38127,0.10111,436.80262,0.10121,437.22394,0.10131,437.64524,0.10141,438.06652,0.10151,438.48778,0.10161,438.90902,0.10171,439.33023,0.10181,439.75143,0.10191,440.17260,0.10201,440.59375,0.10211,441.01488,0.10221,441.43599,0.10231,441.85708,0.10241,442.27814,0.10251,442.69919,0.10261,443.12021,0.10271,443.54121,0.10281,443.96219,0.10291,444.38315,0.10301,444.80408,0.10311,445.22500,0.10321,445.64589,0.10331,446.06677,0.10341,446.48762,0.10351,446.90845,0.10361,447.32925,0.10371,447.75004,0.10381,448.17080,0.10391,448.59155,0.10401,449.01227,0.10411,449.43297,0.10421,449.85365,0.10431,450.27431,0.10441,450.69494,0.10451,451.11556,0.10461,451.53615,0.10471,451.95672,0.10481,452.37727,0.10491,452.79780,0.10501,453.21831,0.10511,453.63879,0.10521,454.05926,0.10531,454.47970,0.10541,454.90012,0.10551,455.32052,0.10561,455.74090,0.10571,456.16126,0.10581,456.58159,0.10591,457.00191,0.10601,457.42220,0.10611,457.84247,0.10621,458.26272,0.10631,458.68295,0.10641,459.10315,0.10651,459.52334,0.10661,459.94350,0.10671,460.36364,0.10681,460.78376,0.10691,461.20386,0.10701,461.62394,0.10711,462.04399,0.10721,462.46403,0.10731,462.88404,0.10741,463.30403,0.10751,463.72400,0.10761,464.14395,0.10771,464.56388,0.10781,464.98378,0.10791,465.40367,0.10801,465.82353,0.10811,466.24337,0.10821,466.66319,0.10831,467.08299,0.10841,467.50276,0.10851,467.92252,0.10861,468.34225,0.10871,468.76196,0.10881,469.18165,0.10891,469.60132,0.10901,470.02097,0.10911,470.44060,0.10921,470.86020,0.10931,471.27978,0.10941,471.69934,0.10951,472.11888,0.10961,472.53840,0.10971,472.95790,0.10981,473.37737,0.10991,473.79683,0.11001,474.21626,0.11011,474.63567,0.11021,475.05506,0.11031,475.47443,0.11041,475.89377,0.11051,476.31310,0.11061,476.73240,0.11071,477.15168,0.11081,477.57094,0.11091,477.99018,0.11101,478.40940,0.11111,478.82860,0.11121,479.24777,0.11131,479.66692,0.11141,480.08605,0.11151,480.50516,0.11161,480.92425,0.11171,481.34332,0.11181,481.76236,0.11191,482.18139,0.11201,482.60039,0.11211,483.01937,0.11221,483.43833,0.11231,483.85726,0.11241,484.27618,0.11251,484.69507,0.11261,485.11395,0.11271,485.53280,0.11281,485.95163,0.11291,486.37044,0.11301,486.78922,0.11311,487.20799,0.11321,487.62673,0.11331,488.04545,0.11341,488.46415,0.11351,488.88283,0.11361,489.30149,0.11371,489.72013,0.11381,490.13874,0.11391,490.55733,0.11401,490.97590,0.11411,491.39445,0.11421,491.81298,0.11431,492.23149,0.11441,492.64997,0.11451,493.06844,0.11461,493.48688,0.11471,493.90530,0.11481,494.32370,0.11491,494.74208,0.11501,495.16043,0.11511,495.57877,0.11521,495.99708,0.11531,496.41537,0.11541,496.83364,0.11551,497.25189,0.11561,497.67011,0.11571,498.08832,0.11581,498.50650,0.11591,498.92467,0.11601,499.34281,0.11611,499.76092,0.11621,500.17902,0.11631,500.59710,0.11641,501.01515,0.11651,501.43319,0.11661,501.85120,0.11671,502.26919,0.11681,502.68715,0.11691,503.10510,0.11701,503.52303,0.11711,503.94093,0.11721,504.35881,0.11731,504.77667,0.11741,505.19451,0.11751,505.61233,0.11761,506.03012,0.11771,506.44790,0.11781,506.86565,0.11791,507.28338,0.11801,507.70109,0.11811,508.11878,0.11821,508.53644,0.11831,508.95409,0.11841,509.37171,0.11851,509.78931,0.11861,510.20689,0.11871,510.62445,0.11881,511.04199,0.11891,511.45950,0.11901,511.87700,0.11911,512.29447,0.11921,512.71192,0.11931,513.12935,0.11941,513.54676,0.11951,513.96414,0.11961,514.38151,0.11971,514.79885,0.11981,515.21617,0.11991,515.63347,0.12001,516.05075,0.12011,516.46801,0.12021,516.88524,0.12031,517.30246,0.12041,517.71965,0.12051,518.13682,0.12061,518.55397,0.12071,518.97109,0.12081,519.38820,0.12091,519.80528,0.12101,520.22235,0.12111,520.63939,0.12121,521.05641,0.12131,521.47341,0.12141,521.89038,0.12151,522.30734,0.12161,522.72427,0.12171,523.14118,0.12181,523.55807,0.12191,523.97494,0.12201,524.39179,0.12211,524.80861,0.12221,525.22541,0.12231,525.64220,0.12241,526.05896,0.12251,526.47570,0.12261,526.89241,0.12271,527.30911,0.12281,527.72578,0.12291,528.14244,0.12301,528.55907,0.12311,528.97568,0.12321,529.39226,0.12331,529.80883,0.12341,530.22538,0.12351,530.64190,0.12361,531.05840,0.12371,531.47488,0.12381,531.89134,0.12391,532.30777,0.12401,532.72419,0.12411,533.14058,0.12421,533.55696,0.12431,533.97331,0.12441,534.38963,0.12451,534.80594,0.12461,535.22223,0.12471,535.63849,0.12481,536.05473,0.12491,536.47095,0.12501,536.88715,0.12511,537.30333,0.12521,537.71949,0.12531,538.13562,0.12541,538.55174,0.12551,538.96783,0.12561,539.38390,0.12571,539.79994,0.12581,540.21597,0.12591,540.63198,0.12601,541.04796,0.12611,541.46392,0.12621,541.87986,0.12631,542.29578,0.12641,542.71168,0.12651,543.12755,0.12661,543.54341,0.12671,543.95924,0.12681,544.37505,0.12691,544.79084,0.12701,545.20660,0.12711,545.62235,0.12721,546.03807,0.12731,546.45378,0.12741,546.86946,0.12751,547.28512,0.12761,547.70076,0.12771,548.11637,0.12781,548.53197,0.12791,548.94754,0.12801,549.36309,0.12811,549.77862,0.12821,550.19413,0.12831,550.60961,0.12841,551.02508,0.12851,551.44052,0.12861,551.85594,0.12871,552.27134,0.12881,552.68672,0.12891,553.10208,0.12901,553.51741,0.12911,553.93273,0.12921,554.34802,0.12931,554.76329,0.12941,555.17854,0.12951,555.59377,0.12961,556.00897,0.12971,556.42416,0.12981,556.83932,0.12991,557.25446,0.13001,557.66958,0.13011,558.08468,0.13021,558.49975,0.13031,558.91481,0.13041,559.32984,0.13051,559.74485,0.13061,560.15984,0.13071,560.57481,0.13081,560.98975,0.13091,561.40468,0.13101,561.81958,0.13111,562.23446,0.13121,562.64932,0.13131,563.06416,0.13141,563.47898,0.13151,563.89377,0.13161,564.30854,0.13171,564.72330,0.13181,565.13803,0.13191,565.55273,0.13201,565.96742,0.13211,566.38209,0.13221,566.79673,0.13231,567.21135,0.13241,567.62595,0.13251,568.04053,0.13261,568.45509,0.13271,568.86962,0.13281,569.28414,0.13291,569.69863,0.13301,570.11310,0.13311,570.52755,0.13321,570.94198,0.13331,571.35638,0.13341,571.77077,0.13351,572.18513,0.13361,572.59947,0.13371,573.01379,0.13381,573.42808,0.13391,573.84236,0.13401,574.25661,0.13411,574.67085,0.13421,575.08506,0.13431,575.49925,0.13441,575.91342,0.13451,576.32756,0.13461,576.74169,0.13471,577.15579,0.13481,577.56987,0.13491,577.98393,0.13501,578.39797,0.13511,578.81198,0.13521,579.22598,0.13531,579.63995,0.13541,580.05390,0.13551,580.46783,0.13561,580.88174,0.13571,581.29563,0.13581,581.70949,0.13591,582.12333,0.13601,582.53716,0.13611,582.95096,0.13621,583.36473,0.13631,583.77849,0.13641,584.19222,0.13651,584.60594,0.13661,585.01963,0.13671,585.43330,0.13681,585.84695,0.13691,586.26057,0.13701,586.67418,0.13711,587.08776,0.13721,587.50132,0.13731,587.91486,0.13741,588.32838,0.13751,588.74188,0.13761,589.15536,0.13771,589.56881,0.13781,589.98224,0.13791,590.39565,0.13801,590.80904,0.13811,591.22241,0.13821,591.63575,0.13831,592.04908,0.13841,592.46238,0.13851,592.87566,0.13861,593.28892,0.13871,593.70215,0.13881,594.11537,0.13891,594.52856,0.13901,594.94173,0.13911,595.35488,0.13921,595.76801,0.13931,596.18112,0.13941,596.59421,0.13951,597.00727,0.13961,597.42031,0.13971,597.83333,0.13981,598.24633,0.13991,598.65931,0.14001,599.07226,0.14011,599.48520,0.14021,599.89811,0.14031,600.31100,0.14041,600.72387,0.14051,601.13672,0.14061,601.54954,0.14071,601.96235,0.14081,602.37513,0.14091,602.78789,0.14101,603.20063,0.14111,603.61335,0.14121,604.02604,0.14131,604.43871,0.14141,604.85137,0.14151,605.26400,0.14161,605.67661,0.14171,606.08919,0.14181,606.50176,0.14191,606.91430,0.14201,607.32683,0.14211,607.73933,0.14221,608.15181,0.14231,608.56426,0.14241,608.97670,0.14251,609.38911,0.14261,609.80150,0.14271,610.21388,0.14281,610.62622,0.14291,611.03855,0.14301,611.45086,0.14311,611.86314,0.14321,612.27540,0.14331,612.68764,0.14341,613.09986,0.14351,613.51206,0.14361,613.92424,0.14371,614.33639,0.14381,614.74852,0.14391,615.16063,0.14401,615.57272,0.14411,615.98479,0.14421,616.39683,0.14431,616.80886,0.14441,617.22086,0.14451,617.63284,0.14461,618.04480,0.14471,618.45674,0.14481,618.86865,0.14491,619.28055,0.14501,619.69242,0.14511,620.10427,0.14521,620.51610,0.14531,620.92791,0.14541,621.33969,0.14551,621.75146,0.14561,622.16320,0.14571,622.57492,0.14581,622.98662,0.14591,623.39829,0.14601,623.80995,0.14611,624.22158,0.14621,624.63320,0.14631,625.04479,0.14641,625.45635,0.14651,625.86790,0.14661,626.27943,0.14671,626.69093,0.14681,627.10241,0.14691,627.51387,0.14701,627.92531,0.14711,628.33673,0.14721,628.74812,0.14731,629.15950,0.14741,629.57085,0.14751,629.98218,0.14761,630.39349,0.14771,630.80477,0.14781,631.21604,0.14791,631.62728,0.14801,632.03850,0.14811,632.44970,0.14821,632.86088,0.14831,633.27204,0.14841,633.68318,0.14851,634.09429,0.14861,634.50538,0.14871,634.91645,0.14881,635.32750,0.14891,635.73852,0.14901,636.14953,0.14911,636.56051,0.14921,636.97147,0.14931,637.38241,0.14941,637.79333,0.14951,638.20423,0.14961,638.61510,0.14971,639.02596,0.14981,639.43679,0.14991,639.84760,0.15002,640.25839,0.15012,640.66915,0.15022,641.07990,0.15032,641.49062,0.15042,641.90132,0.15052,642.31200,0.15062,642.72266,0.15072,643.13329,0.15082,643.54391,0.15092,643.95450,0.15102,644.36507,0.15112,644.77562,0.15122,645.18615,0.15132,645.59666,0.15142,646.00714,0.15152,646.41760,0.15162,646.82804,0.15172,647.23846,0.15182,647.64886,0.15192,648.05924,0.15202,648.46959,0.15212,648.87992,0.15222,649.29023,0.15232,649.70052,0.15242,650.11079,0.15252,650.52103,0.15262,650.93126,0.15272,651.34146,0.15282,651.75164,0.15292,652.16180,0.15302,652.57194,0.15312,652.98205,0.15322,653.39215,0.15332,653.80222,0.15342,654.21227,0.15352,654.62230,0.15362,655.03230,0.15372,655.44229,0.15382,655.85225,0.15392,656.26219,0.15402,656.67211,0.15412,657.08201,0.15422,657.49189,0.15432,657.90174,0.15442,658.31158,0.15452,658.72139,0.15462,659.13118,0.15472,659.54094,0.15482,659.95069,0.15492,660.36042,0.15502,660.77012,0.15512,661.17980,0.15522,661.58946,0.15532,661.99910,0.15542,662.40871,0.15552,662.81831,0.15562,663.22788,0.15572,663.63743,0.15582,664.04696,0.15592,664.45647,0.15602,664.86595,0.15612,665.27542,0.15622,665.68486,0.15632,666.09428,0.15642,666.50368,0.15652,666.91306,0.15662,667.32241,0.15672,667.73175,0.15682,668.14106,0.15692,668.55035,0.15702,668.95962,0.15712,669.36886,0.15722,669.77809,0.15732,670.18729,0.15742,670.59648,0.15752,671.00564,0.15762,671.41477,0.15772,671.82389,0.15782,672.23299,0.15792,672.64206,0.15802,673.05111,0.15812,673.46014,0.15822,673.86915,0.15832,674.27813,0.15842,674.68710,0.15852,675.09604,0.15862,675.50496,0.15872,675.91386,0.15882,676.32274,0.15892,676.73160,0.15902,677.14043,0.15912,677.54924,0.15922,677.95803,0.15932,678.36680,0.15942,678.77555,0.15952,679.18428,0.15962,679.59298,0.15972,680.00166,0.15982,680.41032,0.15992,680.81896,0.16002,681.22758,0.16012,681.63617,0.16022,682.04475,0.16032,682.45330,0.16042,682.86183,0.16052,683.27034,0.16062,683.67882,0.16072,684.08729,0.16082,684.49573,0.16092,684.90415,0.16102,685.31255,0.16112,685.72093,0.16122,686.12929,0.16132,686.53762,0.16142,686.94594,0.16152,687.35423,0.16162,687.76250,0.16172,688.17074,0.16182,688.57897,0.16192,688.98717,0.16202,689.39536,0.16212,689.80352,0.16222,690.21166,0.16232,690.61977,0.16242,691.02787,0.16252,691.43594,0.16262,691.84400,0.16272,692.25203,0.16282,692.66004,0.16292,693.06802,0.16302,693.47599,0.16312,693.88393,0.16322,694.29185,0.16332,694.69975,0.16342,695.10763,0.16352,695.51549,0.16362,695.92332,0.16372,696.33114,0.16382,696.73893,0.16392,697.14670,0.16402,697.55444,0.16412,697.96217,0.16422,698.36988,0.16432,698.77756,0.16442,699.18522,0.16452,699.59286,0.16462,700.00048,0.16472,700.40807,0.16482,700.81565,0.16492,701.22320,0.16502,701.63073,0.16512,702.03824,0.16522,702.44572,0.16532,702.85319,0.16542,703.26063,0.16552,703.66805,0.16562,704.07545,0.16572,704.48283,0.16582,704.89019,0.16592,705.29752,0.16602,705.70484,0.16612,706.11213,0.16622,706.51940,0.16632,706.92665,0.16642,707.33387,0.16652,707.74108,0.16662,708.14826,0.16672,708.55542,0.16682,708.96256,0.16692,709.36968,0.16702,709.77677,0.16712,710.18385,0.16722,710.59090,0.16732,710.99793,0.16742,711.40494,0.16752,711.81193,0.16762,712.21889,0.16772,712.62583,0.16782,713.03276,0.16792,713.43966,0.16802,713.84653,0.16812,714.25339,0.16822,714.66023,0.16832,715.06704,0.16842,715.47383,0.16852,715.88060,0.16862,716.28735,0.16872,716.69407,0.16882,717.10078,0.16892,717.50746,0.16902,717.91412,0.16912,718.32076,0.16922,718.72738,0.16932,719.13397,0.16942,719.54055,0.16952,719.94710,0.16962,720.35363,0.16972,720.76014,0.16982,721.16662,0.16992,721.57309,0.17002,721.97953,0.17012,722.38595,0.17022,722.79235,0.17032,723.19873,0.17042,723.60509,0.17052,724.01142,0.17062,724.41774,0.17072,724.82403,0.17082,725.23030,0.17092,725.63654,0.17102,726.04277,0.17112,726.44897,0.17122,726.85516,0.17132,727.26132,0.17142,727.66746,0.17152,728.07357,0.17162,728.47967,0.17172,728.88574,0.17182,729.29179,0.17192,729.69782,0.17202,730.10383,0.17212,730.50982,0.17222,730.91578,0.17232,731.32173,0.17242,731.72765,0.17252,732.13355,0.17262,732.53943,0.17272,732.94528,0.17282,733.35112,0.17292,733.75693,0.17302,734.16272,0.17312,734.56849,0.17322,734.97424,0.17332,735.37996,0.17342,735.78566,0.17352,736.19135,0.17362,736.59701,0.17372,737.00264,0.17382,737.40826,0.17392,737.81386,0.17402,738.21943,0.17412,738.62498,0.17422,739.03051,0.17432,739.43602,0.17442,739.84150,0.17452,740.24697,0.17462,740.65241,0.17472,741.05783,0.17482,741.46323,0.17492,741.86861,0.17502,742.27396,0.17512,742.67930,0.17522,743.08461,0.17532,743.48990,0.17542,743.89517,0.17552,744.30042,0.17562,744.70564,0.17572,745.11084,0.17582,745.51603,0.17592,745.92119,0.17602,746.32632,0.17612,746.73144,0.17622,747.13653,0.17632,747.54161,0.17642,747.94666,0.17652,748.35169,0.17662,748.75669,0.17672,749.16168,0.17682,749.56664,0.17692,749.97159,0.17702,750.37651,0.17712,750.78141,0.17722,751.18628,0.17732,751.59114,0.17742,751.99597,0.17752,752.40078,0.17762,752.80557,0.17772,753.21034,0.17782,753.61509,0.17792,754.01981,0.17802,754.42451,0.17812,754.82919,0.17822,755.23385,0.17832,755.63849,0.17842,756.04311,0.17852,756.44770,0.17862,756.85227,0.17872,757.25682,0.17882,757.66135,0.17892,758.06586,0.17902,758.47034,0.17912,758.87481,0.17922,759.27925,0.17932,759.68367,0.17942,760.08807,0.17952,760.49244,0.17962,760.89680,0.17972,761.30113,0.17982,761.70544,0.17992,762.10973,0.18002,762.51400,0.18012,762.91824,0.18022,763.32247,0.18032,763.72667,0.18042,764.13085,0.18052,764.53501,0.18062,764.93914,0.18072,765.34326,0.18082,765.74735,0.18092,766.15142,0.18102,766.55547,0.18112,766.95950,0.18122,767.36351,0.18132,767.76749,0.18142,768.17145,0.18152,768.57539,0.18162,768.97931,0.18172,769.38321,0.18182,769.78708,0.18192,770.19094,0.18202,770.59477,0.18212,770.99858,0.18222,771.40237,0.18232,771.80614,0.18242,772.20988,0.18252,772.61360,0.18262,773.01730,0.18272,773.42098,0.18282,773.82464,0.18292,774.22828,0.18302,774.63189,0.18312,775.03548,0.18322,775.43905,0.18332,775.84260,0.18342,776.24613,0.18352,776.64964,0.18362,777.05312,0.18372,777.45658,0.18382,777.86002,0.18392,778.26344,0.18402,778.66684,0.18412,779.07021,0.18422,779.47356,0.18432,779.87689,0.18442,780.28020,0.18452,780.68349,0.18462,781.08676,0.18472,781.49000,0.18482,781.89322,0.18492,782.29642,0.18502,782.69960,0.18512,783.10276,0.18522,783.50589,0.18532,783.90901,0.18542,784.31210,0.18552,784.71517,0.18562,785.11822,0.18572,785.52124,0.18582,785.92425,0.18592,786.32723,0.18602,786.73019,0.18612,787.13313,0.18622,787.53605,0.18632,787.93894,0.18642,788.34182,0.18652,788.74467,0.18662,789.14750,0.18672,789.55031,0.18682,789.95309,0.18692,790.35586,0.18702,790.75860,0.18712,791.16132,0.18722,791.56402,0.18732,791.96670,0.18742,792.36936,0.18752,792.77199,0.18762,793.17460,0.18772,793.57719,0.18782,793.97976,0.18792,794.38231,0.18802,794.78483,0.18812,795.18734,0.18822,795.58982,0.18832,795.99228,0.18842,796.39472,0.18852,796.79713,0.18862,797.19953,0.18872,797.60190,0.18882,798.00425,0.18892,798.40658,0.18902,798.80889,0.18912,799.21117,0.18922,799.61344,0.18932,800.01568,0.18942,800.41790,0.18952,800.82010,0.18962,801.22228,0.18972,801.62443,0.18982,802.02656,0.18992,802.42868,0.19002,802.83077,0.19012,803.23283,0.19022,803.63488,0.19032,804.03690,0.19042,804.43891,0.19052,804.84089,0.19062,805.24285,0.19072,805.64478,0.19082,806.04670,0.19092,806.44859,0.19102,806.85046,0.19112,807.25231,0.19122,807.65414,0.19132,808.05595,0.19142,808.45773,0.19152,808.85949,0.19162,809.26124,0.19172,809.66295,0.19182,810.06465,0.19192,810.46633,0.19202,810.86798,0.19212,811.26961,0.19222,811.67122,0.19232,812.07281,0.19242,812.47438,0.19252,812.87592,0.19262,813.27745,0.19272,813.67895,0.19282,814.08043,0.19292,814.48189,0.19302,814.88332,0.19312,815.28474,0.19322,815.68613,0.19332,816.08750,0.19342,816.48885,0.19352,816.89018,0.19362,817.29148,0.19372,817.69276,0.19382,818.09403,0.19392,818.49527,0.19402,818.89648,0.19412,819.29768,0.19422,819.69885,0.19432,820.10001,0.19442,820.50114,0.19452,820.90225,0.19462,821.30333,0.19472,821.70440,0.19482,822.10544,0.19492,822.50647,0.19502,822.90747,0.19512,823.30845,0.19522,823.70940,0.19532,824.11034,0.19542,824.51125,0.19552,824.91214,0.19562,825.31301,0.19572,825.71386,0.19582,826.11468,0.19592,826.51549,0.19602,826.91627,0.19612,827.31703,0.19622,827.71777,0.19632,828.11849,0.19642,828.51918,0.19652,828.91986,0.19662,829.32051,0.19672,829.72114,0.19682,830.12174,0.19692,830.52233,0.19702,830.92290,0.19712,831.32344,0.19722,831.72396,0.19732,832.12446,0.19742,832.52493,0.19752,832.92539,0.19762,833.32582,0.19772,833.72624,0.19782,834.12663,0.19792,834.52699,0.19802,834.92734,0.19812,835.32767,0.19822,835.72797,0.19832,836.12825,0.19842,836.52851,0.19852,836.92875,0.19862,837.32896,0.19872,837.72916,0.19882,838.12933,0.19892,838.52948,0.19902,838.92961,0.19912,839.32971,0.19922,839.72980,0.19932,840.12986,0.19942,840.52990,0.19952,840.92992,0.19962,841.32992,0.19972,841.72990,0.19982,842.12985,0.19992,842.52978,0.20002,842.92969,0.20012,843.32958,0.20022,843.72945,0.20032,844.12929,0.20042,844.52912,0.20052,844.92892,0.20062,845.32870,0.20072,845.72846,0.20082,846.12819,0.20092,846.52791,0.20102,846.92760,0.20112,847.32727,0.20122,847.72692,0.20132,848.12655,0.20142,848.52615,0.20152,848.92574,0.20162,849.32530,0.20172,849.72484,0.20182,850.12436,0.20192,850.52385,0.20202,850.92333,0.20212,851.32278,0.20222,851.72221,0.20232,852.12162,0.20242,852.52101,0.20252,852.92038,0.20262,853.31972,0.20272,853.71904,0.20282,854.11834,0.20292,854.51762,0.20302,854.91688,0.20312,855.31611,0.20322,855.71533,0.20332,856.11452,0.20342,856.51369,0.20352,856.91283,0.20362,857.31196,0.20372,857.71106,0.20382,858.11015,0.20392,858.50921,0.20402,858.90825,0.20412,859.30726,0.20422,859.70626,0.20432,860.10523,0.20442,860.50418,0.20452,860.90311,0.20462,861.30202,0.20472,861.70091,0.20482,862.09977,0.20492,862.49861,0.20502,862.89744,0.20512,863.29623,0.20522,863.69501,0.20532,864.09377,0.20542,864.49250,0.20552,864.89121,0.20562,865.28990,0.20572,865.68857,0.20582,866.08722,0.20592,866.48584,0.20602,866.88445,0.20612,867.28303,0.20622,867.68159,0.20632,868.08012,0.20642,868.47864,0.20652,868.87713,0.20662,869.27560,0.20672,869.67406,0.20682,870.07248,0.20692,870.47089,0.20702,870.86928,0.20712,871.26764,0.20722,871.66598,0.20732,872.06430,0.20742,872.46260,0.20752,872.86087,0.20762,873.25913,0.20772,873.65736,0.20782,874.05557,0.20792,874.45376,0.20802,874.85193,0.20812,875.25007,0.20822,875.64819,0.20832,876.04630,0.20842,876.44437,0.20852,876.84243,0.20862,877.24047,0.20872,877.63848,0.20882,878.03648,0.20892,878.43445,0.20902,878.83240,0.20912,879.23032,0.20922,879.62823,0.20932,880.02611,0.20942,880.42397,0.20952,880.82181,0.20962,881.21963,0.20972,881.61743,0.20982,882.01520,0.20992,882.41295,0.21002,882.81069,0.21012,883.20839,0.21022,883.60608,0.21032,884.00375,0.21042,884.40139,0.21052,884.79901,0.21062,885.19661,0.21072,885.59419,0.21082,885.99175,0.21092,886.38928,0.21102,886.78680,0.21112,887.18429,0.21122,887.58176,0.21132,887.97920,0.21142,888.37663,0.21152,888.77403,0.21162,889.17142,0.21172,889.56878,0.21182,889.96612,0.21192,890.36343,0.21202,890.76073,0.21212,891.15800,0.21222,891.55525,0.21232,891.95248,0.21242,892.34969,0.21252,892.74687,0.21262,893.14404,0.21272,893.54118,0.21282,893.93830,0.21292,894.33540,0.21302,894.73248,0.21312,895.12953,0.21322,895.52657,0.21332,895.92358,0.21342,896.32057,0.21352,896.71754,0.21362,897.11448,0.21372,897.51141,0.21382,897.90831,0.21392,898.30519,0.21402,898.70205,0.21412,899.09889,0.21422,899.49570,0.21432,899.89249,0.21442,900.28927,0.21452,900.68602,0.21462,901.08274,0.21472,901.47945,0.21482,901.87614,0.21492,902.27280,0.21502,902.66944,0.21512,903.06606,0.21522,903.46266,0.21532,903.85923,0.21542,904.25579,0.21552,904.65232,0.21562,905.04883,0.21572,905.44532,0.21582,905.84178,0.21592,906.23823,0.21602,906.63465,0.21612,907.03105,0.21622,907.42743,0.21632,907.82379,0.21642,908.22012,0.21652,908.61644,0.21662,909.01273,0.21672,909.40900,0.21682,909.80525,0.21692,910.20148,0.21702,910.59768,0.21712,910.99387,0.21722,911.39003,0.21732,911.78617,0.21742,912.18228,0.21752,912.57838,0.21762,912.97445,0.21772,913.37051,0.21782,913.76654,0.21792,914.16255,0.21802,914.55853,0.21812,914.95450,0.21822,915.35044,0.21832,915.74636,0.21842,916.14226,0.21852,916.53814,0.21862,916.93400,0.21872,917.32983,0.21882,917.72564,0.21892,918.12144,0.21902,918.51720,0.21912,918.91295,0.21922,919.30868,0.21932,919.70438,0.21942,920.10006,0.21952,920.49572,0.21962,920.89136,0.21972,921.28698,0.21982,921.68257,0.21992,922.07815,0.22002,922.47370,0.22012,922.86923,0.22022,923.26473,0.22032,923.66022,0.22042,924.05568,0.22052,924.45112,0.22062,924.84655,0.22072,925.24194,0.22082,925.63732,0.22092,926.03268,0.22102,926.42801,0.22112,926.82332,0.22122,927.21861,0.22132,927.61388,0.22142,928.00912,0.22152,928.40435,0.22162,928.79955,0.22172,929.19473,0.22182,929.58989,0.22192,929.98503,0.22202,930.38014,0.22212,930.77523,0.22222,931.17031,0.22232,931.56536,0.22242,931.96038,0.22252,932.35539,0.22262,932.75037,0.22272,933.14534,0.22282,933.54028,0.22292,933.93520,0.22302,934.33009,0.22312,934.72497,0.22322,935.11982,0.22332,935.51465,0.22342,935.90946,0.22352,936.30425,0.22362,936.69902,0.22372,937.09376,0.22382,937.48849,0.22392,937.88319,0.22402,938.27787,0.22412,938.67252,0.22422,939.06716,0.22432,939.46177,0.22442,939.85636,0.22452,940.25093,0.22462,940.64548,0.22472,941.04001,0.22482,941.43451,0.22492,941.82900,0.22502,942.22346,0.22512,942.61790,0.22522,943.01232,0.22532,943.40671,0.22542,943.80109,0.22552,944.19544,0.22562,944.58977,0.22572,944.98408,0.22582,945.37836,0.22592,945.77263,0.22602,946.16687,0.22612,946.56109,0.22622,946.95529,0.22632,947.34947,0.22642,947.74363,0.22652,948.13776,0.22662,948.53187,0.22672,948.92596,0.22682,949.32003,0.22692,949.71408,0.22702,950.10811,0.22712,950.50211,0.22722,950.89609,0.22732,951.29005,0.22742,951.68399,0.22752,952.07791,0.22762,952.47180,0.22772,952.86567,0.22782,953.25952,0.22792,953.65335,0.22802,954.04716,0.22812,954.44095,0.22822,954.83471,0.22832,955.22845,0.22842,955.62217,0.22852,956.01587,0.22862,956.40955,0.22872,956.80320,0.22882,957.19683,0.22892,957.59045,0.22902,957.98403,0.22912,958.37760,0.22922,958.77115,0.22932,959.16467,0.22942,959.55817,0.22952,959.95166,0.22962,960.34511,0.22972,960.73855,0.22982,961.13197,0.22992,961.52536,0.23002,961.91873,0.23012,962.31208,0.23022,962.70541,0.23032,963.09871,0.23042,963.49200,0.23052,963.88526,0.23062,964.27850,0.23072,964.67172,0.23082,965.06492,0.23092,965.45809,0.23102,965.85125,0.23112,966.24438,0.23122,966.63749,0.23132,967.03058,0.23142,967.42364,0.23152,967.81669,0.23162,968.20971,0.23172,968.60271,0.23182,968.99569,0.23192,969.38865,0.23202,969.78158,0.23212,970.17450,0.23222,970.56739,0.23232,970.96026,0.23242,971.35311,0.23252,971.74594,0.23262,972.13874,0.23272,972.53152,0.23282,972.92428,0.23292,973.31702,0.23302,973.70974,0.23312,974.10244,0.23322,974.49511,0.23332,974.88776,0.23342,975.28040,0.23352,975.67300,0.23362,976.06559,0.23372,976.45816,0.23382,976.85070,0.23392,977.24322,0.23402,977.63572,0.23412,978.02820,0.23422,978.42066,0.23432,978.81309,0.23442,979.20550,0.23452,979.59789,0.23462,979.99026,0.23472,980.38261,0.23482,980.77494,0.23492,981.16724,0.23502,981.55952,0.23512,981.95178,0.23522,982.34402,0.23532,982.73624,0.23542,983.12843,0.23552,983.52060,0.23562,983.91276,0.23572,984.30488,0.23582,984.69699,0.23592,985.08908,0.23602,985.48114,0.23612,985.87318,0.23622,986.26521,0.23632,986.65720,0.23642,987.04918,0.23652,987.44114,0.23662,987.83307,0.23672,988.22498,0.23682,988.61687,0.23692,989.00874,0.23702,989.40058,0.23712,989.79241,0.23722,990.18421,0.23732,990.57599,0.23742,990.96775,0.23752,991.35949,0.23762,991.75120,0.23772,992.14290,0.23782,992.53457,0.23792,992.92622,0.23802,993.31785,0.23812,993.70946,0.23822,994.10104,0.23832,994.49260,0.23842,994.88414,0.23852,995.27566,0.23862,995.66716,0.23872,996.05864,0.23882,996.45009,0.23892,996.84152,0.23902,997.23293,0.23912,997.62432,0.23922,998.01569,0.23932,998.40703,0.23942,998.79836,0.23952,999.18966,0.23962,999.58094,0.23972,999.97220,0.23982,1000.36343,0.23992,1000.75465,0.24002,1001.14584,0.24012,1001.53701,0.24022,1001.92816,0.24032,1002.31929,0.24042,1002.71039,0.24052,1003.10148,0.24062,1003.49254,0.24072,1003.88358,0.24082,1004.27460,0.24092,1004.66560,0.24102,1005.05657,0.24112,1005.44752,0.24122,1005.83845,0.24132,1006.22936,0.24142,1006.62025,0.24152,1007.01112,0.24162,1007.40196,0.24172,1007.79278,0.24182,1008.18358,0.24192,1008.57436,0.24202,1008.96512,0.24212,1009.35586,0.24222,1009.74657,0.24232,1010.13726,0.24242,1010.52793,0.24252,1010.91858,0.24262,1011.30921,0.24272,1011.69981,0.24282,1012.09039,0.24292,1012.48095,0.24302,1012.87149,0.24312,1013.26201,0.24322,1013.65251,0.24332,1014.04298,0.24342,1014.43343,0.24352,1014.82386,0.24362,1015.21427,0.24372,1015.60466,0.24382,1015.99502,0.24392,1016.38537,0.24402,1016.77569,0.24412,1017.16599,0.24422,1017.55627,0.24432,1017.94652,0.24442,1018.33676,0.24452,1018.72697,0.24462,1019.11716,0.24472,1019.50733,0.24482,1019.89748,0.24492,1020.28760,0.24502,1020.67770,0.24512,1021.06779,0.24522,1021.45785,0.24532,1021.84789,0.24542,1022.23790,0.24552,1022.62790,0.24562,1023.01787,0.24572,1023.40782,0.24582,1023.79775,0.24592,1024.18766,0.24602,1024.57754,0.24612,1024.96741,0.24622,1025.35725,0.24632,1025.74707,0.24642,1026.13687,0.24652,1026.52665,0.24662,1026.91640,0.24672,1027.30614,0.24682,1027.69585,0.24692,1028.08554,0.24702,1028.47521,0.24712,1028.86485,0.24722,1029.25448,0.24732,1029.64408,0.24742,1030.03366,0.24752,1030.42322,0.24762,1030.81276,0.24772,1031.20227,0.24782,1031.59177,0.24792,1031.98124,0.24802,1032.37069,0.24812,1032.76012,0.24822,1033.14953,0.24832,1033.53891,0.24842,1033.92828,0.24852,1034.31762,0.24862,1034.70694,0.24872,1035.09624,0.24882,1035.48551,0.24892,1035.87477,0.24902,1036.26400,0.24912,1036.65321,0.24922,1037.04240,0.24932,1037.43157,0.24942,1037.82071,0.24952,1038.20984,0.24962,1038.59894,0.24972,1038.98802,0.24982,1039.37708,0.24992,1039.76611,0.25003,1040.15513,0.25013,1040.54412,0.25023,1040.93309,0.25033,1041.32204,0.25043,1041.71097,0.25053,1042.09988,0.25063,1042.48876,0.25073,1042.87763,0.25083,1043.26647,0.25093,1043.65529,0.25103,1044.04408,0.25113,1044.43286,0.25123,1044.82161,0.25133,1045.21035,0.25143,1045.59906,0.25153,1045.98774,0.25163,1046.37641,0.25173,1046.76506,0.25183,1047.15368,0.25193,1047.54228,0.25203,1047.93086,0.25213,1048.31942,0.25223,1048.70795,0.25233,1049.09647,0.25243,1049.48496,0.25253,1049.87343,0.25263,1050.26188,0.25273,1050.65031,0.25283,1051.03872,0.25293,1051.42710,0.25303,1051.81546,0.25313,1052.20380,0.25323,1052.59212,0.25333,1052.98042,0.25343,1053.36869,0.25353,1053.75695,0.25363,1054.14518,0.25373,1054.53339,0.25383,1054.92158,0.25393,1055.30974,0.25403,1055.69789,0.25413,1056.08601,0.25423,1056.47411,0.25433,1056.86219,0.25443,1057.25025,0.25453,1057.63828,0.25463,1058.02630,0.25473,1058.41429,0.25483,1058.80226,0.25493,1059.19021,0.25503,1059.57813,0.25513,1059.96604,0.25523,1060.35392,0.25533,1060.74178,0.25543,1061.12962,0.25553,1061.51744,0.25563,1061.90524,0.25573,1062.29301,0.25583,1062.68077,0.25593,1063.06850,0.25603,1063.45621,0.25613,1063.84389,0.25623,1064.23156,0.25633,1064.61920,0.25643,1065.00683,0.25653,1065.39443,0.25663,1065.78201,0.25673,1066.16956,0.25683,1066.55710,0.25693,1066.94461,0.25703,1067.33210,0.25713,1067.71957,0.25723,1068.10702,0.25733,1068.49445,0.25743,1068.88185,0.25753,1069.26924,0.25763,1069.65660,0.25773,1070.04394,0.25783,1070.43126,0.25793,1070.81855,0.25803,1071.20583,0.25813,1071.59308,0.25823,1071.98031,0.25833,1072.36752,0.25843,1072.75471,0.25853,1073.14187,0.25863,1073.52901,0.25873,1073.91614,0.25883,1074.30324,0.25893,1074.69032,0.25903,1075.07737,0.25913,1075.46441,0.25923,1075.85142,0.25933,1076.23841,0.25943,1076.62538,0.25953,1077.01233,0.25963,1077.39926,0.25973,1077.78616,0.25983,1078.17304,0.25993,1078.55990,0.26003,1078.94674,0.26013,1079.33356,0.26023,1079.72036,0.26033,1080.10713,0.26043,1080.49388,0.26053,1080.88061,0.26063,1081.26732,0.26073,1081.65401,0.26083,1082.04067,0.26093,1082.42732,0.26103,1082.81394,0.26113,1083.20054,0.26123,1083.58712,0.26133,1083.97367,0.26143,1084.36021,0.26153,1084.74672,0.26163,1085.13321,0.26173,1085.51968,0.26183,1085.90613,0.26193,1086.29256,0.26203,1086.67896,0.26213,1087.06534,0.26223,1087.45170,0.26233,1087.83804,0.26243,1088.22436,0.26253,1088.61066,0.26263,1088.99693,0.26273,1089.38318,0.26283,1089.76941,0.26293,1090.15562,0.26303,1090.54181,0.26313,1090.92797,0.26323,1091.31412,0.26333,1091.70024,0.26343,1092.08634,0.26353,1092.47242,0.26363,1092.85847,0.26373,1093.24451,0.26383,1093.63052,0.26393,1094.01651,0.26403,1094.40248,0.26413,1094.78843,0.26423,1095.17435,0.26433,1095.56026,0.26443,1095.94614,0.26453,1096.33200,0.26463,1096.71784,0.26473,1097.10366,0.26483,1097.48945,0.26493,1097.87523,0.26503,1098.26098,0.26513,1098.64671,0.26523,1099.03242,0.26533,1099.41811,0.26543,1099.80377,0.26553,1100.18941,0.26563,1100.57504,0.26573,1100.96064,0.26583,1101.34621,0.26593,1101.73177,0.26603,1102.11731,0.26613,1102.50282,0.26623,1102.88831,0.26633,1103.27378,0.26643,1103.65923,0.26653,1104.04465,0.26663,1104.43006,0.26673,1104.81544,0.26683,1105.20080,0.26693,1105.58614,0.26703,1105.97146,0.26713,1106.35675,0.26723,1106.74203,0.26733,1107.12728,0.26743,1107.51251,0.26753,1107.89772,0.26763,1108.28291,0.26773,1108.66807,0.26783,1109.05321,0.26793,1109.43834,0.26803,1109.82344,0.26813,1110.20851,0.26823,1110.59357,0.26833,1110.97861,0.26843,1111.36362,0.26853,1111.74861,0.26863,1112.13358,0.26873,1112.51853,0.26883,1112.90345,0.26893,1113.28836,0.26903,1113.67324,0.26913,1114.05810,0.26923,1114.44294,0.26933,1114.82776,0.26943,1115.21256,0.26953,1115.59733,0.26963,1115.98208,0.26973,1116.36681,0.26983,1116.75152,0.26993,1117.13621,0.27003,1117.52087,0.27013,1117.90552,0.27023,1118.29014,0.27033,1118.67474,0.27043,1119.05932,0.27053,1119.44388,0.27063,1119.82841,0.27073,1120.21292,0.27083,1120.59742,0.27093,1120.98189,0.27103,1121.36633,0.27113,1121.75076,0.27123,1122.13517,0.27133,1122.51955,0.27143,1122.90391,0.27153,1123.28825,0.27163,1123.67257,0.27173,1124.05686,0.27183,1124.44114,0.27193,1124.82539,0.27203,1125.20962,0.27213,1125.59383,0.27223,1125.97802,0.27233,1126.36219,0.27243,1126.74633,0.27253,1127.13045,0.27263,1127.51455,0.27273,1127.89863,0.27283,1128.28269,0.27293,1128.66673,0.27303,1129.05074,0.27313,1129.43473,0.27323,1129.81870,0.27333,1130.20265,0.27343,1130.58658,0.27353,1130.97048,0.27363,1131.35437,0.27373,1131.73823,0.27383,1132.12207,0.27393,1132.50589,0.27403,1132.88968,0.27413,1133.27346,0.27423,1133.65721,0.27433,1134.04094,0.27443,1134.42465,0.27453,1134.80834,0.27463,1135.19201,0.27473,1135.57565,0.27483,1135.95928,0.27493,1136.34288,0.27503,1136.72646,0.27513,1137.11001,0.27523,1137.49355,0.27533,1137.87706,0.27543,1138.26056,0.27553,1138.64403,0.27563,1139.02748,0.27573,1139.41091,0.27583,1139.79431,0.27593,1140.17770,0.27603,1140.56106,0.27613,1140.94440,0.27623,1141.32772,0.27633,1141.71102,0.27643,1142.09429,0.27653,1142.47755,0.27663,1142.86078,0.27673,1143.24399,0.27683,1143.62718,0.27693,1144.01034,0.27703,1144.39349,0.27713,1144.77661,0.27723,1145.15972,0.27733,1145.54280,0.27743,1145.92585,0.27753,1146.30889,0.27763,1146.69191,0.27773,1147.07490,0.27783,1147.45787,0.27793,1147.84082,0.27803,1148.22375,0.27813,1148.60666,0.27823,1148.98954,0.27833,1149.37240,0.27843,1149.75525,0.27853,1150.13807,0.27863,1150.52086,0.27873,1150.90364,0.27883,1151.28640,0.27893,1151.66913,0.27903,1152.05184,0.27913,1152.43453,0.27923,1152.81720,0.27933,1153.19984,0.27943,1153.58247,0.27953,1153.96507,0.27963,1154.34765,0.27973,1154.73021,0.27983,1155.11275,0.27993,1155.49527,0.28003,1155.87776,0.28013,1156.26023,0.28023,1156.64268,0.28033,1157.02511,0.28043,1157.40752,0.28053,1157.78991,0.28063,1158.17227,0.28073,1158.55461,0.28083,1158.93693,0.28093,1159.31923,0.28103,1159.70151,0.28113,1160.08377,0.28123,1160.46600,0.28133,1160.84821,0.28143,1161.23040,0.28153,1161.61257,0.28163,1161.99472,0.28173,1162.37685,0.28183,1162.75895,0.28193,1163.14103,0.28203,1163.52309,0.28213,1163.90513,0.28223,1164.28715,0.28233,1164.66914,0.28243,1165.05112,0.28253,1165.43307,0.28263,1165.81500,0.28273,1166.19691,0.28283,1166.57880,0.28293,1166.96066,0.28303,1167.34250,0.28313,1167.72433,0.28323,1168.10613,0.28333,1168.48790,0.28343,1168.86966,0.28353,1169.25140,0.28363,1169.63311,0.28373,1170.01480,0.28383,1170.39647,0.28393,1170.77812,0.28403,1171.15975,0.28413,1171.54135,0.28423,1171.92294,0.28433,1172.30450,0.28443,1172.68604,0.28453,1173.06756,0.28463,1173.44905,0.28473,1173.83053,0.28483,1174.21198,0.28493,1174.59341,0.28503,1174.97482,0.28513,1175.35621,0.28523,1175.73758,0.28533,1176.11892,0.28543,1176.50025,0.28553,1176.88155,0.28563,1177.26283,0.28573,1177.64408,0.28583,1178.02532,0.28593,1178.40654,0.28603,1178.78773,0.28613,1179.16890,0.28623,1179.55005,0.28633,1179.93118,0.28643,1180.31228,0.28653,1180.69337,0.28663,1181.07443,0.28673,1181.45547,0.28683,1181.83649,0.28693,1182.21749,0.28703,1182.59847,0.28713,1182.97942,0.28723,1183.36036,0.28733,1183.74127,0.28743,1184.12216,0.28753,1184.50302,0.28763,1184.88387,0.28773,1185.26470,0.28783,1185.64550,0.28793,1186.02628,0.28803,1186.40704,0.28813,1186.78778,0.28823,1187.16849,0.28833,1187.54919,0.28843,1187.92986,0.28853,1188.31051,0.28863,1188.69114,0.28873,1189.07175,0.28883,1189.45234,0.28893,1189.83290,0.28903,1190.21344,0.28913,1190.59396,0.28923,1190.97446,0.28933,1191.35494,0.28943,1191.73540,0.28953,1192.11583,0.28963,1192.49625,0.28973,1192.87664,0.28983,1193.25701,0.28993,1193.63735,0.29003,1194.01768,0.29013,1194.39799,0.29023,1194.77827,0.29033,1195.15853,0.29043,1195.53877,0.29053,1195.91899,0.29063,1196.29918,0.29073,1196.67936,0.29083,1197.05951,0.29093,1197.43964,0.29103,1197.81975,0.29113,1198.19984,0.29123,1198.57991,0.29133,1198.95995,0.29143,1199.33997,0.29153,1199.71998,0.29163,1200.09996,0.29173,1200.47991,0.29183,1200.85985,0.29193,1201.23976,0.29203,1201.61966,0.29213,1201.99953,0.29223,1202.37938,0.29233,1202.75921,0.29243,1203.13901,0.29253,1203.51880,0.29263,1203.89856,0.29273,1204.27830,0.29283,1204.65802,0.29293,1205.03772,0.29303,1205.41740,0.29313,1205.79705,0.29323,1206.17668,0.29333,1206.55630,0.29343,1206.93589,0.29353,1207.31545,0.29363,1207.69500,0.29373,1208.07453,0.29383,1208.45403,0.29393,1208.83351,0.29403,1209.21297,0.29413,1209.59241,0.29423,1209.97182,0.29433,1210.35122,0.29443,1210.73059,0.29453,1211.10994,0.29463,1211.48927,0.29473,1211.86858,0.29483,1212.24787,0.29493,1212.62714,0.29503,1213.00638,0.29513,1213.38560,0.29523,1213.76480,0.29533,1214.14398,0.29543,1214.52314,0.29553,1214.90227,0.29563,1215.28138,0.29573,1215.66048,0.29583,1216.03955,0.29593,1216.41860,0.29603,1216.79762,0.29613,1217.17663,0.29623,1217.55561,0.29633,1217.93457,0.29643,1218.31351,0.29653,1218.69243,0.29663,1219.07133,0.29673,1219.45021,0.29683,1219.82906,0.29693,1220.20789,0.29703,1220.58670,0.29713,1220.96549,0.29723,1221.34426,0.29733,1221.72300,0.29743,1222.10173,0.29753,1222.48043,0.29763,1222.85911,0.29773,1223.23777,0.29783,1223.61641,0.29793,1223.99502,0.29803,1224.37362,0.29813,1224.75219,0.29823,1225.13074,0.29833,1225.50927,0.29843,1225.88778,0.29853,1226.26626,0.29863,1226.64473,0.29873,1227.02317,0.29883,1227.40159,0.29893,1227.77999,0.29903,1228.15837,0.29913,1228.53673,0.29923,1228.91506,0.29933,1229.29337,0.29943,1229.67166,0.29953,1230.04993,0.29963,1230.42818,0.29973,1230.80641,0.29983,1231.18461,0.29993,1231.56280,0.30003,1231.94096,0.30013,1232.31910,0.30023,1232.69722,0.30033,1233.07531,0.30043,1233.45339,0.30053,1233.83144,0.30063,1234.20947,0.30073,1234.58748,0.30083,1234.96547,0.30093,1235.34344,0.30103,1235.72138,0.30113,1236.09931,0.30123,1236.47721,0.30133,1236.85509,0.30143,1237.23295,0.30153,1237.61079,0.30163,1237.98860,0.30173,1238.36640,0.30183,1238.74417,0.30193,1239.12192,0.30203,1239.49965,0.30213,1239.87736,0.30223,1240.25504,0.30233,1240.63271,0.30243,1241.01035,0.30253,1241.38797,0.30263,1241.76557,0.30273,1242.14315,0.30283,1242.52070,0.30293,1242.89824,0.30303,1243.27575,0.30313,1243.65324,0.30323,1244.03071,0.30333,1244.40816,0.30343,1244.78558,0.30353,1245.16299,0.30363,1245.54037,0.30373,1245.91773,0.30383,1246.29507,0.30393,1246.67239,0.30403,1247.04969,0.30413,1247.42696,0.30423,1247.80422,0.30433,1248.18145,0.30443,1248.55866,0.30453,1248.93585,0.30463,1249.31302,0.30473,1249.69016,0.30483,1250.06729,0.30493,1250.44439,0.30503,1250.82147,0.30513,1251.19853,0.30523,1251.57556,0.30533,1251.95258,0.30543,1252.32957,0.30553,1252.70655,0.30563,1253.08350,0.30573,1253.46043,0.30583,1253.83733,0.30593,1254.21422,0.30603,1254.59109,0.30613,1254.96793,0.30623,1255.34475,0.30633,1255.72155,0.30643,1256.09833,0.30653,1256.47508,0.30663,1256.85182,0.30673,1257.22853,0.30683,1257.60522,0.30693,1257.98189,0.30703,1258.35854,0.30713,1258.73517,0.30723,1259.11177,0.30733,1259.48836,0.30743,1259.86492,0.30753,1260.24146,0.30763,1260.61798,0.30773,1260.99448,0.30783,1261.37095,0.30793,1261.74741,0.30803,1262.12384,0.30813,1262.50025,0.30823,1262.87664,0.30833,1263.25301,0.30843,1263.62935,0.30853,1264.00568,0.30863,1264.38198,0.30873,1264.75826,0.30883,1265.13452,0.30893,1265.51076,0.30903,1265.88698,0.30913,1266.26317,0.30923,1266.63935,0.30933,1267.01550,0.30943,1267.39163,0.30953,1267.76774,0.30963,1268.14382,0.30973,1268.51989,0.30983,1268.89593,0.30993,1269.27195,0.31003,1269.64796,0.31013,1270.02393,0.31023,1270.39989,0.31033,1270.77583,0.31043,1271.15174,0.31053,1271.52763,0.31063,1271.90351,0.31073,1272.27936,0.31083,1272.65518,0.31093,1273.03099,0.31103,1273.40677,0.31113,1273.78254,0.31123,1274.15828,0.31133,1274.53400,0.31143,1274.90970,0.31153,1275.28538,0.31163,1275.66103,0.31173,1276.03666,0.31183,1276.41228,0.31193,1276.78787,0.31203,1277.16344,0.31213,1277.53898,0.31223,1277.91451,0.31233,1278.29001,0.31243,1278.66550,0.31253,1279.04096,0.31263,1279.41640,0.31273,1279.79181,0.31283,1280.16721,0.31293,1280.54259,0.31303,1280.91794,0.31313,1281.29327,0.31323,1281.66858,0.31333,1282.04387,0.31343,1282.41914,0.31353,1282.79438,0.31363,1283.16960,0.31373,1283.54481,0.31383,1283.91999,0.31393,1284.29515,0.31403,1284.67028,0.31413,1285.04540,0.31423,1285.42049,0.31433,1285.79557,0.31443,1286.17062,0.31453,1286.54565,0.31463,1286.92066,0.31473,1287.29564,0.31483,1287.67061,0.31493,1288.04555,0.31503,1288.42047,0.31513,1288.79537,0.31523,1289.17025,0.31533,1289.54511,0.31543,1289.91994,0.31553,1290.29476,0.31563,1290.66955,0.31573,1291.04432,0.31583,1291.41907,0.31593,1291.79380,0.31603,1292.16850,0.31613,1292.54319,0.31623,1292.91785,0.31633,1293.29249,0.31643,1293.66711,0.31653,1294.04171,0.31663,1294.41629,0.31673,1294.79084,0.31683,1295.16538,0.31693,1295.53989,0.31703,1295.91438,0.31713,1296.28885,0.31723,1296.66330,0.31733,1297.03772,0.31743,1297.41213,0.31753,1297.78651,0.31763,1298.16087,0.31773,1298.53521,0.31783,1298.90953,0.31793,1299.28382,0.31803,1299.65810,0.31813,1300.03235,0.31823,1300.40658,0.31833,1300.78080,0.31843,1301.15498,0.31853,1301.52915,0.31863,1301.90330,0.31873,1302.27742,0.31883,1302.65152,0.31893,1303.02560,0.31903,1303.39966,0.31913,1303.77370,0.31923,1304.14772,0.31933,1304.52171,0.31943,1304.89569,0.31953,1305.26964,0.31963,1305.64357,0.31973,1306.01748,0.31983,1306.39136,0.31993,1306.76523,0.32003,1307.13907,0.32013,1307.51290,0.32023,1307.88670,0.32033,1308.26048,0.32043,1308.63423,0.32053,1309.00797,0.32063,1309.38169,0.32073,1309.75538,0.32083,1310.12905,0.32093,1310.50270,0.32103,1310.87633,0.32113,1311.24994,0.32123,1311.62352,0.32133,1311.99709,0.32143,1312.37063,0.32153,1312.74415,0.32163,1313.11765,0.32173,1313.49113,0.32183,1313.86458,0.32193,1314.23802,0.32203,1314.61143,0.32213,1314.98482,0.32223,1315.35819,0.32233,1315.73154,0.32243,1316.10487,0.32253,1316.47817,0.32263,1316.85146,0.32273,1317.22472,0.32283,1317.59796,0.32293,1317.97118,0.32303,1318.34438,0.32313,1318.71755,0.32323,1319.09071,0.32333,1319.46384,0.32343,1319.83695,0.32353,1320.21004,0.32363,1320.58311,0.32373,1320.95616,0.32383,1321.32919,0.32393,1321.70219,0.32403,1322.07517,0.32413,1322.44813,0.32423,1322.82107,0.32433,1323.19399,0.32443,1323.56689,0.32453,1323.93976,0.32463,1324.31262,0.32473,1324.68545,0.32483,1325.05826,0.32493,1325.43105,0.32503,1325.80381,0.32513,1326.17656,0.32523,1326.54928,0.32533,1326.92199,0.32543,1327.29467,0.32553,1327.66733,0.32563,1328.03997,0.32573,1328.41258,0.32583,1328.78518,0.32593,1329.15775,0.32603,1329.53030,0.32613,1329.90284,0.32623,1330.27534,0.32633,1330.64783,0.32643,1331.02030,0.32653,1331.39274,0.32663,1331.76517,0.32673,1332.13757,0.32683,1332.50995,0.32693,1332.88231,0.32703,1333.25464,0.32713,1333.62696,0.32723,1333.99925,0.32733,1334.37153,0.32743,1334.74378,0.32753,1335.11601,0.32763,1335.48821,0.32773,1335.86040,0.32783,1336.23256,0.32793,1336.60471,0.32803,1336.97683,0.32813,1337.34893,0.32823,1337.72101,0.32833,1338.09307,0.32843,1338.46510,0.32853,1338.83712,0.32863,1339.20911,0.32873,1339.58108,0.32883,1339.95303,0.32893,1340.32496,0.32903,1340.69687,0.32913,1341.06875,0.32923,1341.44062,0.32933,1341.81246,0.32943,1342.18428,0.32953,1342.55608,0.32963,1342.92786,0.32973,1343.29961,0.32983,1343.67135,0.32993,1344.04306,0.33003,1344.41475,0.33013,1344.78642,0.33023,1345.15807,0.33033,1345.52970,0.33043,1345.90131,0.33053,1346.27289,0.33063,1346.64445,0.33073,1347.01600,0.33083,1347.38752,0.33093,1347.75901,0.33103,1348.13049,0.33113,1348.50195,0.33123,1348.87338,0.33133,1349.24479,0.33143,1349.61618,0.33153,1349.98755,0.33163,1350.35890,0.33173,1350.73023,0.33183,1351.10153,0.33193,1351.47282,0.33203,1351.84408,0.33213,1352.21532,0.33223,1352.58654,0.33233,1352.95774,0.33243,1353.32891,0.33253,1353.70007,0.33263,1354.07120,0.33273,1354.44231,0.33283,1354.81340,0.33293,1355.18447,0.33303,1355.55552,0.33313,1355.92654,0.33323,1356.29755,0.33333,1356.66853,0.33343,1357.03949,0.33353,1357.41043,0.33363,1357.78135,0.33373,1358.15225,0.33383,1358.52312,0.33393,1358.89398,0.33403,1359.26481,0.33413,1359.63562,0.33423,1360.00641,0.33433,1360.37718,0.33443,1360.74792,0.33453,1361.11865,0.33463,1361.48935,0.33473,1361.86003,0.33483,1362.23070,0.33493,1362.60133,0.33503,1362.97195,0.33513,1363.34255,0.33523,1363.71312,0.33533,1364.08368,0.33543,1364.45421,0.33553,1364.82472,0.33563,1365.19521,0.33573,1365.56568,0.33583,1365.93612,0.33593,1366.30655,0.33603,1366.67695,0.33613,1367.04733,0.33623,1367.41769,0.33633,1367.78803,0.33643,1368.15835,0.33653,1368.52864,0.33663,1368.89892,0.33673,1369.26917,0.33683,1369.63940,0.33693,1370.00961,0.33703,1370.37980,0.33713,1370.74997,0.33723,1371.12011,0.33733,1371.49024,0.33743,1371.86034,0.33753,1372.23042,0.33763,1372.60048,0.33773,1372.97052,0.33783,1373.34053,0.33793,1373.71053,0.33803,1374.08050,0.33813,1374.45045,0.33823,1374.82039,0.33833,1375.19030,0.33843,1375.56018,0.33853,1375.93005,0.33863,1376.29989,0.33873,1376.66972,0.33883,1377.03952,0.33893,1377.40930,0.33903,1377.77906,0.33913,1378.14880,0.33923,1378.51851,0.33933,1378.88821,0.33943,1379.25788,0.33953,1379.62754,0.33963,1379.99717,0.33973,1380.36678,0.33983,1380.73636,0.33993,1381.10593,0.34003,1381.47547,0.34013,1381.84500,0.34023,1382.21450,0.34033,1382.58398,0.34043,1382.95344,0.34053,1383.32288,0.34063,1383.69229,0.34073,1384.06169,0.34083,1384.43106,0.34093,1384.80041,0.34103,1385.16974,0.34113,1385.53905,0.34123,1385.90834,0.34133,1386.27761,0.34143,1386.64685,0.34153,1387.01607,0.34163,1387.38527,0.34173,1387.75445,0.34183,1388.12361,0.34193,1388.49275,0.34203,1388.86187,0.34213,1389.23096,0.34223,1389.60003,0.34233,1389.96909,0.34243,1390.33812,0.34253,1390.70713,0.34263,1391.07611,0.34273,1391.44508,0.34283,1391.81402,0.34293,1392.18295,0.34303,1392.55185,0.34313,1392.92073,0.34323,1393.28959,0.34333,1393.65843,0.34343,1394.02724,0.34353,1394.39604,0.34363,1394.76481,0.34373,1395.13356,0.34383,1395.50229,0.34393,1395.87100,0.34403,1396.23969,0.34413,1396.60835,0.34423,1396.97700,0.34433,1397.34562,0.34443,1397.71422,0.34453,1398.08280,0.34463,1398.45136,0.34473,1398.81990,0.34483,1399.18842,0.34493,1399.55691,0.34503,1399.92538,0.34513,1400.29383,0.34523,1400.66227,0.34533,1401.03067,0.34543,1401.39906,0.34553,1401.76743,0.34563,1402.13577,0.34573,1402.50410,0.34583,1402.87240,0.34593,1403.24068,0.34603,1403.60894,0.34613,1403.97718,0.34623,1404.34539,0.34633,1404.71359,0.34643,1405.08176,0.34653,1405.44991,0.34663,1405.81804,0.34673,1406.18615,0.34683,1406.55424,0.34693,1406.92231,0.34703,1407.29035,0.34713,1407.65837,0.34723,1408.02638,0.34733,1408.39436,0.34743,1408.76232,0.34753,1409.13025,0.34763,1409.49817,0.34773,1409.86607,0.34783,1410.23394,0.34793,1410.60179,0.34803,1410.96962,0.34813,1411.33743,0.34823,1411.70522,0.34833,1412.07299,0.34843,1412.44073,0.34853,1412.80846,0.34863,1413.17616,0.34873,1413.54384,0.34883,1413.91150,0.34893,1414.27914,0.34903,1414.64676,0.34913,1415.01435,0.34923,1415.38192,0.34933,1415.74948,0.34943,1416.11701,0.34953,1416.48452,0.34963,1416.85201,0.34973,1417.21947,0.34983,1417.58692,0.34993,1417.95434,0.35004,1418.32175,0.35014,1418.68913,0.35024,1419.05649,0.35034,1419.42383,0.35044,1419.79115,0.35054,1420.15844,0.35064,1420.52572,0.35074,1420.89297,0.35084,1421.26020,0.35094,1421.62741,0.35104,1421.99460,0.35114,1422.36177,0.35124,1422.72892,0.35134,1423.09604,0.35144,1423.46314,0.35154,1423.83023,0.35164,1424.19729,0.35174,1424.56433,0.35184,1424.93134,0.35194,1425.29834,0.35204,1425.66532,0.35214,1426.03227,0.35224,1426.39920,0.35234,1426.76611,0.35244,1427.13300,0.35254,1427.49987,0.35264,1427.86672,0.35274,1428.23355,0.35284,1428.60035,0.35294,1428.96713,0.35304,1429.33389,0.35314,1429.70063,0.35324,1430.06735,0.35334,1430.43405,0.35344,1430.80073,0.35354,1431.16738,0.35364,1431.53401,0.35374,1431.90063,0.35384,1432.26722,0.35394,1432.63379,0.35404,1433.00033,0.35414,1433.36686,0.35424,1433.73337,0.35434,1434.09985,0.35444,1434.46631,0.35454,1434.83275,0.35464,1435.19917,0.35474,1435.56557,0.35484,1435.93195,0.35494,1436.29830,0.35504,1436.66464,0.35514,1437.03095,0.35524,1437.39724,0.35534,1437.76351,0.35544,1438.12976,0.35554,1438.49599,0.35564,1438.86219,0.35574,1439.22838,0.35584,1439.59454,0.35594,1439.96068,0.35604,1440.32681,0.35614,1440.69290,0.35624,1441.05898,0.35634,1441.42504,0.35644,1441.79107,0.35654,1442.15709,0.35664,1442.52308,0.35674,1442.88905,0.35684,1443.25500,0.35694,1443.62093,0.35704,1443.98684,0.35714,1444.35272,0.35724,1444.71859,0.35734,1445.08443,0.35744,1445.45025,0.35754,1445.81605,0.35764,1446.18183,0.35774,1446.54759,0.35784,1446.91333,0.35794,1447.27904,0.35804,1447.64473,0.35814,1448.01041,0.35824,1448.37606,0.35834,1448.74169,0.35844,1449.10729,0.35854,1449.47288,0.35864,1449.83845,0.35874,1450.20399,0.35884,1450.56951,0.35894,1450.93502,0.35904,1451.30050,0.35914,1451.66595,0.35924,1452.03139,0.35934,1452.39681,0.35944,1452.76220,0.35954,1453.12758,0.35964,1453.49293,0.35974,1453.85826,0.35984,1454.22357,0.35994,1454.58886,0.36004,1454.95412,0.36014,1455.31937,0.36024,1455.68459,0.36034,1456.04980,0.36044,1456.41498,0.36054,1456.78014,0.36064,1457.14528,0.36074,1457.51040,0.36084,1457.87549,0.36094,1458.24057,0.36104,1458.60562,0.36114,1458.97065,0.36124,1459.33566,0.36134,1459.70065,0.36144,1460.06562,0.36154,1460.43057,0.36164,1460.79549,0.36174,1461.16040,0.36184,1461.52528,0.36194,1461.89014,0.36204,1462.25498,0.36214,1462.61980,0.36224,1462.98460,0.36234,1463.34938,0.36244,1463.71413,0.36254,1464.07887,0.36264,1464.44358,0.36274,1464.80827,0.36284,1465.17294,0.36294,1465.53759,0.36304,1465.90221,0.36314,1466.26682,0.36324,1466.63140,0.36334,1466.99597,0.36344,1467.36051,0.36354,1467.72503,0.36364,1468.08953,0.36374,1468.45401,0.36384,1468.81846,0.36394,1469.18290,0.36404,1469.54731,0.36414,1469.91171,0.36424,1470.27608,0.36434,1470.64043,0.36444,1471.00476,0.36454,1471.36906,0.36464,1471.73335,0.36474,1472.09762,0.36484,1472.46186,0.36494,1472.82608,0.36504,1473.19028,0.36514,1473.55446,0.36524,1473.91862,0.36534,1474.28276,0.36544,1474.64687,0.36554,1475.01097,0.36564,1475.37504,0.36574,1475.73909,0.36584,1476.10312,0.36594,1476.46713,0.36604,1476.83112,0.36614,1477.19509,0.36624,1477.55903,0.36634,1477.92296,0.36644,1478.28686,0.36654,1478.65074,0.36664,1479.01460,0.36674,1479.37844,0.36684,1479.74226,0.36694,1480.10606,0.36704,1480.46983,0.36714,1480.83358,0.36724,1481.19732,0.36734,1481.56103,0.36744,1481.92472,0.36754,1482.28839,0.36764,1482.65203,0.36774,1483.01566,0.36784,1483.37927,0.36794,1483.74285,0.36804,1484.10641,0.36814,1484.46995,0.36824,1484.83347,0.36834,1485.19697,0.36844,1485.56045,0.36854,1485.92390,0.36864,1486.28734,0.36874,1486.65075,0.36884,1487.01414,0.36894,1487.37751,0.36904,1487.74086,0.36914,1488.10419,0.36924,1488.46750,0.36934,1488.83078,0.36944,1489.19405,0.36954,1489.55729,0.36964,1489.92051,0.36974,1490.28371,0.36984,1490.64689,0.36994,1491.01005,0.37004,1491.37319,0.37014,1491.73630,0.37024,1492.09940,0.37034,1492.46247,0.37044,1492.82552,0.37054,1493.18855,0.37064,1493.55156,0.37074,1493.91455,0.37084,1494.27752,0.37094,1494.64046,0.37104,1495.00338,0.37114,1495.36629,0.37124,1495.72917,0.37134,1496.09203,0.37144,1496.45487,0.37154,1496.81769,0.37164,1497.18048,0.37174,1497.54326,0.37184,1497.90601,0.37194,1498.26874,0.37204,1498.63146,0.37214,1498.99415,0.37224,1499.35681,0.37234,1499.71946,0.37244,1500.08209,0.37254,1500.44469,0.37264,1500.80728,0.37274,1501.16984,0.37284,1501.53238,0.37294,1501.89490,0.37304,1502.25740,0.37314,1502.61988,0.37324,1502.98234,0.37334,1503.34477,0.37344,1503.70718,0.37354,1504.06958,0.37364,1504.43195,0.37374,1504.79430,0.37384,1505.15663,0.37394,1505.51893,0.37404,1505.88122,0.37414,1506.24349,0.37424,1506.60573,0.37434,1506.96795,0.37444,1507.33015,0.37454,1507.69233,0.37464,1508.05449,0.37474,1508.41663,0.37484,1508.77875,0.37494,1509.14084,0.37504,1509.50292,0.37514,1509.86497,0.37524,1510.22700,0.37534,1510.58901,0.37544,1510.95100,0.37554,1511.31297,0.37564,1511.67492,0.37574,1512.03684,0.37584,1512.39874,0.37594,1512.76063,0.37604,1513.12249,0.37614,1513.48433,0.37624,1513.84615,0.37634,1514.20795,0.37644,1514.56972,0.37654,1514.93148,0.37664,1515.29321,0.37674,1515.65493,0.37684,1516.01662,0.37694,1516.37829,0.37704,1516.73994,0.37714,1517.10157,0.37724,1517.46317,0.37734,1517.82476,0.37744,1518.18632,0.37754,1518.54787,0.37764,1518.90939,0.37774,1519.27089,0.37784,1519.63237,0.37794,1519.99383,0.37804,1520.35526,0.37814,1520.71668,0.37824,1521.07807,0.37834,1521.43945,0.37844,1521.80080,0.37854,1522.16213,0.37864,1522.52344,0.37874,1522.88473,0.37884,1523.24600,0.37894,1523.60724,0.37904,1523.96847,0.37914,1524.32967,0.37924,1524.69085,0.37934,1525.05202,0.37944,1525.41316,0.37954,1525.77427,0.37964,1526.13537,0.37974,1526.49645,0.37984,1526.85750,0.37994,1527.21854,0.38004,1527.57955,0.38014,1527.94054,0.38024,1528.30151,0.38034,1528.66246,0.38044,1529.02339,0.38054,1529.38430,0.38064,1529.74518,0.38074,1530.10605,0.38084,1530.46689,0.38094,1530.82771,0.38104,1531.18851,0.38114,1531.54929,0.38124,1531.91005,0.38134,1532.27079,0.38144,1532.63151,0.38154,1532.99220,0.38164,1533.35287,0.38174,1533.71353,0.38184,1534.07416,0.38194,1534.43477,0.38204,1534.79536,0.38214,1535.15592,0.38224,1535.51647,0.38234,1535.87700,0.38244,1536.23750,0.38254,1536.59798,0.38264,1536.95845,0.38274,1537.31889,0.38284,1537.67931,0.38294,1538.03970,0.38304,1538.40008,0.38314,1538.76044,0.38324,1539.12077,0.38334,1539.48108,0.38344,1539.84138,0.38354,1540.20165,0.38364,1540.56190,0.38374,1540.92213,0.38384,1541.28233,0.38394,1541.64252,0.38404,1542.00269,0.38414,1542.36283,0.38424,1542.72295,0.38434,1543.08305,0.38444,1543.44313,0.38454,1543.80319,0.38464,1544.16323,0.38474,1544.52325,0.38484,1544.88324,0.38494,1545.24322,0.38504,1545.60317,0.38514,1545.96310,0.38524,1546.32302,0.38534,1546.68291,0.38544,1547.04277,0.38554,1547.40262,0.38564,1547.76245,0.38574,1548.12225,0.38584,1548.48204,0.38594,1548.84180,0.38604,1549.20154,0.38614,1549.56126,0.38624,1549.92096,0.38634,1550.28064,0.38644,1550.64030,0.38654,1550.99993,0.38664,1551.35955,0.38674,1551.71914,0.38684,1552.07871,0.38694,1552.43827,0.38704,1552.79780,0.38714,1553.15730,0.38724,1553.51679,0.38734,1553.87626,0.38744,1554.23570,0.38754,1554.59513,0.38764,1554.95453,0.38774,1555.31391,0.38784,1555.67328,0.38794,1556.03261,0.38804,1556.39193,0.38814,1556.75123,0.38824,1557.11051,0.38834,1557.46976,0.38844,1557.82900,0.38854,1558.18821,0.38864,1558.54740,0.38874,1558.90657,0.38884,1559.26572,0.38894,1559.62485,0.38904,1559.98396,0.38914,1560.34304,0.38924,1560.70211,0.38934,1561.06115,0.38944,1561.42017,0.38954,1561.77917,0.38964,1562.13815,0.38974,1562.49711,0.38984,1562.85605,0.38994,1563.21497,0.39004,1563.57386,0.39014,1563.93274,0.39024,1564.29159,0.39034,1564.65042,0.39044,1565.00924,0.39054,1565.36803,0.39064,1565.72679,0.39074,1566.08554,0.39084,1566.44427,0.39094,1566.80297,0.39104,1567.16166,0.39114,1567.52032,0.39124,1567.87896,0.39134,1568.23758,0.39144,1568.59618,0.39154,1568.95476,0.39164,1569.31332,0.39174,1569.67186,0.39184,1570.03037,0.39194,1570.38887,0.39204,1570.74734,0.39214,1571.10579,0.39224,1571.46422,0.39234,1571.82263,0.39244,1572.18102,0.39254,1572.53939,0.39264,1572.89773,0.39274,1573.25606,0.39284,1573.61436,0.39294,1573.97265,0.39304,1574.33091,0.39314,1574.68915,0.39324,1575.04737,0.39334,1575.40557,0.39344,1575.76374,0.39354,1576.12190,0.39364,1576.48004,0.39374,1576.83815,0.39384,1577.19624,0.39394,1577.55431,0.39404,1577.91237,0.39414,1578.27039,0.39424,1578.62840,0.39434,1578.98639,0.39444,1579.34436,0.39454,1579.70230,0.39464,1580.06023,0.39474,1580.41813,0.39484,1580.77601,0.39494,1581.13387,0.39504,1581.49171,0.39514,1581.84953,0.39524,1582.20733,0.39534,1582.56510,0.39544,1582.92286,0.39554,1583.28059,0.39564,1583.63831,0.39574,1583.99600,0.39584,1584.35367,0.39594,1584.71132,0.39604,1585.06895,0.39614,1585.42655,0.39624,1585.78414,0.39634,1586.14171,0.39644,1586.49925,0.39654,1586.85677,0.39664,1587.21428,0.39674,1587.57176,0.39684,1587.92922,0.39694,1588.28666,0.39704,1588.64407,0.39714,1589.00147,0.39724,1589.35885,0.39734,1589.71620,0.39744,1590.07353,0.39754,1590.43085,0.39764,1590.78814,0.39774,1591.14541,0.39784,1591.50266,0.39794,1591.85988,0.39804,1592.21709,0.39814,1592.57428,0.39824,1592.93144,0.39834,1593.28859,0.39844,1593.64571,0.39854,1594.00281,0.39864,1594.35989,0.39874,1594.71695,0.39884,1595.07399,0.39894,1595.43100,0.39904,1595.78800,0.39914,1596.14498,0.39924,1596.50193,0.39934,1596.85886,0.39944,1597.21577,0.39954,1597.57266,0.39964,1597.92953,0.39974,1598.28638,0.39984,1598.64321,0.39994,1599.00002,0.40004,1599.35680,0.40014,1599.71357,0.40024,1600.07031,0.40034,1600.42703,0.40044,1600.78373,0.40054,1601.14041,0.40064,1601.49707,0.40074,1601.85371,0.40084,1602.21033,0.40094,1602.56692,0.40104,1602.92350,0.40114,1603.28005,0.40124,1603.63658,0.40134,1603.99310,0.40144,1604.34959,0.40154,1604.70606,0.40164,1605.06250,0.40174,1605.41893,0.40184,1605.77534,0.40194,1606.13172,0.40204,1606.48809,0.40214,1606.84443,0.40224,1607.20075,0.40234,1607.55705,0.40244,1607.91333,0.40254,1608.26959,0.40264,1608.62583,0.40274,1608.98205,0.40284,1609.33824,0.40294,1609.69442,0.40304,1610.05057,0.40314,1610.40670,0.40324,1610.76282,0.40334,1611.11891,0.40344,1611.47498,0.40354,1611.83102,0.40364,1612.18705,0.40374,1612.54306,0.40384,1612.89904,0.40394,1613.25501,0.40404,1613.61095,0.40414,1613.96687,0.40424,1614.32277,0.40434,1614.67865,0.40444,1615.03451,0.40454,1615.39035,0.40464,1615.74617,0.40474,1616.10196,0.40484,1616.45774,0.40494,1616.81349,0.40504,1617.16923,0.40514,1617.52494,0.40524,1617.88063,0.40534,1618.23630,0.40544,1618.59195,0.40554,1618.94758,0.40564,1619.30318,0.40574,1619.65877,0.40584,1620.01433,0.40594,1620.36988,0.40604,1620.72540,0.40614,1621.08090,0.40624,1621.43638,0.40634,1621.79184,0.40644,1622.14728,0.40654,1622.50270,0.40664,1622.85809,0.40674,1623.21347,0.40684,1623.56882,0.40694,1623.92416,0.40704,1624.27947,0.40714,1624.63476,0.40724,1624.99003,0.40734,1625.34528,0.40744,1625.70051,0.40754,1626.05572,0.40764,1626.41090,0.40774,1626.76607,0.40784,1627.12121,0.40794,1627.47634,0.40804,1627.83144,0.40814,1628.18652,0.40824,1628.54158,0.40834,1628.89662,0.40844,1629.25164,0.40854,1629.60663,0.40864,1629.96161,0.40874,1630.31657,0.40884,1630.67150,0.40894,1631.02641,0.40904,1631.38131,0.40914,1631.73618,0.40924,1632.09103,0.40934,1632.44586,0.40944,1632.80066,0.40954,1633.15545,0.40964,1633.51022,0.40974,1633.86496,0.40984,1634.21969,0.40994,1634.57439,0.41004,1634.92907,0.41014,1635.28373,0.41024,1635.63838,0.41034,1635.99299,0.41044,1636.34759,0.41054,1636.70217,0.41064,1637.05673,0.41074,1637.41126,0.41084,1637.76578,0.41094,1638.12027,0.41104,1638.47474,0.41114,1638.82919,0.41124,1639.18362,0.41134,1639.53803,0.41144,1639.89242,0.41154,1640.24679,0.41164,1640.60114,0.41174,1640.95546,0.41184,1641.30977,0.41194,1641.66405,0.41204,1642.01831,0.41214,1642.37255,0.41224,1642.72677,0.41234,1643.08097,0.41244,1643.43515,0.41254,1643.78931,0.41264,1644.14345,0.41274,1644.49756,0.41284,1644.85166,0.41294,1645.20573,0.41304,1645.55978,0.41314,1645.91382,0.41324,1646.26783,0.41334,1646.62182,0.41344,1646.97579,0.41354,1647.32973,0.41364,1647.68366,0.41374,1648.03757,0.41384,1648.39145,0.41394,1648.74532,0.41404,1649.09916,0.41414,1649.45298,0.41424,1649.80678,0.41434,1650.16056,0.41444,1650.51432,0.41454,1650.86806,0.41464,1651.22178,0.41474,1651.57547,0.41484,1651.92915,0.41494,1652.28280,0.41504,1652.63644,0.41514,1652.99005,0.41524,1653.34364,0.41534,1653.69721,0.41544,1654.05076,0.41554,1654.40429,0.41564,1654.75780,0.41574,1655.11129,0.41584,1655.46475,0.41594,1655.81820,0.41604,1656.17162,0.41614,1656.52502,0.41624,1656.87841,0.41634,1657.23177,0.41644,1657.58511,0.41654,1657.93843,0.41664,1658.29173,0.41674,1658.64500,0.41684,1658.99826,0.41694,1659.35149,0.41704,1659.70471,0.41714,1660.05790,0.41724,1660.41108,0.41734,1660.76423,0.41744,1661.11736,0.41754,1661.47047,0.41764,1661.82356,0.41774,1662.17663,0.41784,1662.52967,0.41794,1662.88270,0.41804,1663.23570,0.41814,1663.58869,0.41824,1663.94165,0.41834,1664.29459,0.41844,1664.64752,0.41854,1665.00042,0.41864,1665.35330,0.41874,1665.70615,0.41884,1666.05899,0.41894,1666.41181,0.41904,1666.76460,0.41914,1667.11738,0.41924,1667.47013,0.41934,1667.82287,0.41944,1668.17558,0.41954,1668.52827,0.41964,1668.88094,0.41974,1669.23359,0.41984,1669.58622,0.41994,1669.93883,0.42004,1670.29141,0.42014,1670.64398,0.42024,1670.99652,0.42034,1671.34905,0.42044,1671.70155,0.42054,1672.05403,0.42064,1672.40649,0.42074,1672.75893,0.42084,1673.11135,0.42094,1673.46375,0.42104,1673.81613,0.42114,1674.16849,0.42124,1674.52082,0.42134,1674.87314,0.42144,1675.22543,0.42154,1675.57770,0.42164,1675.92996,0.42174,1676.28219,0.42184,1676.63440,0.42194,1676.98659,0.42204,1677.33876,0.42214,1677.69090,0.42224,1678.04303,0.42234,1678.39514,0.42244,1678.74722,0.42254,1679.09929,0.42264,1679.45133,0.42274,1679.80335,0.42284,1680.15535,0.42294,1680.50733,0.42304,1680.85929,0.42314,1681.21123,0.42324,1681.56315,0.42334,1681.91505,0.42344,1682.26692,0.42354,1682.61878,0.42364,1682.97061,0.42374,1683.32242,0.42384,1683.67422,0.42394,1684.02599,0.42404,1684.37774,0.42414,1684.72947,0.42424,1685.08118,0.42434,1685.43286,0.42444,1685.78453,0.42454,1686.13618,0.42464,1686.48780,0.42474,1686.83941,0.42484,1687.19099,0.42494,1687.54255,0.42504,1687.89410,0.42514,1688.24562,0.42524,1688.59712,0.42534,1688.94860,0.42544,1689.30005,0.42554,1689.65149,0.42564,1690.00291,0.42574,1690.35430,0.42584,1690.70568,0.42594,1691.05703,0.42604,1691.40837,0.42614,1691.75968,0.42624,1692.11097,0.42634,1692.46224,0.42644,1692.81349,0.42654,1693.16472,0.42664,1693.51593,0.42674,1693.86711,0.42684,1694.21828,0.42694,1694.56942,0.42704,1694.92055,0.42714,1695.27165,0.42724,1695.62274,0.42734,1695.97380,0.42744,1696.32484,0.42754,1696.67586,0.42764,1697.02686,0.42774,1697.37784,0.42784,1697.72879,0.42794,1698.07973,0.42804,1698.43065,0.42814,1698.78154,0.42824,1699.13241,0.42834,1699.48327,0.42844,1699.83410,0.42854,1700.18491,0.42864,1700.53570,0.42874,1700.88647,0.42884,1701.23722,0.42894,1701.58795,0.42904,1701.93866,0.42914,1702.28934,0.42924,1702.64001,0.42934,1702.99065,0.42944,1703.34128,0.42954,1703.69188,0.42964,1704.04246,0.42974,1704.39302,0.42984,1704.74357,0.42994,1705.09409,0.43004,1705.44458,0.43014,1705.79506,0.43024,1706.14552,0.43034,1706.49596,0.43044,1706.84637,0.43054,1707.19677,0.43064,1707.54714,0.43074,1707.89749,0.43084,1708.24782,0.43094,1708.59814,0.43104,1708.94843,0.43114,1709.29870,0.43124,1709.64895,0.43134,1709.99917,0.43144,1710.34938,0.43154,1710.69957,0.43164,1711.04973,0.43174,1711.39988,0.43184,1711.75000,0.43194,1712.10010,0.43204,1712.45019,0.43214,1712.80025,0.43224,1713.15029,0.43234,1713.50031,0.43244,1713.85031,0.43254,1714.20029,0.43264,1714.55024,0.43274,1714.90018,0.43284,1715.25010,0.43294,1715.59999,0.43304,1715.94986,0.43314,1716.29972,0.43324,1716.64955,0.43334,1716.99936,0.43344,1717.34915,0.43354,1717.69892,0.43364,1718.04867,0.43374,1718.39840,0.43384,1718.74811,0.43394,1719.09779,0.43404,1719.44746,0.43414,1719.79711,0.43424,1720.14673,0.43434,1720.49633,0.43444,1720.84592,0.43454,1721.19548,0.43464,1721.54502,0.43474,1721.89454,0.43484,1722.24404,0.43494,1722.59352,0.43504,1722.94298,0.43514,1723.29241,0.43524,1723.64183,0.43534,1723.99123,0.43544,1724.34060,0.43554,1724.68995,0.43564,1725.03929,0.43574,1725.38860,0.43584,1725.73789,0.43594,1726.08716,0.43604,1726.43641,0.43614,1726.78564,0.43624,1727.13485,0.43634,1727.48404,0.43644,1727.83320,0.43654,1728.18235,0.43664,1728.53148,0.43674,1728.88058,0.43684,1729.22966,0.43694,1729.57873,0.43704,1729.92777,0.43714,1730.27679,0.43724,1730.62579,0.43734,1730.97477,0.43744,1731.32373,0.43754,1731.67267,0.43764,1732.02159,0.43774,1732.37048,0.43784,1732.71936,0.43794,1733.06821,0.43804,1733.41705,0.43814,1733.76586,0.43824,1734.11465,0.43834,1734.46343,0.43844,1734.81218,0.43854,1735.16091,0.43864,1735.50962,0.43874,1735.85831,0.43884,1736.20697,0.43894,1736.55562,0.43904,1736.90425,0.43914,1737.25285,0.43924,1737.60144,0.43934,1737.95000,0.43944,1738.29855,0.43954,1738.64707,0.43964,1738.99557,0.43974,1739.34405,0.43984,1739.69251,0.43994,1740.04095,0.44004,1740.38937,0.44014,1740.73777,0.44024,1741.08615,0.44034,1741.43451,0.44044,1741.78284,0.44054,1742.13116,0.44064,1742.47945,0.44074,1742.82773,0.44084,1743.17598,0.44094,1743.52421,0.44104,1743.87242,0.44114,1744.22061,0.44124,1744.56878,0.44134,1744.91693,0.44144,1745.26506,0.44154,1745.61317,0.44164,1745.96126,0.44174,1746.30932,0.44184,1746.65737,0.44194,1747.00539,0.44204,1747.35340,0.44214,1747.70138,0.44224,1748.04934,0.44234,1748.39729,0.44244,1748.74521,0.44254,1749.09311,0.44264,1749.44099,0.44274,1749.78885,0.44284,1750.13668,0.44294,1750.48450,0.44304,1750.83230,0.44314,1751.18007,0.44324,1751.52783,0.44334,1751.87556,0.44344,1752.22328,0.44354,1752.57097,0.44364,1752.91864,0.44374,1753.26629,0.44384,1753.61393,0.44394,1753.96154,0.44404,1754.30912,0.44414,1754.65669,0.44424,1755.00424,0.44434,1755.35177,0.44444,1755.69928,0.44454,1756.04676,0.44464,1756.39423,0.44474,1756.74167,0.44484,1757.08909,0.44494,1757.43650,0.44504,1757.78388,0.44514,1758.13124,0.44524,1758.47858,0.44534,1758.82590,0.44544,1759.17320,0.44554,1759.52048,0.44564,1759.86774,0.44574,1760.21498,0.44584,1760.56219,0.44594,1760.90939,0.44604,1761.25656,0.44614,1761.60372,0.44624,1761.95085,0.44634,1762.29796,0.44644,1762.64506,0.44654,1762.99213,0.44664,1763.33918,0.44674,1763.68621,0.44684,1764.03322,0.44694,1764.38021,0.44704,1764.72718,0.44714,1765.07412,0.44724,1765.42105,0.44734,1765.76796,0.44744,1766.11484,0.44754,1766.46171,0.44764,1766.80855,0.44774,1767.15537,0.44784,1767.50217,0.44794,1767.84896,0.44804,1768.19572,0.44814,1768.54246,0.44824,1768.88918,0.44834,1769.23588,0.44844,1769.58256,0.44854,1769.92921,0.44864,1770.27585,0.44874,1770.62247,0.44884,1770.96906,0.44894,1771.31564,0.44904,1771.66219,0.44914,1772.00873,0.44924,1772.35524,0.44934,1772.70173,0.44944,1773.04820,0.44954,1773.39465,0.44964,1773.74108,0.44974,1774.08749,0.44984,1774.43388,0.44994,1774.78025,0.45005,1775.12660,0.45015,1775.47292,0.45025,1775.81923,0.45035,1776.16552,0.45045,1776.51178,0.45055,1776.85802,0.45065,1777.20425,0.45075,1777.55045,0.45085,1777.89663,0.45095,1778.24279,0.45105,1778.58894,0.45115,1778.93506,0.45125,1779.28115,0.45135,1779.62723,0.45145,1779.97329,0.45155,1780.31933,0.45165,1780.66535,0.45175,1781.01134,0.45185,1781.35732,0.45195,1781.70327,0.45205,1782.04921,0.45215,1782.39512,0.45225,1782.74101,0.45235,1783.08689,0.45245,1783.43274,0.45255,1783.77857,0.45265,1784.12438,0.45275,1784.47017,0.45285,1784.81594,0.45295,1785.16169,0.45305,1785.50741,0.45315,1785.85312,0.45325,1786.19881,0.45335,1786.54447,0.45345,1786.89012,0.45355,1787.23574,0.45365,1787.58135,0.45375,1787.92693,0.45385,1788.27249,0.45395,1788.61803,0.45405,1788.96355,0.45415,1789.30905,0.45425,1789.65453,0.45435,1789.99999,0.45445,1790.34543,0.45455,1790.69085,0.45465,1791.03625,0.45475,1791.38163,0.45485,1791.72698,0.45495,1792.07232,0.45505,1792.41763,0.45515,1792.76293,0.45525,1793.10820,0.45535,1793.45345,0.45545,1793.79869,0.45555,1794.14390,0.45565,1794.48909,0.45575,1794.83426,0.45585,1795.17941,0.45595,1795.52454,0.45605,1795.86965,0.45615,1796.21473,0.45625,1796.55980,0.45635,1796.90485,0.45645,1797.24987,0.45655,1797.59488,0.45665,1797.93987,0.45675,1798.28483,0.45685,1798.62977,0.45695,1798.97470,0.45705,1799.31960,0.45715,1799.66448,0.45725,1800.00934,0.45735,1800.35418,0.45745,1800.69900,0.45755,1801.04380,0.45765,1801.38858,0.45775,1801.73334,0.45785,1802.07808,0.45795,1802.42279,0.45805,1802.76749,0.45815,1803.11217,0.45825,1803.45682,0.45835,1803.80146,0.45845,1804.14607,0.45855,1804.49066,0.45865,1804.83524,0.45875,1805.17979,0.45885,1805.52432,0.45895,1805.86883,0.45905,1806.21332,0.45915,1806.55779,0.45925,1806.90224,0.45935,1807.24667,0.45945,1807.59108,0.45955,1807.93546,0.45965,1808.27983,0.45975,1808.62418,0.45985,1808.96850,0.45995,1809.31281,0.46005,1809.65709,0.46015,1810.00135,0.46025,1810.34560,0.46035,1810.68982,0.46045,1811.03402,0.46055,1811.37820,0.46065,1811.72237,0.46075,1812.06651,0.46085,1812.41063,0.46095,1812.75472,0.46105,1813.09880,0.46115,1813.44286,0.46125,1813.78690,0.46135,1814.13092,0.46145,1814.47491,0.46155,1814.81889,0.46165,1815.16284,0.46175,1815.50678,0.46185,1815.85069,0.46195,1816.19458,0.46205,1816.53846,0.46215,1816.88231,0.46225,1817.22614,0.46235,1817.56995,0.46245,1817.91374,0.46255,1818.25751,0.46265,1818.60126,0.46275,1818.94499,0.46285,1819.28870,0.46295,1819.63239,0.46305,1819.97606,0.46315,1820.31970,0.46325,1820.66333,0.46335,1821.00693,0.46345,1821.35052,0.46355,1821.69408,0.46365,1822.03763,0.46375,1822.38115,0.46385,1822.72465,0.46395,1823.06814,0.46405,1823.41160,0.46415,1823.75504,0.46425,1824.09846,0.46435,1824.44186,0.46445,1824.78524,0.46455,1825.12860,0.46465,1825.47193,0.46475,1825.81525,0.46485,1826.15855,0.46495,1826.50183,0.46505,1826.84508,0.46515,1827.18832,0.46525,1827.53153,0.46535,1827.87473,0.46545,1828.21790,0.46555,1828.56106,0.46565,1828.90419,0.46575,1829.24730,0.46585,1829.59039,0.46595,1829.93346,0.46605,1830.27651,0.46615,1830.61955,0.46625,1830.96255,0.46635,1831.30554,0.46645,1831.64851,0.46655,1831.99146,0.46665,1832.33439,0.46675,1832.67730,0.46685,1833.02018,0.46695,1833.36305,0.46705,1833.70589,0.46715,1834.04872,0.46725,1834.39152,0.46735,1834.73431,0.46745,1835.07707,0.46755,1835.41981,0.46765,1835.76254,0.46775,1836.10524,0.46785,1836.44792,0.46795,1836.79058,0.46805,1837.13322,0.46815,1837.47584,0.46825,1837.81844,0.46835,1838.16102,0.46845,1838.50358,0.46855,1838.84611,0.46865,1839.18863,0.46875,1839.53113,0.46885,1839.87360,0.46895,1840.21606,0.46905,1840.55849,0.46915,1840.90091,0.46925,1841.24330,0.46935,1841.58568,0.46945,1841.92803,0.46955,1842.27036,0.46965,1842.61267,0.46975,1842.95497,0.46985,1843.29724,0.46995,1843.63949,0.47005,1843.98172,0.47015,1844.32393,0.47025,1844.66612,0.47035,1845.00828,0.47045,1845.35043,0.47055,1845.69256,0.47065,1846.03467,0.47075,1846.37675,0.47085,1846.71882,0.47095,1847.06087,0.47105,1847.40289,0.47115,1847.74490,0.47125,1848.08688,0.47135,1848.42884,0.47145,1848.77079,0.47155,1849.11271,0.47165,1849.45461,0.47175,1849.79649,0.47185,1850.13835,0.47195,1850.48020,0.47205,1850.82202,0.47215,1851.16382,0.47225,1851.50559,0.47235,1851.84735,0.47245,1852.18909,0.47255,1852.53081,0.47265,1852.87251,0.47275,1853.21418,0.47285,1853.55584,0.47295,1853.89748,0.47305,1854.23909,0.47315,1854.58069,0.47325,1854.92226,0.47335,1855.26382,0.47345,1855.60535,0.47355,1855.94686,0.47365,1856.28836,0.47375,1856.62983,0.47385,1856.97128,0.47395,1857.31271,0.47405,1857.65412,0.47415,1857.99551,0.47425,1858.33688,0.47435,1858.67823,0.47445,1859.01956,0.47455,1859.36087,0.47465,1859.70216,0.47475,1860.04342,0.47485,1860.38467,0.47495,1860.72590,0.47505,1861.06710,0.47515,1861.40829,0.47525,1861.74946,0.47535,1862.09060,0.47545,1862.43173,0.47555,1862.77283,0.47565,1863.11391,0.47575,1863.45498,0.47585,1863.79602,0.47595,1864.13704,0.47605,1864.47804,0.47615,1864.81902,0.47625,1865.15998,0.47635,1865.50092,0.47645,1865.84184,0.47655,1866.18274,0.47665,1866.52362,0.47675,1866.86448,0.47685,1867.20532,0.47695,1867.54614,0.47705,1867.88694,0.47715,1868.22771,0.47725,1868.56847,0.47735,1868.90921,0.47745,1869.24992,0.47755,1869.59062,0.47765,1869.93129,0.47775,1870.27195,0.47785,1870.61258,0.47795,1870.95319,0.47805,1871.29379,0.47815,1871.63436,0.47825,1871.97491,0.47835,1872.31544,0.47845,1872.65595,0.47855,1872.99644,0.47865,1873.33691,0.47875,1873.67736,0.47885,1874.01779,0.47895,1874.35820,0.47905,1874.69859,0.47915,1875.03896,0.47925,1875.37931,0.47935,1875.71964,0.47945,1876.05994,0.47955,1876.40023,0.47965,1876.74050,0.47975,1877.08074,0.47985,1877.42097,0.47995,1877.76117,0.48005,1878.10136,0.48015,1878.44152,0.48025,1878.78167,0.48035,1879.12179,0.48045,1879.46189,0.48055,1879.80198,0.48065,1880.14204,0.48075,1880.48208,0.48085,1880.82210,0.48095,1881.16210,0.48105,1881.50208,0.48115,1881.84204,0.48125,1882.18198,0.48135,1882.52190,0.48145,1882.86180,0.48155,1883.20168,0.48165,1883.54154,0.48175,1883.88138,0.48185,1884.22119,0.48195,1884.56099,0.48205,1884.90077,0.48215,1885.24052,0.48225,1885.58026,0.48235,1885.91997,0.48245,1886.25967,0.48255,1886.59934,0.48265,1886.93900,0.48275,1887.27863,0.48285,1887.61825,0.48295,1887.95784,0.48305,1888.29741,0.48315,1888.63697,0.48325,1888.97650,0.48335,1889.31601,0.48345,1889.65550,0.48355,1889.99497,0.48365,1890.33442,0.48375,1890.67385,0.48385,1891.01326,0.48395,1891.35265,0.48405,1891.69202,0.48415,1892.03137,0.48425,1892.37070,0.48435,1892.71001,0.48445,1893.04929,0.48455,1893.38856,0.48465,1893.72781,0.48475,1894.06703,0.48485,1894.40624,0.48495,1894.74543,0.48505,1895.08459,0.48515,1895.42374,0.48525,1895.76286,0.48535,1896.10196,0.48545,1896.44105,0.48555,1896.78011,0.48565,1897.11916,0.48575,1897.45818,0.48585,1897.79718,0.48595,1898.13616,0.48605,1898.47512,0.48615,1898.81407,0.48625,1899.15299,0.48635,1899.49189,0.48645,1899.83077,0.48655,1900.16963,0.48665,1900.50847,0.48675,1900.84729,0.48685,1901.18609,0.48695,1901.52487,0.48705,1901.86362,0.48715,1902.20236,0.48725,1902.54108,0.48735,1902.87978,0.48745,1903.21845,0.48755,1903.55711,0.48765,1903.89575,0.48775,1904.23436,0.48785,1904.57296,0.48795,1904.91153,0.48805,1905.25009,0.48815,1905.58862,0.48825,1905.92714,0.48835,1906.26563,0.48845,1906.60410,0.48855,1906.94256,0.48865,1907.28099,0.48875,1907.61940,0.48885,1907.95780,0.48895,1908.29617,0.48905,1908.63452,0.48915,1908.97285,0.48925,1909.31116,0.48935,1909.64945,0.48945,1909.98772,0.48955,1910.32597,0.48965,1910.66420,0.48975,1911.00241,0.48985,1911.34060,0.48995,1911.67877,0.49005,1912.01692,0.49015,1912.35505,0.49025,1912.69315,0.49035,1913.03124,0.49045,1913.36931,0.49055,1913.70735,0.49065,1914.04538,0.49075,1914.38339,0.49085,1914.72137,0.49095,1915.05934,0.49105,1915.39728,0.49115,1915.73521,0.49125,1916.07311,0.49135,1916.41100,0.49145,1916.74886,0.49155,1917.08671,0.49165,1917.42453,0.49175,1917.76233,0.49185,1918.10012,0.49195,1918.43788,0.49205,1918.77562,0.49215,1919.11334,0.49225,1919.45104,0.49235,1919.78873,0.49245,1920.12639,0.49255,1920.46403,0.49265,1920.80165,0.49275,1921.13925,0.49285,1921.47683,0.49295,1921.81439,0.49305,1922.15193,0.49315,1922.48945,0.49325,1922.82695,0.49335,1923.16442,0.49345,1923.50188,0.49355,1923.83932,0.49365,1924.17674,0.49375,1924.51413,0.49385,1924.85151,0.49395,1925.18887,0.49405,1925.52621,0.49415,1925.86352,0.49425,1926.20082,0.49435,1926.53809,0.49445,1926.87535,0.49455,1927.21258,0.49465,1927.54980,0.49475,1927.88699,0.49485,1928.22417,0.49495,1928.56132,0.49505,1928.89846,0.49515,1929.23557,0.49525,1929.57266,0.49535,1929.90973,0.49545,1930.24679,0.49555,1930.58382,0.49565,1930.92083,0.49575,1931.25782,0.49585,1931.59480,0.49595,1931.93175,0.49605,1932.26868,0.49615,1932.60559,0.49625,1932.94248,0.49635,1933.27935,0.49645,1933.61620,0.49655,1933.95303,0.49665,1934.28984,0.49675,1934.62663,0.49685,1934.96340,0.49695,1935.30015,0.49705,1935.63688,0.49715,1935.97358,0.49725,1936.31027,0.49735,1936.64694,0.49745,1936.98359,0.49755,1937.32021,0.49765,1937.65682,0.49775,1937.99341,0.49785,1938.32997,0.49795,1938.66652,0.49805,1939.00305,0.49815,1939.33955,0.49825,1939.67604,0.49835,1940.01250,0.49845,1940.34895,0.49855,1940.68537,0.49865,1941.02178,0.49875,1941.35816,0.49885,1941.69453,0.49895,1942.03087,0.49905,1942.36720,0.49915,1942.70350,0.49925,1943.03978,0.49935,1943.37605,0.49945,1943.71229,0.49955,1944.04851,0.49965,1944.38471,0.49975,1944.72090,0.49985,1945.05706,0.49995,1945.39320,0.50005,1945.72932,0.50015,1946.06542,0.50025,1946.40150,0.50035,1946.73756,0.50045,1947.07360,0.50055,1947.40962,0.50065,1947.74562,0.50075,1948.08160,0.50085,1948.41756,0.50095,1948.75350,0.50105,1949.08942,0.50115,1949.42532,0.50125,1949.76120,0.50135,1950.09706,0.50145,1950.43290,0.50155,1950.76872,0.50165,1951.10451,0.50175,1951.44029,0.50185,1951.77605,0.50195,1952.11179,0.50205,1952.44750,0.50215,1952.78320,0.50225,1953.11888,0.50235,1953.45454,0.50245,1953.79017,0.50255,1954.12579,0.50265,1954.46138,0.50275,1954.79696,0.50285,1955.13252,0.50295,1955.46805,0.50305,1955.80357,0.50315,1956.13906,0.50325,1956.47454,0.50335,1956.80999,0.50345,1957.14543,0.50355,1957.48084,0.50365,1957.81623,0.50375,1958.15161,0.50385,1958.48696,0.50395,1958.82229,0.50405,1959.15761,0.50415,1959.49290,0.50425,1959.82817,0.50435,1960.16343,0.50445,1960.49866,0.50455,1960.83387,0.50465,1961.16906,0.50475,1961.50424,0.50485,1961.83939,0.50495,1962.17452,0.50505,1962.50963,0.50515,1962.84472,0.50525,1963.17979,0.50535,1963.51485,0.50545,1963.84988,0.50555,1964.18489,0.50565,1964.51988,0.50575,1964.85485,0.50585,1965.18980,0.50595,1965.52473,0.50605,1965.85964,0.50615,1966.19453,0.50625,1966.52940,0.50635,1966.86425,0.50645,1967.19908,0.50655,1967.53389,0.50665,1967.86867,0.50675,1968.20344,0.50685,1968.53819,0.50695,1968.87292,0.50705,1969.20763,0.50715,1969.54232,0.50725,1969.87698,0.50735,1970.21163,0.50745,1970.54626,0.50755,1970.88087,0.50765,1971.21545,0.50775,1971.55002,0.50785,1971.88457,0.50795,1972.21910,0.50805,1972.55360,0.50815,1972.88809,0.50825,1973.22256,0.50835,1973.55700,0.50845,1973.89143,0.50855,1974.22583,0.50865,1974.56022,0.50875,1974.89459,0.50885,1975.22893,0.50895,1975.56326,0.50905,1975.89756,0.50915,1976.23185,0.50925,1976.56611,0.50935,1976.90036,0.50945,1977.23458,0.50955,1977.56879,0.50965,1977.90297,0.50975,1978.23714,0.50985,1978.57128,0.50995,1978.90541,0.51005,1979.23951,0.51015,1979.57359,0.51025,1979.90766,0.51035,1980.24170,0.51045,1980.57572,0.51055,1980.90973,0.51065,1981.24371,0.51075,1981.57767,0.51085,1981.91162,0.51095,1982.24554,0.51105,1982.57944,0.51115,1982.91333,0.51125,1983.24719,0.51135,1983.58103,0.51145,1983.91485,0.51155,1984.24866,0.51165,1984.58244,0.51175,1984.91620,0.51185,1985.24994,0.51195,1985.58366,0.51205,1985.91737,0.51215,1986.25105,0.51225,1986.58471,0.51235,1986.91835,0.51245,1987.25197,0.51255,1987.58557,0.51265,1987.91915,0.51275,1988.25272,0.51285,1988.58626,0.51295,1988.91978,0.51305,1989.25328,0.51315,1989.58676,0.51325,1989.92022,0.51335,1990.25366,0.51345,1990.58708,0.51355,1990.92048,0.51365,1991.25386,0.51375,1991.58722,0.51385,1991.92056,0.51395,1992.25388,0.51405,1992.58718,0.51415,1992.92046,0.51425,1993.25372,0.51435,1993.58696,0.51445,1993.92018,0.51455,1994.25338,0.51465,1994.58656,0.51475,1994.91972,0.51485,1995.25286,0.51495,1995.58597,0.51505,1995.91907,0.51515,1996.25215,0.51525,1996.58521,0.51535,1996.91825,0.51545,1997.25127,0.51555,1997.58427,0.51565,1997.91725,0.51575,1998.25020,0.51585,1998.58314,0.51595,1998.91606,0.51605,1999.24896,0.51615,1999.58184,0.51625,1999.91470,0.51635,2000.24753,0.51645,2000.58035,0.51655,2000.91315,0.51665,2001.24593,0.51675,2001.57869,0.51685,2001.91142,0.51695,2002.24414,0.51705,2002.57684,0.51715,2002.90952,0.51725,2003.24217,0.51735,2003.57481,0.51745,2003.90743,0.51755,2004.24003,0.51765,2004.57260,0.51775,2004.90516,0.51785,2005.23770,0.51795,2005.57022,0.51805,2005.90271,0.51815,2006.23519,0.51825,2006.56765,0.51835,2006.90008,0.51845,2007.23250,0.51855,2007.56490,0.51865,2007.89727,0.51875,2008.22963,0.51885,2008.56197,0.51895,2008.89428,0.51905,2009.22658,0.51915,2009.55886,0.51925,2009.89111,0.51935,2010.22335,0.51945,2010.55557,0.51955,2010.88776,0.51965,2011.21994,0.51975,2011.55210,0.51985,2011.88423,0.51995,2012.21635,0.52005,2012.54845,0.52015,2012.88052,0.52025,2013.21258,0.52035,2013.54462,0.52045,2013.87663,0.52055,2014.20863,0.52065,2014.54060,0.52075,2014.87256,0.52085,2015.20450,0.52095,2015.53641,0.52105,2015.86831,0.52115,2016.20018,0.52125,2016.53204,0.52135,2016.86388,0.52145,2017.19569,0.52155,2017.52749,0.52165,2017.85927,0.52175,2018.19102,0.52185,2018.52276,0.52195,2018.85447,0.52205,2019.18617,0.52215,2019.51785,0.52225,2019.84950,0.52235,2020.18114,0.52245,2020.51275,0.52255,2020.84435,0.52265,2021.17593,0.52275,2021.50748,0.52285,2021.83902,0.52295,2022.17053,0.52305,2022.50203,0.52315,2022.83351,0.52325,2023.16496,0.52335,2023.49640,0.52345,2023.82781,0.52355,2024.15921,0.52365,2024.49059,0.52375,2024.82194,0.52385,2025.15328,0.52395,2025.48459,0.52405,2025.81589,0.52415,2026.14717,0.52425,2026.47842,0.52435,2026.80966,0.52445,2027.14087,0.52455,2027.47207,0.52465,2027.80325,0.52475,2028.13440,0.52485,2028.46554,0.52495,2028.79665,0.52505,2029.12775,0.52515,2029.45883,0.52525,2029.78988,0.52535,2030.12092,0.52545,2030.45194,0.52555,2030.78293,0.52565,2031.11391,0.52575,2031.44486,0.52585,2031.77580,0.52595,2032.10672,0.52605,2032.43761,0.52615,2032.76849,0.52625,2033.09935,0.52635,2033.43018,0.52645,2033.76100,0.52655,2034.09180,0.52665,2034.42257,0.52675,2034.75333,0.52685,2035.08407,0.52695,2035.41478,0.52705,2035.74548,0.52715,2036.07616,0.52725,2036.40681,0.52735,2036.73745,0.52745,2037.06807,0.52755,2037.39867,0.52765,2037.72924,0.52775,2038.05980,0.52785,2038.39034,0.52795,2038.72085,0.52805,2039.05135,0.52815,2039.38183,0.52825,2039.71229,0.52835,2040.04272,0.52845,2040.37314,0.52855,2040.70354,0.52865,2041.03392,0.52875,2041.36427,0.52885,2041.69461,0.52895,2042.02493,0.52905,2042.35523,0.52915,2042.68551,0.52925,2043.01576,0.52935,2043.34600,0.52945,2043.67622,0.52955,2044.00642,0.52965,2044.33660,0.52975,2044.66676,0.52985,2044.99689,0.52995,2045.32701,0.53005,2045.65711,0.53015,2045.98719,0.53025,2046.31725,0.53035,2046.64729,0.53045,2046.97731,0.53055,2047.30730,0.53065,2047.63728,0.53075,2047.96724,0.53085,2048.29718,0.53095,2048.62710,0.53105,2048.95700,0.53115,2049.28688,0.53125,2049.61674,0.53135,2049.94658,0.53145,2050.27640,0.53155,2050.60620,0.53165,2050.93598,0.53175,2051.26574,0.53185,2051.59548,0.53195,2051.92520,0.53205,2052.25490,0.53215,2052.58458,0.53225,2052.91424,0.53235,2053.24388,0.53245,2053.57350,0.53255,2053.90310,0.53265,2054.23268,0.53275,2054.56224,0.53285,2054.89178,0.53295,2055.22130,0.53305,2055.55080,0.53315,2055.88029,0.53325,2056.20975,0.53335,2056.53919,0.53345,2056.86861,0.53355,2057.19801,0.53365,2057.52739,0.53375,2057.85676,0.53385,2058.18610,0.53395,2058.51542,0.53405,2058.84472,0.53415,2059.17400,0.53425,2059.50327,0.53435,2059.83251,0.53445,2060.16173,0.53455,2060.49093,0.53465,2060.82012,0.53475,2061.14928,0.53485,2061.47842,0.53495,2061.80755,0.53505,2062.13665,0.53515,2062.46573,0.53525,2062.79480,0.53535,2063.12384,0.53545,2063.45287,0.53555,2063.78187,0.53565,2064.11085,0.53575,2064.43982,0.53585,2064.76876,0.53595,2065.09769,0.53605,2065.42659,0.53615,2065.75548,0.53625,2066.08434,0.53635,2066.41319,0.53645,2066.74201,0.53655,2067.07082,0.53665,2067.39960,0.53675,2067.72837,0.53685,2068.05711,0.53695,2068.38584,0.53705,2068.71455,0.53715,2069.04323,0.53725,2069.37190,0.53735,2069.70054,0.53745,2070.02917,0.53755,2070.35778,0.53765,2070.68636,0.53775,2071.01493,0.53785,2071.34348,0.53795,2071.67201,0.53805,2072.00051,0.53815,2072.32900,0.53825,2072.65747,0.53835,2072.98592,0.53845,2073.31434,0.53855,2073.64275,0.53865,2073.97114,0.53875,2074.29951,0.53885,2074.62786,0.53895,2074.95619,0.53905,2075.28450,0.53915,2075.61279,0.53925,2075.94106,0.53935,2076.26931,0.53945,2076.59753,0.53955,2076.92574,0.53965,2077.25393,0.53975,2077.58211,0.53985,2077.91026,0.53995,2078.23839,0.54005,2078.56650,0.54015,2078.89459,0.54025,2079.22266,0.54035,2079.55071,0.54045,2079.87874,0.54055,2080.20675,0.54065,2080.53474,0.54075,2080.86272,0.54085,2081.19067,0.54095,2081.51860,0.54105,2081.84651,0.54115,2082.17441,0.54125,2082.50228,0.54135,2082.83013,0.54145,2083.15797,0.54155,2083.48578,0.54165,2083.81357,0.54175,2084.14135,0.54185,2084.46910,0.54195,2084.79684,0.54205,2085.12455,0.54215,2085.45224,0.54225,2085.77992,0.54235,2086.10757,0.54245,2086.43521,0.54255,2086.76282,0.54265,2087.09042,0.54275,2087.41800,0.54285,2087.74555,0.54295,2088.07309,0.54305,2088.40060,0.54315,2088.72810,0.54325,2089.05558,0.54335,2089.38303,0.54345,2089.71047,0.54355,2090.03789,0.54365,2090.36529,0.54375,2090.69266,0.54385,2091.02002,0.54395,2091.34736,0.54405,2091.67468,0.54415,2092.00198,0.54425,2092.32926,0.54435,2092.65652,0.54445,2092.98376,0.54455,2093.31098,0.54465,2093.63817,0.54475,2093.96535,0.54485,2094.29252,0.54495,2094.61966,0.54505,2094.94678,0.54515,2095.27388,0.54525,2095.60096,0.54535,2095.92802,0.54545,2096.25506,0.54555,2096.58208,0.54565,2096.90908,0.54575,2097.23607,0.54585,2097.56303,0.54595,2097.88997,0.54605,2098.21690,0.54615,2098.54380,0.54625,2098.87068,0.54635,2099.19755,0.54645,2099.52439,0.54655,2099.85121,0.54665,2100.17802,0.54675,2100.50480,0.54685,2100.83157,0.54695,2101.15831,0.54705,2101.48504,0.54715,2101.81174,0.54725,2102.13843,0.54735,2102.46510,0.54745,2102.79174,0.54755,2103.11837,0.54765,2103.44497,0.54775,2103.77156,0.54785,2104.09813,0.54795,2104.42468,0.54805,2104.75120,0.54815,2105.07771,0.54825,2105.40420,0.54835,2105.73067,0.54845,2106.05712,0.54855,2106.38355,0.54865,2106.70996,0.54875,2107.03635,0.54885,2107.36272,0.54895,2107.68907,0.54905,2108.01540,0.54915,2108.34171,0.54925,2108.66800,0.54935,2108.99427,0.54945,2109.32052,0.54955,2109.64675,0.54965,2109.97297,0.54975,2110.29916,0.54985,2110.62533,0.54995,2110.95148,0.55006,2111.27762,0.55016,2111.60373,0.55026,2111.92982,0.55036,2112.25590,0.55046,2112.58195,0.55056,2112.90799,0.55066,2113.23400,0.55076,2113.56000,0.55086,2113.88597,0.55096,2114.21193,0.55106,2114.53786,0.55116,2114.86378,0.55126,2115.18968,0.55136,2115.51555,0.55146,2115.84141,0.55156,2116.16725,0.55166,2116.49306,0.55176,2116.81886,0.55186,2117.14464,0.55196,2117.47040,0.55206,2117.79614,0.55216,2118.12186,0.55226,2118.44756,0.55236,2118.77324,0.55246,2119.09890,0.55256,2119.42454,0.55266,2119.75016,0.55276,2120.07576,0.55286,2120.40134,0.55296,2120.72690,0.55306,2121.05244,0.55316,2121.37797,0.55326,2121.70347,0.55336,2122.02895,0.55346,2122.35441,0.55356,2122.67986,0.55366,2123.00528,0.55376,2123.33068,0.55386,2123.65607,0.55396,2123.98143,0.55406,2124.30678,0.55416,2124.63210,0.55426,2124.95741,0.55436,2125.28270,0.55446,2125.60796,0.55456,2125.93321,0.55466,2126.25844,0.55476,2126.58364,0.55486,2126.90883,0.55496,2127.23400,0.55506,2127.55915,0.55516,2127.88427,0.55526,2128.20938,0.55536,2128.53447,0.55546,2128.85954,0.55556,2129.18459,0.55566,2129.50962,0.55576,2129.83463,0.55586,2130.15962,0.55596,2130.48459,0.55606,2130.80955,0.55616,2131.13448,0.55626,2131.45939,0.55636,2131.78428,0.55646,2132.10916,0.55656,2132.43401,0.55666,2132.75884,0.55676,2133.08366,0.55686,2133.40845,0.55696,2133.73323,0.55706,2134.05798,0.55716,2134.38272,0.55726,2134.70743,0.55736,2135.03213,0.55746,2135.35680,0.55756,2135.68146,0.55766,2136.00610,0.55776,2136.33071,0.55786,2136.65531,0.55796,2136.97989,0.55806,2137.30445,0.55816,2137.62899,0.55826,2137.95351,0.55836,2138.27801,0.55846,2138.60249,0.55856,2138.92695,0.55866,2139.25139,0.55876,2139.57581,0.55886,2139.90021,0.55896,2140.22459,0.55906,2140.54895,0.55916,2140.87330,0.55926,2141.19762,0.55936,2141.52192,0.55946,2141.84621,0.55956,2142.17047,0.55966,2142.49471,0.55976,2142.81894,0.55986,2143.14314,0.55996,2143.46733,0.56006,2143.79150,0.56016,2144.11564,0.56026,2144.43977,0.56036,2144.76387,0.56046,2145.08796,0.56056,2145.41203,0.56066,2145.73608,0.56076,2146.06011,0.56086,2146.38412,0.56096,2146.70810,0.56106,2147.03207,0.56116,2147.35602,0.56126,2147.67996,0.56136,2148.00387,0.56146,2148.32776,0.56156,2148.65163,0.56166,2148.97548,0.56176,2149.29931,0.56186,2149.62313,0.56196,2149.94692,0.56206,2150.27069,0.56216,2150.59445,0.56226,2150.91818,0.56236,2151.24190,0.56246,2151.56559,0.56256,2151.88927,0.56266,2152.21292,0.56276,2152.53656,0.56286,2152.86018,0.56296,2153.18377,0.56306,2153.50735,0.56316,2153.83091,0.56326,2154.15445,0.56336,2154.47797,0.56346,2154.80147,0.56356,2155.12495,0.56366,2155.44841,0.56376,2155.77185,0.56386,2156.09527,0.56396,2156.41867,0.56406,2156.74205,0.56416,2157.06541,0.56426,2157.38876,0.56436,2157.71208,0.56446,2158.03538,0.56456,2158.35867,0.56466,2158.68193,0.56476,2159.00518,0.56486,2159.32840,0.56496,2159.65161,0.56506,2159.97479,0.56516,2160.29796,0.56526,2160.62111,0.56536,2160.94423,0.56546,2161.26734,0.56556,2161.59043,0.56566,2161.91350,0.56576,2162.23655,0.56586,2162.55958,0.56596,2162.88259,0.56606,2163.20558,0.56616,2163.52855,0.56626,2163.85150,0.56636,2164.17443,0.56646,2164.49735,0.56656,2164.82024,0.56666,2165.14311,0.56676,2165.46596,0.56686,2165.78880,0.56696,2166.11161,0.56706,2166.43441,0.56716,2166.75718,0.56726,2167.07994,0.56736,2167.40268,0.56746,2167.72539,0.56756,2168.04809,0.56766,2168.37077,0.56776,2168.69343,0.56786,2169.01606,0.56796,2169.33868,0.56806,2169.66128,0.56816,2169.98386,0.56826,2170.30642,0.56836,2170.62896,0.56846,2170.95149,0.56856,2171.27399,0.56866,2171.59647,0.56876,2171.91893,0.56886,2172.24138,0.56896,2172.56380,0.56906,2172.88620,0.56916,2173.20859,0.56926,2173.53095,0.56936,2173.85330,0.56946,2174.17563,0.56956,2174.49793,0.56966,2174.82022,0.56976,2175.14249,0.56986,2175.46474,0.56996,2175.78696,0.57006,2176.10917,0.57016,2176.43136,0.57026,2176.75353,0.57036,2177.07568,0.57046,2177.39781,0.57056,2177.71993,0.57066,2178.04202,0.57076,2178.36409,0.57086,2178.68614,0.57096,2179.00818,0.57106,2179.33019,0.57116,2179.65218,0.57126,2179.97416,0.57136,2180.29611,0.57146,2180.61805,0.57156,2180.93997,0.57166,2181.26186,0.57176,2181.58374,0.57186,2181.90560,0.57196,2182.22744,0.57206,2182.54926,0.57216,2182.87106,0.57226,2183.19284,0.57236,2183.51460,0.57246,2183.83634,0.57256,2184.15806,0.57266,2184.47976,0.57276,2184.80144,0.57286,2185.12311,0.57296,2185.44475,0.57306,2185.76637,0.57316,2186.08798,0.57326,2186.40956,0.57336,2186.73113,0.57346,2187.05267,0.57356,2187.37420,0.57366,2187.69571,0.57376,2188.01720,0.57386,2188.33866,0.57396,2188.66011,0.57406,2188.98154,0.57416,2189.30295,0.57426,2189.62434,0.57436,2189.94571,0.57446,2190.26706,0.57456,2190.58840,0.57466,2190.90971,0.57476,2191.23100,0.57486,2191.55228,0.57496,2191.87353,0.57506,2192.19476,0.57516,2192.51598,0.57526,2192.83718,0.57536,2193.15835,0.57546,2193.47951,0.57556,2193.80065,0.57566,2194.12176,0.57576,2194.44286,0.57586,2194.76394,0.57596,2195.08500,0.57606,2195.40604,0.57616,2195.72706,0.57626,2196.04806,0.57636,2196.36904,0.57646,2196.69001,0.57656,2197.01095,0.57666,2197.33187,0.57676,2197.65278,0.57686,2197.97366,0.57696,2198.29452,0.57706,2198.61537,0.57716,2198.93620,0.57726,2199.25700,0.57736,2199.57779,0.57746,2199.89856,0.57756,2200.21931,0.57766,2200.54004,0.57776,2200.86074,0.57786,2201.18143,0.57796,2201.50211,0.57806,2201.82276,0.57816,2202.14339,0.57826,2202.46400,0.57836,2202.78459,0.57846,2203.10517,0.57856,2203.42572,0.57866,2203.74626,0.57876,2204.06677,0.57886,2204.38727,0.57896,2204.70774,0.57906,2205.02820,0.57916,2205.34864,0.57926,2205.66906,0.57936,2205.98945,0.57946,2206.30983,0.57956,2206.63019,0.57966,2206.95053,0.57976,2207.27085,0.57986,2207.59116,0.57996,2207.91144,0.58006,2208.23170,0.58016,2208.55194,0.58026,2208.87217,0.58036,2209.19237,0.58046,2209.51256,0.58056,2209.83272,0.58066,2210.15287,0.58076,2210.47300,0.58086,2210.79311,0.58096,2211.11319,0.58106,2211.43326,0.58116,2211.75331,0.58126,2212.07334,0.58136,2212.39335,0.58146,2212.71334,0.58156,2213.03332,0.58166,2213.35327,0.58176,2213.67320,0.58186,2213.99311,0.58196,2214.31301,0.58206,2214.63288,0.58216,2214.95274,0.58226,2215.27257,0.58236,2215.59239,0.58246,2215.91219,0.58256,2216.23197,0.58266,2216.55172,0.58276,2216.87146,0.58286,2217.19118,0.58296,2217.51088,0.58306,2217.83057,0.58316,2218.15023,0.58326,2218.46987,0.58336,2218.78949,0.58346,2219.10910,0.58356,2219.42868,0.58366,2219.74824,0.58376,2220.06779,0.58386,2220.38732,0.58396,2220.70682,0.58406,2221.02631,0.58416,2221.34578,0.58426,2221.66523,0.58436,2221.98466,0.58446,2222.30407,0.58456,2222.62346,0.58466,2222.94283,0.58476,2223.26218,0.58486,2223.58151,0.58496,2223.90082,0.58506,2224.22012,0.58516,2224.53939,0.58526,2224.85865,0.58536,2225.17788,0.58546,2225.49710,0.58556,2225.81630,0.58566,2226.13547,0.58576,2226.45463,0.58586,2226.77377,0.58596,2227.09289,0.58606,2227.41199,0.58616,2227.73107,0.58626,2228.05013,0.58636,2228.36917,0.58646,2228.68820,0.58656,2229.00720,0.58666,2229.32619,0.58676,2229.64515,0.58686,2229.96410,0.58696,2230.28302,0.58706,2230.60193,0.58716,2230.92082,0.58726,2231.23968,0.58736,2231.55853,0.58746,2231.87736,0.58756,2232.19617,0.58766,2232.51496,0.58776,2232.83373,0.58786,2233.15249,0.58796,2233.47122,0.58806,2233.78993,0.58816,2234.10863,0.58826,2234.42730,0.58836,2234.74596,0.58846,2235.06459,0.58856,2235.38321,0.58866,2235.70181,0.58876,2236.02039,0.58886,2236.33895,0.58896,2236.65749,0.58906,2236.97601,0.58916,2237.29451,0.58926,2237.61299,0.58936,2237.93145,0.58946,2238.24989,0.58956,2238.56832,0.58966,2238.88672,0.58976,2239.20511,0.58986,2239.52347,0.58996,2239.84182,0.59006,2240.16015,0.59016,2240.47846,0.59026,2240.79674,0.59036,2241.11501,0.59046,2241.43326,0.59056,2241.75149,0.59066,2242.06971,0.59076,2242.38790,0.59086,2242.70607,0.59096,2243.02422,0.59106,2243.34236,0.59116,2243.66047,0.59126,2243.97857,0.59136,2244.29665,0.59146,2244.61470,0.59156,2244.93274,0.59166,2245.25076,0.59176,2245.56876,0.59186,2245.88674,0.59196,2246.20470,0.59206,2246.52264,0.59216,2246.84056,0.59226,2247.15847,0.59236,2247.47635,0.59246,2247.79421,0.59256,2248.11206,0.59266,2248.42988,0.59276,2248.74769,0.59286,2249.06548,0.59296,2249.38325,0.59306,2249.70100,0.59316,2250.01872,0.59326,2250.33643,0.59336,2250.65413,0.59346,2250.97180,0.59356,2251.28945,0.59366,2251.60708,0.59376,2251.92470,0.59386,2252.24229,0.59396,2252.55987,0.59406,2252.87742,0.59416,2253.19496,0.59426,2253.51248,0.59436,2253.82998,0.59446,2254.14745,0.59456,2254.46491,0.59466,2254.78235,0.59476,2255.09978,0.59486,2255.41718,0.59496,2255.73456,0.59506,2256.05192,0.59516,2256.36927,0.59526,2256.68659,0.59536,2257.00390,0.59546,2257.32119,0.59556,2257.63845,0.59566,2257.95570,0.59576,2258.27293,0.59586,2258.59014,0.59596,2258.90733,0.59606,2259.22450,0.59616,2259.54165,0.59626,2259.85879,0.59636,2260.17590,0.59646,2260.49299,0.59656,2260.81007,0.59666,2261.12712,0.59676,2261.44416,0.59686,2261.76118,0.59696,2262.07818,0.59706,2262.39516,0.59716,2262.71212,0.59726,2263.02906,0.59736,2263.34598,0.59746,2263.66288,0.59756,2263.97976,0.59766,2264.29662,0.59776,2264.61347,0.59786,2264.93029,0.59796,2265.24710,0.59806,2265.56389,0.59816,2265.88065,0.59826,2266.19740,0.59836,2266.51413,0.59846,2266.83084,0.59856,2267.14753,0.59866,2267.46420,0.59876,2267.78086,0.59886,2268.09749,0.59896,2268.41410,0.59906,2268.73070,0.59916,2269.04727,0.59926,2269.36383,0.59936,2269.68036,0.59946,2269.99688,0.59956,2270.31338,0.59966,2270.62986,0.59976,2270.94632,0.59986,2271.26276,0.59996,2271.57918,0.60006,2271.89559,0.60016,2272.21197,0.60026,2272.52833,0.60036,2272.84468,0.60046,2273.16100,0.60056,2273.47731,0.60066,2273.79360,0.60076,2274.10987,0.60086,2274.42612,0.60096,2274.74235,0.60106,2275.05856,0.60116,2275.37475,0.60126,2275.69092,0.60136,2276.00707,0.60146,2276.32321,0.60156,2276.63932,0.60166,2276.95542,0.60176,2277.27149,0.60186,2277.58755,0.60196,2277.90359,0.60206,2278.21961,0.60216,2278.53561,0.60226,2278.85159,0.60236,2279.16755,0.60246,2279.48349,0.60256,2279.79942,0.60266,2280.11532,0.60276,2280.43120,0.60286,2280.74707,0.60296,2281.06292,0.60306,2281.37874,0.60316,2281.69455,0.60326,2282.01034,0.60336,2282.32611,0.60346,2282.64186,0.60356,2282.95759,0.60366,2283.27331,0.60376,2283.58900,0.60386,2283.90467,0.60396,2284.22033,0.60406,2284.53596,0.60416,2284.85158,0.60426,2285.16718,0.60436,2285.48276,0.60446,2285.79831,0.60456,2286.11385,0.60466,2286.42938,0.60476,2286.74488,0.60486,2287.06036,0.60496,2287.37582,0.60506,2287.69127,0.60516,2288.00669,0.60526,2288.32210,0.60536,2288.63749,0.60546,2288.95285,0.60556,2289.26820,0.60566,2289.58353,0.60576,2289.89884,0.60586,2290.21413,0.60596,2290.52941,0.60606,2290.84466,0.60616,2291.15989,0.60626,2291.47511,0.60636,2291.79030,0.60646,2292.10548,0.60656,2292.42064,0.60666,2292.73577,0.60676,2293.05089,0.60686,2293.36599,0.60696,2293.68107,0.60706,2293.99614,0.60716,2294.31118,0.60726,2294.62620,0.60736,2294.94121,0.60746,2295.25619,0.60756,2295.57116,0.60766,2295.88611,0.60776,2296.20103,0.60786,2296.51594,0.60796,2296.83083,0.60806,2297.14570,0.60816,2297.46055,0.60826,2297.77539,0.60836,2298.09020,0.60846,2298.40499,0.60856,2298.71977,0.60866,2299.03453,0.60876,2299.34926,0.60886,2299.66398,0.60896,2299.97868,0.60906,2300.29336,0.60916,2300.60802,0.60926,2300.92266,0.60936,2301.23728,0.60946,2301.55189,0.60956,2301.86647,0.60966,2302.18104,0.60976,2302.49558,0.60986,2302.81011,0.60996,2303.12462,0.61006,2303.43911,0.61016,2303.75358,0.61026,2304.06803,0.61036,2304.38246,0.61046,2304.69687,0.61056,2305.01126,0.61066,2305.32564,0.61076,2305.63999,0.61086,2305.95433,0.61096,2306.26865,0.61106,2306.58294,0.61116,2306.89722,0.61126,2307.21148,0.61136,2307.52572,0.61146,2307.83995,0.61156,2308.15415,0.61166,2308.46833,0.61176,2308.78250,0.61186,2309.09664,0.61196,2309.41077,0.61206,2309.72488,0.61216,2310.03896,0.61226,2310.35303,0.61236,2310.66708,0.61246,2310.98111,0.61256,2311.29513,0.61266,2311.60912,0.61276,2311.92309,0.61286,2312.23705,0.61296,2312.55098,0.61306,2312.86490,0.61316,2313.17880,0.61326,2313.49268,0.61336,2313.80654,0.61346,2314.12038,0.61356,2314.43420,0.61366,2314.74800,0.61376,2315.06179,0.61386,2315.37555,0.61396,2315.68930,0.61406,2316.00302,0.61416,2316.31673,0.61426,2316.63042,0.61436,2316.94409,0.61446,2317.25774,0.61456,2317.57137,0.61466,2317.88498,0.61476,2318.19857,0.61486,2318.51215,0.61496,2318.82570,0.61506,2319.13924,0.61516,2319.45276,0.61526,2319.76625,0.61536,2320.07973,0.61546,2320.39319,0.61556,2320.70663,0.61566,2321.02006,0.61576,2321.33346,0.61586,2321.64684,0.61596,2321.96021,0.61606,2322.27355,0.61616,2322.58688,0.61626,2322.90019,0.61636,2323.21348,0.61646,2323.52675,0.61656,2323.84000,0.61666,2324.15323,0.61676,2324.46644,0.61686,2324.77963,0.61696,2325.09281,0.61706,2325.40596,0.61716,2325.71910,0.61726,2326.03222,0.61736,2326.34532,0.61746,2326.65840,0.61756,2326.97146,0.61766,2327.28450,0.61776,2327.59752,0.61786,2327.91053,0.61796,2328.22351,0.61806,2328.53648,0.61816,2328.84942,0.61826,2329.16235,0.61836,2329.47526,0.61846,2329.78815,0.61856,2330.10102,0.61866,2330.41387,0.61876,2330.72670,0.61886,2331.03952,0.61896,2331.35231,0.61906,2331.66509,0.61916,2331.97785,0.61926,2332.29058,0.61936,2332.60330,0.61946,2332.91600,0.61956,2333.22868,0.61966,2333.54135,0.61976,2333.85399,0.61986,2334.16661,0.61996,2334.47922,0.62006,2334.79180,0.62016,2335.10437,0.62026,2335.41692,0.62036,2335.72945,0.62046,2336.04196,0.62056,2336.35445,0.62066,2336.66692,0.62076,2336.97937,0.62086,2337.29181,0.62096,2337.60422,0.62106,2337.91662,0.62116,2338.22900,0.62126,2338.54136,0.62136,2338.85370,0.62146,2339.16602,0.62156,2339.47832,0.62166,2339.79060,0.62176,2340.10286,0.62186,2340.41511,0.62196,2340.72733,0.62206,2341.03954,0.62216,2341.35173,0.62226,2341.66390,0.62236,2341.97605,0.62246,2342.28818,0.62256,2342.60029,0.62266,2342.91238,0.62276,2343.22446,0.62286,2343.53651,0.62296,2343.84855,0.62306,2344.16057,0.62316,2344.47257,0.62326,2344.78455,0.62336,2345.09651,0.62346,2345.40845,0.62356,2345.72037,0.62366,2346.03227,0.62376,2346.34416,0.62386,2346.65603,0.62396,2346.96787,0.62406,2347.27970,0.62416,2347.59151,0.62426,2347.90330,0.62436,2348.21507,0.62446,2348.52682,0.62456,2348.83856,0.62466,2349.15027,0.62476,2349.46197,0.62486,2349.77364,0.62496,2350.08530,0.62506,2350.39694,0.62516,2350.70856,0.62526,2351.02016,0.62536,2351.33174,0.62546,2351.64331,0.62556,2351.95485,0.62566,2352.26638,0.62576,2352.57788,0.62586,2352.88937,0.62596,2353.20084,0.62606,2353.51229,0.62616,2353.82372,0.62626,2354.13513,0.62636,2354.44652,0.62646,2354.75790,0.62656,2355.06925,0.62666,2355.38059,0.62676,2355.69191,0.62686,2356.00321,0.62696,2356.31448,0.62706,2356.62575,0.62716,2356.93699,0.62726,2357.24821,0.62736,2357.55941,0.62746,2357.87060,0.62756,2358.18177,0.62766,2358.49291,0.62776,2358.80404,0.62786,2359.11515,0.62796,2359.42624,0.62806,2359.73731,0.62816,2360.04837,0.62826,2360.35940,0.62836,2360.67042,0.62846,2360.98141,0.62856,2361.29239,0.62866,2361.60335,0.62876,2361.91429,0.62886,2362.22521,0.62896,2362.53611,0.62906,2362.84699,0.62916,2363.15786,0.62926,2363.46870,0.62936,2363.77953,0.62946,2364.09034,0.62956,2364.40113,0.62966,2364.71190,0.62976,2365.02265,0.62986,2365.33338,0.62996,2365.64409,0.63006,2365.95479,0.63016,2366.26546,0.63026,2366.57612,0.63036,2366.88676,0.63046,2367.19738,0.63056,2367.50798,0.63066,2367.81856,0.63076,2368.12912,0.63086,2368.43966,0.63096,2368.75019,0.63106,2369.06069,0.63116,2369.37118,0.63126,2369.68165,0.63136,2369.99210,0.63146,2370.30253,0.63156,2370.61294,0.63166,2370.92333,0.63176,2371.23371,0.63186,2371.54406,0.63196,2371.85440,0.63206,2372.16472,0.63216,2372.47501,0.63226,2372.78529,0.63236,2373.09555,0.63246,2373.40580,0.63256,2373.71602,0.63266,2374.02622,0.63276,2374.33641,0.63286,2374.64658,0.63296,2374.95672,0.63306,2375.26685,0.63316,2375.57696,0.63326,2375.88706,0.63336,2376.19713,0.63346,2376.50718,0.63356,2376.81722,0.63366,2377.12723,0.63376,2377.43723,0.63386,2377.74721,0.63396,2378.05717,0.63406,2378.36711,0.63416,2378.67703,0.63426,2378.98693,0.63436,2379.29682,0.63446,2379.60668,0.63456,2379.91653,0.63466,2380.22636,0.63476,2380.53617,0.63486,2380.84596,0.63496,2381.15573,0.63506,2381.46548,0.63516,2381.77522,0.63526,2382.08493,0.63536,2382.39463,0.63546,2382.70431,0.63556,2383.01397,0.63566,2383.32361,0.63576,2383.63323,0.63586,2383.94283,0.63596,2384.25241,0.63606,2384.56198,0.63616,2384.87152,0.63626,2385.18105,0.63636,2385.49056,0.63646,2385.80005,0.63656,2386.10952,0.63666,2386.41897,0.63676,2386.72840,0.63686,2387.03782,0.63696,2387.34721,0.63706,2387.65659,0.63716,2387.96595,0.63726,2388.27529,0.63736,2388.58461,0.63746,2388.89391,0.63756,2389.20319,0.63766,2389.51246,0.63776,2389.82170,0.63786,2390.13093,0.63796,2390.44014,0.63806,2390.74933,0.63816,2391.05850,0.63826,2391.36765,0.63836,2391.67678,0.63846,2391.98590,0.63856,2392.29499,0.63866,2392.60407,0.63876,2392.91313,0.63886,2393.22217,0.63896,2393.53119,0.63906,2393.84019,0.63916,2394.14917,0.63926,2394.45813,0.63936,2394.76708,0.63946,2395.07601,0.63956,2395.38491,0.63966,2395.69380,0.63976,2396.00267,0.63986,2396.31152,0.63996,2396.62036,0.64006,2396.92917,0.64016,2397.23797,0.64026,2397.54674,0.64036,2397.85550,0.64046,2398.16424,0.64056,2398.47296,0.64066,2398.78166,0.64076,2399.09034,0.64086,2399.39901,0.64096,2399.70765,0.64106,2400.01628,0.64116,2400.32489,0.64126,2400.63348,0.64136,2400.94205,0.64146,2401.25060,0.64156,2401.55913,0.64166,2401.86764,0.64176,2402.17614,0.64186,2402.48462,0.64196,2402.79308,0.64206,2403.10151,0.64216,2403.40993,0.64226,2403.71834,0.64236,2404.02672,0.64246,2404.33508,0.64256,2404.64343,0.64266,2404.95176,0.64276,2405.26006,0.64286,2405.56835,0.64296,2405.87662,0.64306,2406.18488,0.64316,2406.49311,0.64326,2406.80132,0.64336,2407.10952,0.64346,2407.41770,0.64356,2407.72586,0.64366,2408.03400,0.64376,2408.34212,0.64386,2408.65022,0.64396,2408.95830,0.64406,2409.26637,0.64416,2409.57441,0.64426,2409.88244,0.64436,2410.19045,0.64446,2410.49844,0.64456,2410.80641,0.64466,2411.11436,0.64476,2411.42230,0.64486,2411.73021,0.64496,2412.03811,0.64506,2412.34599,0.64516,2412.65385,0.64526,2412.96169,0.64536,2413.26951,0.64546,2413.57731,0.64556,2413.88510,0.64566,2414.19286,0.64576,2414.50061,0.64586,2414.80834,0.64596,2415.11605,0.64606,2415.42374,0.64616,2415.73141,0.64626,2416.03907,0.64636,2416.34670,0.64646,2416.65432,0.64656,2416.96192,0.64666,2417.26949,0.64676,2417.57706,0.64686,2417.88460,0.64696,2418.19212,0.64706,2418.49962,0.64716,2418.80711,0.64726,2419.11458,0.64736,2419.42203,0.64746,2419.72945,0.64756,2420.03687,0.64766,2420.34426,0.64776,2420.65163,0.64786,2420.95899,0.64796,2421.26632,0.64806,2421.57364,0.64816,2421.88094,0.64826,2422.18822,0.64836,2422.49548,0.64846,2422.80273,0.64856,2423.10995,0.64866,2423.41716,0.64876,2423.72434,0.64886,2424.03151,0.64896,2424.33866,0.64906,2424.64579,0.64916,2424.95290,0.64926,2425.26000,0.64936,2425.56707,0.64946,2425.87413,0.64956,2426.18117,0.64966,2426.48819,0.64976,2426.79519,0.64986,2427.10217,0.64996,2427.40913,0.65007,2427.71608,0.65017,2428.02301,0.65027,2428.32991,0.65037,2428.63680,0.65047,2428.94367,0.65057,2429.25052,0.65067,2429.55736,0.65077,2429.86417,0.65087,2430.17097,0.65097,2430.47774,0.65107,2430.78450,0.65117,2431.09124,0.65127,2431.39796,0.65137,2431.70467,0.65147,2432.01135,0.65157,2432.31802,0.65167,2432.62466,0.65177,2432.93129,0.65187,2433.23790,0.65197,2433.54449,0.65207,2433.85106,0.65217,2434.15762,0.65227,2434.46415,0.65237,2434.77067,0.65247,2435.07717,0.65257,2435.38365,0.65267,2435.69011,0.65277,2435.99655,0.65287,2436.30297,0.65297,2436.60938,0.65307,2436.91576,0.65317,2437.22213,0.65327,2437.52848,0.65337,2437.83481,0.65347,2438.14112,0.65357,2438.44741,0.65367,2438.75369,0.65377,2439.05995,0.65387,2439.36618,0.65397,2439.67240,0.65407,2439.97860,0.65417,2440.28478,0.65427,2440.59095,0.65437,2440.89709,0.65447,2441.20322,0.65457,2441.50932,0.65467,2441.81541,0.65477,2442.12148,0.65487,2442.42754,0.65497,2442.73357,0.65507,2443.03958,0.65517,2443.34558,0.65527,2443.65156,0.65537,2443.95751,0.65547,2444.26345,0.65557,2444.56938,0.65567,2444.87528,0.65577,2445.18116,0.65587,2445.48703,0.65597,2445.79288,0.65607,2446.09871,0.65617,2446.40452,0.65627,2446.71031,0.65637,2447.01608,0.65647,2447.32184,0.65657,2447.62757,0.65667,2447.93329,0.65677,2448.23899,0.65687,2448.54467,0.65697,2448.85033,0.65707,2449.15597,0.65717,2449.46160,0.65727,2449.76720,0.65737,2450.07279,0.65747,2450.37836,0.65757,2450.68391,0.65767,2450.98944,0.65777,2451.29495,0.65787,2451.60045,0.65797,2451.90593,0.65807,2452.21138,0.65817,2452.51682,0.65827,2452.82224,0.65837,2453.12764,0.65847,2453.43303,0.65857,2453.73839,0.65867,2454.04374,0.65877,2454.34907,0.65887,2454.65438,0.65897,2454.95967,0.65907,2455.26494,0.65917,2455.57019,0.65927,2455.87543,0.65937,2456.18064,0.65947,2456.48584,0.65957,2456.79102,0.65967,2457.09618,0.65977,2457.40133,0.65987,2457.70645,0.65997,2458.01156,0.66007,2458.31664,0.66017,2458.62171,0.66027,2458.92676,0.66037,2459.23179,0.66047,2459.53681,0.66057,2459.84180,0.66067,2460.14678,0.66077,2460.45173,0.66087,2460.75667,0.66097,2461.06159,0.66107,2461.36649,0.66117,2461.67138,0.66127,2461.97624,0.66137,2462.28109,0.66147,2462.58592,0.66157,2462.89073,0.66167,2463.19552,0.66177,2463.50029,0.66187,2463.80504,0.66197,2464.10978,0.66207,2464.41449,0.66217,2464.71919,0.66227,2465.02387,0.66237,2465.32853,0.66247,2465.63318,0.66257,2465.93780,0.66267,2466.24241,0.66277,2466.54699,0.66287,2466.85156,0.66297,2467.15611,0.66307,2467.46064,0.66317,2467.76516,0.66327,2468.06965,0.66337,2468.37413,0.66347,2468.67859,0.66357,2468.98303,0.66367,2469.28745,0.66377,2469.59185,0.66387,2469.89623,0.66397,2470.20060,0.66407,2470.50495,0.66417,2470.80927,0.66427,2471.11358,0.66437,2471.41788,0.66447,2471.72215,0.66457,2472.02640,0.66467,2472.33064,0.66477,2472.63486,0.66487,2472.93906,0.66497,2473.24324,0.66507,2473.54740,0.66517,2473.85154,0.66527,2474.15567,0.66537,2474.45978,0.66547,2474.76387,0.66557,2475.06794,0.66567,2475.37199,0.66577,2475.67602,0.66587,2475.98003,0.66597,2476.28403,0.66607,2476.58801,0.66617,2476.89197,0.66627,2477.19591,0.66637,2477.49983,0.66647,2477.80373,0.66657,2478.10762,0.66667,2478.41149,0.66677,2478.71534,0.66687,2479.01917,0.66697,2479.32298,0.66707,2479.62677,0.66717,2479.93055,0.66727,2480.23430,0.66737,2480.53804,0.66747,2480.84176,0.66757,2481.14546,0.66767,2481.44914,0.66777,2481.75281,0.66787,2482.05645,0.66797,2482.36008,0.66807,2482.66369,0.66817,2482.96728,0.66827,2483.27085,0.66837,2483.57441,0.66847,2483.87794,0.66857,2484.18146,0.66867,2484.48496,0.66877,2484.78844,0.66887,2485.09190,0.66897,2485.39534,0.66907,2485.69877,0.66917,2486.00217,0.66927,2486.30556,0.66937,2486.60893,0.66947,2486.91228,0.66957,2487.21561,0.66967,2487.51893,0.66977,2487.82222,0.66987,2488.12550,0.66997,2488.42876,0.67007,2488.73200,0.67017,2489.03522,0.67027,2489.33843,0.67037,2489.64161,0.67047,2489.94478,0.67057,2490.24793,0.67067,2490.55106,0.67077,2490.85417,0.67087,2491.15726,0.67097,2491.46034,0.67107,2491.76339,0.67117,2492.06643,0.67127,2492.36945,0.67137,2492.67245,0.67147,2492.97543,0.67157,2493.27840,0.67167,2493.58135,0.67177,2493.88427,0.67187,2494.18718,0.67197,2494.49007,0.67207,2494.79295,0.67217,2495.09580,0.67227,2495.39864,0.67237,2495.70145,0.67247,2496.00425,0.67257,2496.30703,0.67267,2496.60980,0.67277,2496.91254,0.67287,2497.21527,0.67297,2497.51797,0.67307,2497.82066,0.67317,2498.12333,0.67327,2498.42598,0.67337,2498.72862,0.67347,2499.03123,0.67357,2499.33383,0.67367,2499.63641,0.67377,2499.93897,0.67387,2500.24151,0.67397,2500.54403,0.67407,2500.84654,0.67417,2501.14903,0.67427,2501.45150,0.67437,2501.75395,0.67447,2502.05638,0.67457,2502.35879,0.67467,2502.66119,0.67477,2502.96356,0.67487,2503.26592,0.67497,2503.56826,0.67507,2503.87058,0.67517,2504.17289,0.67527,2504.47517,0.67537,2504.77744,0.67547,2505.07969,0.67557,2505.38192,0.67567,2505.68413,0.67577,2505.98632,0.67587,2506.28849,0.67597,2506.59065,0.67607,2506.89279,0.67617,2507.19491,0.67627,2507.49701,0.67637,2507.79909,0.67647,2508.10116,0.67657,2508.40321,0.67667,2508.70523,0.67677,2509.00724,0.67687,2509.30924,0.67697,2509.61121,0.67707,2509.91316,0.67717,2510.21510,0.67727,2510.51702,0.67737,2510.81892,0.67747,2511.12080,0.67757,2511.42266,0.67767,2511.72451,0.67777,2512.02633,0.67787,2512.32814,0.67797,2512.62993,0.67807,2512.93170,0.67817,2513.23346,0.67827,2513.53519,0.67837,2513.83691,0.67847,2514.13861,0.67857,2514.44029,0.67867,2514.74195,0.67877,2515.04359,0.67887,2515.34522,0.67897,2515.64683,0.67907,2515.94841,0.67917,2516.24998,0.67927,2516.55154,0.67937,2516.85307,0.67947,2517.15459,0.67957,2517.45608,0.67967,2517.75756,0.67977,2518.05902,0.67987,2518.36046,0.67997,2518.66189,0.68007,2518.96329,0.68017,2519.26468,0.68027,2519.56605,0.68037,2519.86740,0.68047,2520.16873,0.68057,2520.47005,0.68067,2520.77134,0.68077,2521.07262,0.68087,2521.37388,0.68097,2521.67512,0.68107,2521.97634,0.68117,2522.27755,0.68127,2522.57873,0.68137,2522.87990,0.68147,2523.18105,0.68157,2523.48218,0.68167,2523.78330,0.68177,2524.08439,0.68187,2524.38547,0.68197,2524.68653,0.68207,2524.98757,0.68217,2525.28859,0.68227,2525.58959,0.68237,2525.89058,0.68247,2526.19154,0.68257,2526.49249,0.68267,2526.79342,0.68277,2527.09433,0.68287,2527.39523,0.68297,2527.69610,0.68307,2527.99696,0.68317,2528.29780,0.68327,2528.59862,0.68337,2528.89942,0.68347,2529.20021,0.68357,2529.50097,0.68367,2529.80172,0.68377,2530.10245,0.68387,2530.40316,0.68397,2530.70386,0.68407,2531.00453,0.68417,2531.30519,0.68427,2531.60583,0.68437,2531.90645,0.68447,2532.20705,0.68457,2532.50763,0.68467,2532.80820,0.68477,2533.10874,0.68487,2533.40927,0.68497,2533.70978,0.68507,2534.01028,0.68517,2534.31075,0.68527,2534.61121,0.68537,2534.91164,0.68547,2535.21206,0.68557,2535.51246,0.68567,2535.81285,0.68577,2536.11321,0.68587,2536.41356,0.68597,2536.71389,0.68607,2537.01420,0.68617,2537.31449,0.68627,2537.61476,0.68637,2537.91502,0.68647,2538.21526,0.68657,2538.51547,0.68667,2538.81568,0.68677,2539.11586,0.68687,2539.41602,0.68697,2539.71617,0.68707,2540.01630,0.68717,2540.31641,0.68727,2540.61650,0.68737,2540.91657,0.68747,2541.21663,0.68757,2541.51666,0.68767,2541.81668,0.68777,2542.11668,0.68787,2542.41666,0.68797,2542.71663,0.68807,2543.01657,0.68817,2543.31650,0.68827,2543.61641,0.68837,2543.91630,0.68847,2544.21618,0.68857,2544.51603,0.68867,2544.81587,0.68877,2545.11569,0.68887,2545.41549,0.68897,2545.71527,0.68907,2546.01503,0.68917,2546.31478,0.68927,2546.61451,0.68937,2546.91422,0.68947,2547.21391,0.68957,2547.51358,0.68967,2547.81324,0.68977,2548.11287,0.68987,2548.41249,0.68997,2548.71209,0.69007,2549.01167,0.69017,2549.31124,0.69027,2549.61078,0.69037,2549.91031,0.69047,2550.20982,0.69057,2550.50931,0.69067,2550.80878,0.69077,2551.10824,0.69087,2551.40768,0.69097,2551.70709,0.69107,2552.00650,0.69117,2552.30588,0.69127,2552.60524,0.69137,2552.90459,0.69147,2553.20392,0.69157,2553.50322,0.69167,2553.80252,0.69177,2554.10179,0.69187,2554.40104,0.69197,2554.70028,0.69207,2554.99950,0.69217,2555.29870,0.69227,2555.59788,0.69237,2555.89705,0.69247,2556.19619,0.69257,2556.49532,0.69267,2556.79443,0.69277,2557.09352,0.69287,2557.39260,0.69297,2557.69165,0.69307,2557.99069,0.69317,2558.28971,0.69327,2558.58871,0.69337,2558.88769,0.69347,2559.18666,0.69357,2559.48561,0.69367,2559.78453,0.69377,2560.08344,0.69387,2560.38234,0.69397,2560.68121,0.69407,2560.98007,0.69417,2561.27891,0.69427,2561.57773,0.69437,2561.87653,0.69447,2562.17531,0.69457,2562.47408,0.69467,2562.77282,0.69477,2563.07155,0.69487,2563.37026,0.69497,2563.66896,0.69507,2563.96763,0.69517,2564.26629,0.69527,2564.56493,0.69537,2564.86355,0.69547,2565.16215,0.69557,2565.46073,0.69567,2565.75930,0.69577,2566.05785,0.69587,2566.35638,0.69597,2566.65489,0.69607,2566.95338,0.69617,2567.25186,0.69627,2567.55032,0.69637,2567.84875,0.69647,2568.14718,0.69657,2568.44558,0.69667,2568.74396,0.69677,2569.04233,0.69687,2569.34068,0.69697,2569.63901,0.69707,2569.93732,0.69717,2570.23562,0.69727,2570.53389,0.69737,2570.83215,0.69747,2571.13039,0.69757,2571.42861,0.69767,2571.72682,0.69777,2572.02500,0.69787,2572.32317,0.69797,2572.62132,0.69807,2572.91945,0.69817,2573.21757,0.69827,2573.51566,0.69837,2573.81374,0.69847,2574.11180,0.69857,2574.40984,0.69867,2574.70787,0.69877,2575.00587,0.69887,2575.30386,0.69897,2575.60183,0.69907,2575.89978,0.69917,2576.19771,0.69927,2576.49563,0.69937,2576.79352,0.69947,2577.09140,0.69957,2577.38926,0.69967,2577.68710,0.69977,2577.98493,0.69987,2578.28273,0.69997,2578.58052,0.70007,2578.87829,0.70017,2579.17605,0.70027,2579.47378,0.70037,2579.77150,0.70047,2580.06919,0.70057,2580.36687,0.70067,2580.66454,0.70077,2580.96218,0.70087,2581.25981,0.70097,2581.55741,0.70107,2581.85500,0.70117,2582.15257,0.70127,2582.45013,0.70137,2582.74766,0.70147,2583.04518,0.70157,2583.34268,0.70167,2583.64016,0.70177,2583.93763,0.70187,2584.23507,0.70197,2584.53250,0.70207,2584.82991,0.70217,2585.12730,0.70227,2585.42467,0.70237,2585.72203,0.70247,2586.01936,0.70257,2586.31668,0.70267,2586.61399,0.70277,2586.91127,0.70287,2587.20853,0.70297,2587.50578,0.70307,2587.80301,0.70317,2588.10022,0.70327,2588.39741,0.70337,2588.69459,0.70347,2588.99175,0.70357,2589.28888,0.70367,2589.58600,0.70377,2589.88311,0.70387,2590.18019,0.70397,2590.47726,0.70407,2590.77431,0.70417,2591.07134,0.70427,2591.36835,0.70437,2591.66535,0.70447,2591.96232,0.70457,2592.25928,0.70467,2592.55622,0.70477,2592.85315,0.70487,2593.15005,0.70497,2593.44694,0.70507,2593.74381,0.70517,2594.04066,0.70527,2594.33749,0.70537,2594.63430,0.70547,2594.93110,0.70557,2595.22788,0.70567,2595.52464,0.70577,2595.82138,0.70587,2596.11811,0.70597,2596.41481,0.70607,2596.71150,0.70617,2597.00817,0.70627,2597.30483,0.70637,2597.60146,0.70647,2597.89808,0.70657,2598.19468,0.70667,2598.49126,0.70677,2598.78782,0.70687,2599.08437,0.70697,2599.38089,0.70707,2599.67740,0.70717,2599.97389,0.70727,2600.27036,0.70737,2600.56682,0.70747,2600.86326,0.70757,2601.15968,0.70767,2601.45608,0.70777,2601.75246,0.70787,2602.04882,0.70797,2602.34517,0.70807,2602.64150,0.70817,2602.93781,0.70827,2603.23411,0.70837,2603.53038,0.70847,2603.82664,0.70857,2604.12288,0.70867,2604.41910,0.70877,2604.71530,0.70887,2605.01149,0.70897,2605.30766,0.70907,2605.60380,0.70917,2605.89994,0.70927,2606.19605,0.70937,2606.49215,0.70947,2606.78822,0.70957,2607.08428,0.70967,2607.38032,0.70977,2607.67635,0.70987,2607.97235,0.70997,2608.26834,0.71007,2608.56431,0.71017,2608.86026,0.71027,2609.15620,0.71037,2609.45212,0.71047,2609.74801,0.71057,2610.04389,0.71067,2610.33976,0.71077,2610.63560,0.71087,2610.93143,0.71097,2611.22724,0.71107,2611.52303,0.71117,2611.81880,0.71127,2612.11455,0.71137,2612.41029,0.71147,2612.70601,0.71157,2613.00171,0.71167,2613.29739,0.71177,2613.59306,0.71187,2613.88871,0.71197,2614.18434,0.71207,2614.47995,0.71217,2614.77554,0.71227,2615.07112,0.71237,2615.36667,0.71247,2615.66221,0.71257,2615.95773,0.71267,2616.25324,0.71277,2616.54872,0.71287,2616.84419,0.71297,2617.13964,0.71307,2617.43508,0.71317,2617.73049,0.71327,2618.02589,0.71337,2618.32126,0.71347,2618.61663,0.71357,2618.91197,0.71367,2619.20729,0.71377,2619.50260,0.71387,2619.79789,0.71397,2620.09316,0.71407,2620.38841,0.71417,2620.68365,0.71427,2620.97887,0.71437,2621.27406,0.71447,2621.56925,0.71457,2621.86441,0.71467,2622.15956,0.71477,2622.45468,0.71487,2622.74979,0.71497,2623.04489,0.71507,2623.33996,0.71517,2623.63502,0.71527,2623.93006,0.71537,2624.22508,0.71547,2624.52008,0.71557,2624.81506,0.71567,2625.11003,0.71577,2625.40498,0.71587,2625.69991,0.71597,2625.99482,0.71607,2626.28972,0.71617,2626.58460,0.71627,2626.87946,0.71637,2627.17430,0.71647,2627.46912,0.71657,2627.76393,0.71667,2628.05872,0.71677,2628.35349,0.71687,2628.64824,0.71697,2628.94297,0.71707,2629.23769,0.71717,2629.53239,0.71727,2629.82707,0.71737,2630.12173,0.71747,2630.41638,0.71757,2630.71101,0.71767,2631.00562,0.71777,2631.30021,0.71787,2631.59478,0.71797,2631.88934,0.71807,2632.18388,0.71817,2632.47840,0.71827,2632.77290,0.71837,2633.06738,0.71847,2633.36185,0.71857,2633.65630,0.71867,2633.95073,0.71877,2634.24514,0.71887,2634.53954,0.71897,2634.83392,0.71907,2635.12828,0.71917,2635.42262,0.71927,2635.71694,0.71937,2636.01125,0.71947,2636.30554,0.71957,2636.59981,0.71967,2636.89406,0.71977,2637.18830,0.71987,2637.48251,0.71997,2637.77671,0.72007,2638.07089,0.72017,2638.36506,0.72027,2638.65920,0.72037,2638.95333,0.72047,2639.24744,0.72057,2639.54153,0.72067,2639.83561,0.72077,2640.12966,0.72087,2640.42370,0.72097,2640.71772,0.72107,2641.01173,0.72117,2641.30571,0.72127,2641.59968,0.72137,2641.89363,0.72147,2642.18756,0.72157,2642.48148,0.72167,2642.77537,0.72177,2643.06925,0.72187,2643.36311,0.72197,2643.65696,0.72207,2643.95078,0.72217,2644.24459,0.72227,2644.53838,0.72237,2644.83215,0.72247,2645.12590,0.72257,2645.41964,0.72267,2645.71336,0.72277,2646.00706,0.72287,2646.30074,0.72297,2646.59441,0.72307,2646.88805,0.72317,2647.18168,0.72327,2647.47529,0.72337,2647.76889,0.72347,2648.06246,0.72357,2648.35602,0.72367,2648.64956,0.72377,2648.94308,0.72387,2649.23659,0.72397,2649.53008,0.72407,2649.82354,0.72417,2650.11700,0.72427,2650.41043,0.72437,2650.70385,0.72447,2650.99724,0.72457,2651.29062,0.72467,2651.58399,0.72477,2651.87733,0.72487,2652.17066,0.72497,2652.46397,0.72507,2652.75726,0.72517,2653.05053,0.72527,2653.34379,0.72537,2653.63703,0.72547,2653.93025,0.72557,2654.22345,0.72567,2654.51664,0.72577,2654.80980,0.72587,2655.10295,0.72597,2655.39608,0.72607,2655.68920,0.72617,2655.98229,0.72627,2656.27537,0.72637,2656.56843,0.72647,2656.86148,0.72657,2657.15450,0.72667,2657.44751,0.72677,2657.74050,0.72687,2658.03347,0.72697,2658.32642,0.72707,2658.61936,0.72717,2658.91228,0.72727,2659.20518,0.72737,2659.49806,0.72747,2659.79093,0.72757,2660.08377,0.72767,2660.37660,0.72777,2660.66942,0.72787,2660.96221,0.72797,2661.25499,0.72807,2661.54775,0.72817,2661.84049,0.72827,2662.13321,0.72837,2662.42592,0.72847,2662.71861,0.72857,2663.01128,0.72867,2663.30393,0.72877,2663.59656,0.72887,2663.88918,0.72897,2664.18178,0.72907,2664.47436,0.72917,2664.76692,0.72927,2665.05947,0.72937,2665.35200,0.72947,2665.64451,0.72957,2665.93700,0.72967,2666.22948,0.72977,2666.52194,0.72987,2666.81438,0.72997,2667.10680,0.73007,2667.39920,0.73017,2667.69159,0.73027,2667.98396,0.73037,2668.27631,0.73047,2668.56864,0.73057,2668.86096,0.73067,2669.15326,0.73077,2669.44554,0.73087,2669.73780,0.73097,2670.03005,0.73107,2670.32227,0.73117,2670.61448,0.73127,2670.90668,0.73137,2671.19885,0.73147,2671.49101,0.73157,2671.78315,0.73167,2672.07527,0.73177,2672.36737,0.73187,2672.65946,0.73197,2672.95152,0.73207,2673.24357,0.73217,2673.53561,0.73227,2673.82762,0.73237,2674.11962,0.73247,2674.41160,0.73257,2674.70356,0.73267,2674.99551,0.73277,2675.28743,0.73287,2675.57934,0.73297,2675.87123,0.73307,2676.16311,0.73317,2676.45496,0.73327,2676.74680,0.73337,2677.03862,0.73347,2677.33043,0.73357,2677.62221,0.73367,2677.91398,0.73377,2678.20573,0.73387,2678.49746,0.73397,2678.78917,0.73407,2679.08087,0.73417,2679.37255,0.73427,2679.66421,0.73437,2679.95586,0.73447,2680.24748,0.73457,2680.53909,0.73467,2680.83068,0.73477,2681.12226,0.73487,2681.41381,0.73497,2681.70535,0.73507,2681.99687,0.73517,2682.28837,0.73527,2682.57986,0.73537,2682.87133,0.73547,2683.16278,0.73557,2683.45421,0.73567,2683.74562,0.73577,2684.03702,0.73587,2684.32840,0.73597,2684.61976,0.73607,2684.91110,0.73617,2685.20243,0.73627,2685.49374,0.73637,2685.78503,0.73647,2686.07630,0.73657,2686.36756,0.73667,2686.65880,0.73677,2686.95002,0.73687,2687.24122,0.73697,2687.53241,0.73707,2687.82357,0.73717,2688.11472,0.73727,2688.40586,0.73737,2688.69697,0.73747,2688.98807,0.73757,2689.27915,0.73767,2689.57021,0.73777,2689.86125,0.73787,2690.15228,0.73797,2690.44329,0.73807,2690.73428,0.73817,2691.02525,0.73827,2691.31621,0.73837,2691.60715,0.73847,2691.89807,0.73857,2692.18897,0.73867,2692.47986,0.73877,2692.77073,0.73887,2693.06158,0.73897,2693.35241,0.73907,2693.64322,0.73917,2693.93402,0.73927,2694.22480,0.73937,2694.51556,0.73947,2694.80631,0.73957,2695.09704,0.73967,2695.38775,0.73977,2695.67844,0.73987,2695.96911,0.73997,2696.25977,0.74007,2696.55041,0.74017,2696.84103,0.74027,2697.13163,0.74037,2697.42222,0.74047,2697.71279,0.74057,2698.00334,0.74067,2698.29387,0.74077,2698.58439,0.74087,2698.87489,0.74097,2699.16537,0.74107,2699.45583,0.74117,2699.74628,0.74127,2700.03671,0.74137,2700.32712,0.74147,2700.61751,0.74157,2700.90789,0.74167,2701.19824,0.74177,2701.48858,0.74187,2701.77891,0.74197,2702.06921,0.74207,2702.35950,0.74217,2702.64977,0.74227,2702.94002,0.74237,2703.23026,0.74247,2703.52047,0.74257,2703.81067,0.74267,2704.10086,0.74277,2704.39102,0.74287,2704.68117,0.74297,2704.97130,0.74307,2705.26141,0.74317,2705.55150,0.74327,2705.84158,0.74337,2706.13164,0.74347,2706.42168,0.74357,2706.71171,0.74367,2707.00171,0.74377,2707.29170,0.74387,2707.58167,0.74397,2707.87163,0.74407,2708.16156,0.74417,2708.45148,0.74427,2708.74138,0.74437,2709.03127,0.74447,2709.32113,0.74457,2709.61098,0.74467,2709.90081,0.74477,2710.19063,0.74487,2710.48042,0.74497,2710.77020,0.74507,2711.05996,0.74517,2711.34971,0.74527,2711.63943,0.74537,2711.92914,0.74547,2712.21883,0.74557,2712.50850,0.74567,2712.79816,0.74577,2713.08780,0.74587,2713.37742,0.74597,2713.66702,0.74607,2713.95661,0.74617,2714.24618,0.74627,2714.53573,0.74637,2714.82526,0.74647,2715.11478,0.74657,2715.40427,0.74667,2715.69375,0.74677,2715.98322,0.74687,2716.27266,0.74697,2716.56209,0.74707,2716.85150,0.74717,2717.14089,0.74727,2717.43027,0.74737,2717.71963,0.74747,2718.00897,0.74757,2718.29829,0.74767,2718.58760,0.74777,2718.87689,0.74787,2719.16616,0.74797,2719.45541,0.74807,2719.74464,0.74817,2720.03386,0.74827,2720.32306,0.74837,2720.61225,0.74847,2720.90141,0.74857,2721.19056,0.74867,2721.47969,0.74877,2721.76880,0.74887,2722.05790,0.74897,2722.34698,0.74907,2722.63604,0.74917,2722.92508,0.74927,2723.21411,0.74937,2723.50311,0.74947,2723.79210,0.74957,2724.08108,0.74967,2724.37003,0.74977,2724.65897,0.74987,2724.94789,0.74997,2725.23679,0.75008,2725.52568,0.75018,2725.81455,0.75028,2726.10340,0.75038,2726.39223,0.75048,2726.68105,0.75058,2726.96985,0.75068,2727.25863,0.75078,2727.54739,0.75088,2727.83614,0.75098,2728.12487,0.75108,2728.41358,0.75118,2728.70227,0.75128,2728.99095,0.75138,2729.27960,0.75148,2729.56825,0.75158,2729.85687,0.75168,2730.14548,0.75178,2730.43406,0.75188,2730.72264,0.75198,2731.01119,0.75208,2731.29973,0.75218,2731.58825,0.75228,2731.87675,0.75238,2732.16523,0.75248,2732.45370,0.75258,2732.74215,0.75268,2733.03058,0.75278,2733.31899,0.75288,2733.60739,0.75298,2733.89577,0.75308,2734.18413,0.75318,2734.47248,0.75328,2734.76080,0.75338,2735.04911,0.75348,2735.33740,0.75358,2735.62568,0.75368,2735.91394,0.75378,2736.20218,0.75388,2736.49040,0.75398,2736.77860,0.75408,2737.06679,0.75418,2737.35496,0.75428,2737.64311,0.75438,2737.93125,0.75448,2738.21937,0.75458,2738.50747,0.75468,2738.79555,0.75478,2739.08362,0.75488,2739.37166,0.75498,2739.65970,0.75508,2739.94771,0.75518,2740.23570,0.75528,2740.52368,0.75538,2740.81164,0.75548,2741.09959,0.75558,2741.38751,0.75568,2741.67542,0.75578,2741.96331,0.75588,2742.25119,0.75598,2742.53904,0.75608,2742.82688,0.75618,2743.11471,0.75628,2743.40251,0.75638,2743.69030,0.75648,2743.97807,0.75658,2744.26582,0.75668,2744.55355,0.75678,2744.84127,0.75688,2745.12897,0.75698,2745.41665,0.75708,2745.70432,0.75718,2745.99197,0.75728,2746.27960,0.75738,2746.56721,0.75748,2746.85481,0.75758,2747.14238,0.75768,2747.42995,0.75778,2747.71749,0.75788,2748.00502,0.75798,2748.29252,0.75808,2748.58002,0.75818,2748.86749,0.75828,2749.15495,0.75838,2749.44238,0.75848,2749.72981,0.75858,2750.01721,0.75868,2750.30460,0.75878,2750.59197,0.75888,2750.87932,0.75898,2751.16666,0.75908,2751.45397,0.75918,2751.74127,0.75928,2752.02856,0.75938,2752.31582,0.75948,2752.60307,0.75958,2752.89030,0.75968,2753.17751,0.75978,2753.46471,0.75988,2753.75189,0.75998,2754.03905,0.76008,2754.32619,0.76018,2754.61332,0.76028,2754.90043,0.76038,2755.18752,0.76048,2755.47460,0.76058,2755.76165,0.76068,2756.04869,0.76078,2756.33571,0.76088,2756.62272,0.76098,2756.90971,0.76108,2757.19668,0.76118,2757.48363,0.76128,2757.77057,0.76138,2758.05748,0.76148,2758.34439,0.76158,2758.63127,0.76168,2758.91814,0.76178,2759.20498,0.76188,2759.49182,0.76198,2759.77863,0.76208,2760.06543,0.76218,2760.35221,0.76228,2760.63897,0.76238,2760.92571,0.76248,2761.21244,0.76258,2761.49915,0.76268,2761.78585,0.76278,2762.07252,0.76288,2762.35918,0.76298,2762.64582,0.76308,2762.93244,0.76318,2763.21905,0.76328,2763.50564,0.76338,2763.79221,0.76348,2764.07877,0.76358,2764.36530,0.76368,2764.65182,0.76378,2764.93833,0.76388,2765.22481,0.76398,2765.51128,0.76408,2765.79773,0.76418,2766.08416,0.76428,2766.37058,0.76438,2766.65698,0.76448,2766.94336,0.76458,2767.22972,0.76468,2767.51607,0.76478,2767.80240,0.76488,2768.08871,0.76498,2768.37500,0.76508,2768.66128,0.76518,2768.94754,0.76528,2769.23378,0.76538,2769.52001,0.76548,2769.80622,0.76558,2770.09241,0.76568,2770.37858,0.76578,2770.66474,0.76588,2770.95088,0.76598,2771.23700,0.76608,2771.52310,0.76618,2771.80919,0.76628,2772.09526,0.76638,2772.38131,0.76648,2772.66735,0.76658,2772.95337,0.76668,2773.23937,0.76678,2773.52535,0.76688,2773.81132,0.76698,2774.09727,0.76708,2774.38320,0.76718,2774.66911,0.76728,2774.95501,0.76738,2775.24089,0.76748,2775.52675,0.76758,2775.81260,0.76768,2776.09843,0.76778,2776.38424,0.76788,2776.67003,0.76798,2776.95581,0.76808,2777.24157,0.76818,2777.52731,0.76828,2777.81303,0.76838,2778.09874,0.76848,2778.38443,0.76858,2778.67010,0.76868,2778.95576,0.76878,2779.24139,0.76888,2779.52702,0.76898,2779.81262,0.76908,2780.09821,0.76918,2780.38377,0.76928,2780.66933,0.76938,2780.95486,0.76948,2781.24038,0.76958,2781.52588,0.76968,2781.81136,0.76978,2782.09683,0.76988,2782.38228,0.76998,2782.66771,0.77008,2782.95312,0.77018,2783.23852,0.77028,2783.52390,0.77038,2783.80926,0.77048,2784.09460,0.77058,2784.37993,0.77068,2784.66524,0.77078,2784.95053,0.77088,2785.23581,0.77098,2785.52107,0.77108,2785.80631,0.77118,2786.09154,0.77128,2786.37674,0.77138,2786.66193,0.77148,2786.94711,0.77158,2787.23226,0.77168,2787.51740,0.77178,2787.80252,0.77188,2788.08762,0.77198,2788.37271,0.77208,2788.65778,0.77218,2788.94283,0.77228,2789.22787,0.77238,2789.51288,0.77248,2789.79788,0.77258,2790.08287,0.77268,2790.36783,0.77278,2790.65278,0.77288,2790.93771,0.77298,2791.22263,0.77308,2791.50753,0.77318,2791.79241,0.77328,2792.07727,0.77338,2792.36211,0.77348,2792.64694,0.77358,2792.93175,0.77368,2793.21655,0.77378,2793.50132,0.77388,2793.78608,0.77398,2794.07083,0.77408,2794.35555,0.77418,2794.64026,0.77428,2794.92495,0.77438,2795.20962,0.77448,2795.49428,0.77458,2795.77892,0.77468,2796.06354,0.77478,2796.34815,0.77488,2796.63273,0.77498,2796.91730,0.77508,2797.20186,0.77518,2797.48639,0.77528,2797.77091,0.77538,2798.05541,0.77548,2798.33990,0.77558,2798.62437,0.77568,2798.90882,0.77578,2799.19325,0.77588,2799.47766,0.77598,2799.76206,0.77608,2800.04644,0.77618,2800.33081,0.77628,2800.61516,0.77638,2800.89949,0.77648,2801.18380,0.77658,2801.46809,0.77668,2801.75237,0.77678,2802.03663,0.77688,2802.32088,0.77698,2802.60511,0.77708,2802.88932,0.77718,2803.17351,0.77728,2803.45768,0.77738,2803.74184,0.77748,2804.02598,0.77758,2804.31011,0.77768,2804.59421,0.77778,2804.87830,0.77788,2805.16238,0.77798,2805.44643,0.77808,2805.73047,0.77818,2806.01449,0.77828,2806.29850,0.77838,2806.58248,0.77848,2806.86645,0.77858,2807.15040,0.77868,2807.43434,0.77878,2807.71826,0.77888,2808.00216,0.77898,2808.28604,0.77908,2808.56991,0.77918,2808.85376,0.77928,2809.13759,0.77938,2809.42141,0.77948,2809.70521,0.77958,2809.98899,0.77968,2810.27275,0.77978,2810.55650,0.77988,2810.84023,0.77998,2811.12394,0.78008,2811.40764,0.78018,2811.69132,0.78028,2811.97498,0.78038,2812.25862,0.78048,2812.54225,0.78058,2812.82586,0.78068,2813.10945,0.78078,2813.39303,0.78088,2813.67659,0.78098,2813.96013,0.78108,2814.24365,0.78118,2814.52716,0.78128,2814.81065,0.78138,2815.09412,0.78148,2815.37758,0.78158,2815.66102,0.78168,2815.94444,0.78178,2816.22784,0.78188,2816.51123,0.78198,2816.79460,0.78208,2817.07796,0.78218,2817.36129,0.78228,2817.64461,0.78238,2817.92791,0.78248,2818.21120,0.78258,2818.49447,0.78268,2818.77772,0.78278,2819.06095,0.78288,2819.34417,0.78298,2819.62737,0.78308,2819.91055,0.78318,2820.19371,0.78328,2820.47686,0.78338,2820.75999,0.78348,2821.04311,0.78358,2821.32620,0.78368,2821.60928,0.78378,2821.89235,0.78388,2822.17539,0.78398,2822.45842,0.78408,2822.74143,0.78418,2823.02443,0.78428,2823.30740,0.78438,2823.59037,0.78448,2823.87331,0.78458,2824.15623,0.78468,2824.43914,0.78478,2824.72204,0.78488,2825.00491,0.78498,2825.28777,0.78508,2825.57061,0.78518,2825.85343,0.78528,2826.13624,0.78538,2826.41903,0.78548,2826.70180,0.78558,2826.98456,0.78568,2827.26730,0.78578,2827.55002,0.78588,2827.83272,0.78598,2828.11541,0.78608,2828.39808,0.78618,2828.68073,0.78628,2828.96337,0.78638,2829.24599,0.78648,2829.52859,0.78658,2829.81118,0.78668,2830.09374,0.78678,2830.37629,0.78688,2830.65883,0.78698,2830.94135,0.78708,2831.22385,0.78718,2831.50633,0.78728,2831.78879,0.78738,2832.07124,0.78748,2832.35367,0.78758,2832.63609,0.78768,2832.91849,0.78778,2833.20087,0.78788,2833.48323,0.78798,2833.76558,0.78808,2834.04791,0.78818,2834.33022,0.78828,2834.61251,0.78838,2834.89479,0.78848,2835.17705,0.78858,2835.45930,0.78868,2835.74153,0.78878,2836.02374,0.78888,2836.30593,0.78898,2836.58810,0.78908,2836.87026,0.78918,2837.15241,0.78928,2837.43453,0.78938,2837.71664,0.78948,2837.99873,0.78958,2838.28080,0.78968,2838.56286,0.78978,2838.84490,0.78988,2839.12693,0.78998,2839.40893,0.79008,2839.69092,0.79018,2839.97289,0.79028,2840.25485,0.79038,2840.53679,0.79048,2840.81871,0.79058,2841.10061,0.79068,2841.38250,0.79078,2841.66437,0.79088,2841.94622,0.79098,2842.22806,0.79108,2842.50988,0.79118,2842.79168,0.79128,2843.07346,0.79138,2843.35523,0.79148,2843.63698,0.79158,2843.91872,0.79168,2844.20043,0.79178,2844.48213,0.79188,2844.76382,0.79198,2845.04548,0.79208,2845.32713,0.79218,2845.60876,0.79228,2845.89038,0.79238,2846.17198,0.79248,2846.45356,0.79258,2846.73512,0.79268,2847.01667,0.79278,2847.29820,0.79288,2847.57971,0.79298,2847.86121,0.79308,2848.14269,0.79318,2848.42415,0.79328,2848.70560,0.79338,2848.98703,0.79348,2849.26844,0.79358,2849.54983,0.79368,2849.83121,0.79378,2850.11257,0.79388,2850.39391,0.79398,2850.67524,0.79408,2850.95655,0.79418,2851.23784,0.79428,2851.51912,0.79438,2851.80038,0.79448,2852.08162,0.79458,2852.36285,0.79468,2852.64405,0.79478,2852.92524,0.79488,2853.20642,0.79498,2853.48758,0.79508,2853.76872,0.79518,2854.04984,0.79528,2854.33095,0.79538,2854.61204,0.79548,2854.89311,0.79558,2855.17416,0.79568,2855.45520,0.79578,2855.73622,0.79588,2856.01723,0.79598,2856.29822,0.79608,2856.57919,0.79618,2856.86014,0.79628,2857.14108,0.79638,2857.42200,0.79648,2857.70290,0.79658,2857.98379,0.79668,2858.26466,0.79678,2858.54551,0.79688,2858.82634,0.79698,2859.10716,0.79708,2859.38796,0.79718,2859.66875,0.79728,2859.94952,0.79738,2860.23027,0.79748,2860.51100,0.79758,2860.79172,0.79768,2861.07242,0.79778,2861.35310,0.79788,2861.63377,0.79798,2861.91442,0.79808,2862.19505,0.79818,2862.47566,0.79828,2862.75626,0.79838,2863.03684,0.79848,2863.31741,0.79858,2863.59796,0.79868,2863.87849,0.79878,2864.15900,0.79888,2864.43950,0.79898,2864.71998,0.79908,2865.00044,0.79918,2865.28089,0.79928,2865.56132,0.79938,2865.84173,0.79948,2866.12213,0.79958,2866.40250,0.79968,2866.68287,0.79978,2866.96321,0.79988,2867.24354,0.79998,2867.52385,0.80008,2867.80415,0.80018,2868.08442,0.80028,2868.36468,0.80038,2868.64493,0.80048,2868.92515,0.80058,2869.20536,0.80068,2869.48556,0.80078,2869.76573,0.80088,2870.04589,0.80098,2870.32603,0.80108,2870.60616,0.80118,2870.88627,0.80128,2871.16636,0.80138,2871.44643,0.80148,2871.72649,0.80158,2872.00653,0.80168,2872.28656,0.80178,2872.56657,0.80188,2872.84656,0.80198,2873.12653,0.80208,2873.40649,0.80218,2873.68643,0.80228,2873.96635,0.80238,2874.24626,0.80248,2874.52614,0.80258,2874.80602,0.80268,2875.08587,0.80278,2875.36571,0.80288,2875.64553,0.80298,2875.92534,0.80308,2876.20513,0.80318,2876.48490,0.80328,2876.76465,0.80338,2877.04439,0.80348,2877.32411,0.80358,2877.60381,0.80368,2877.88350,0.80378,2878.16317,0.80388,2878.44283,0.80398,2878.72246,0.80408,2879.00208,0.80418,2879.28168,0.80428,2879.56127,0.80438,2879.84084,0.80448,2880.12039,0.80458,2880.39993,0.80468,2880.67945,0.80478,2880.95895,0.80488,2881.23843,0.80498,2881.51790,0.80508,2881.79735,0.80518,2882.07679,0.80528,2882.35621,0.80538,2882.63561,0.80548,2882.91499,0.80558,2883.19436,0.80568,2883.47371,0.80578,2883.75304,0.80588,2884.03236,0.80598,2884.31166,0.80608,2884.59094,0.80618,2884.87021,0.80628,2885.14946,0.80638,2885.42869,0.80648,2885.70791,0.80658,2885.98710,0.80668,2886.26629,0.80678,2886.54545,0.80688,2886.82460,0.80698,2887.10373,0.80708,2887.38285,0.80718,2887.66195,0.80728,2887.94103,0.80738,2888.22009,0.80748,2888.49914,0.80758,2888.77817,0.80768,2889.05718,0.80778,2889.33618,0.80788,2889.61516,0.80798,2889.89413,0.80808,2890.17307,0.80818,2890.45200,0.80828,2890.73092,0.80838,2891.00981,0.80848,2891.28869,0.80858,2891.56756,0.80868,2891.84640,0.80878,2892.12523,0.80888,2892.40404,0.80898,2892.68284,0.80908,2892.96162,0.80918,2893.24038,0.80928,2893.51913,0.80938,2893.79786,0.80948,2894.07657,0.80958,2894.35526,0.80968,2894.63394,0.80978,2894.91260,0.80988,2895.19125,0.80998,2895.46988,0.81008,2895.74849,0.81018,2896.02708,0.81028,2896.30566,0.81038,2896.58422,0.81048,2896.86276,0.81058,2897.14129,0.81068,2897.41980,0.81078,2897.69830,0.81088,2897.97677,0.81098,2898.25523,0.81108,2898.53368,0.81118,2898.81211,0.81128,2899.09052,0.81138,2899.36891,0.81148,2899.64729,0.81158,2899.92565,0.81168,2900.20399,0.81178,2900.48231,0.81188,2900.76062,0.81198,2901.03892,0.81208,2901.31719,0.81218,2901.59545,0.81228,2901.87370,0.81238,2902.15192,0.81248,2902.43013,0.81258,2902.70832,0.81268,2902.98650,0.81278,2903.26466,0.81288,2903.54280,0.81298,2903.82092,0.81308,2904.09903,0.81318,2904.37713,0.81328,2904.65520,0.81338,2904.93326,0.81348,2905.21130,0.81358,2905.48933,0.81368,2905.76733,0.81378,2906.04533,0.81388,2906.32330,0.81398,2906.60126,0.81408,2906.87920,0.81418,2907.15712,0.81428,2907.43503,0.81438,2907.71292,0.81448,2907.99080,0.81458,2908.26866,0.81468,2908.54650,0.81478,2908.82432,0.81488,2909.10213,0.81498,2909.37992,0.81508,2909.65769,0.81518,2909.93545,0.81528,2910.21319,0.81538,2910.49092,0.81548,2910.76862,0.81558,2911.04632,0.81568,2911.32399,0.81578,2911.60165,0.81588,2911.87929,0.81598,2912.15691,0.81608,2912.43452,0.81618,2912.71211,0.81628,2912.98968,0.81638,2913.26724,0.81648,2913.54478,0.81658,2913.82230,0.81668,2914.09981,0.81678,2914.37730,0.81688,2914.65477,0.81698,2914.93223,0.81708,2915.20967,0.81718,2915.48709,0.81728,2915.76450,0.81738,2916.04189,0.81748,2916.31926,0.81758,2916.59662,0.81768,2916.87396,0.81778,2917.15129,0.81788,2917.42859,0.81798,2917.70588,0.81808,2917.98316,0.81818,2918.26041,0.81828,2918.53765,0.81838,2918.81488,0.81848,2919.09208,0.81858,2919.36927,0.81868,2919.64645,0.81878,2919.92360,0.81888,2920.20074,0.81898,2920.47787,0.81908,2920.75497,0.81918,2921.03206,0.81928,2921.30914,0.81938,2921.58619,0.81948,2921.86323,0.81958,2922.14026,0.81968,2922.41726,0.81978,2922.69425,0.81988,2922.97123,0.81998,2923.24818,0.82008,2923.52512,0.82018,2923.80205,0.82028,2924.07895,0.82038,2924.35584,0.82048,2924.63272,0.82058,2924.90957,0.82068,2925.18641,0.82078,2925.46324,0.82088,2925.74004,0.82098,2926.01683,0.82108,2926.29361,0.82118,2926.57036,0.82128,2926.84710,0.82138,2927.12383,0.82148,2927.40053,0.82158,2927.67723,0.82168,2927.95390,0.82178,2928.23056,0.82188,2928.50720,0.82198,2928.78382,0.82208,2929.06043,0.82218,2929.33702,0.82228,2929.61359,0.82238,2929.89015,0.82248,2930.16669,0.82258,2930.44321,0.82268,2930.71972,0.82278,2930.99621,0.82288,2931.27269,0.82298,2931.54914,0.82308,2931.82559,0.82318,2932.10201,0.82328,2932.37842,0.82338,2932.65481,0.82348,2932.93118,0.82358,2933.20754,0.82368,2933.48388,0.82378,2933.76021,0.82388,2934.03651,0.82398,2934.31281,0.82408,2934.58908,0.82418,2934.86534,0.82428,2935.14158,0.82438,2935.41781,0.82448,2935.69401,0.82458,2935.97021,0.82468,2936.24638,0.82478,2936.52254,0.82488,2936.79868,0.82498,2937.07481,0.82508,2937.35092,0.82518,2937.62701,0.82528,2937.90308,0.82538,2938.17914,0.82548,2938.45519,0.82558,2938.73121,0.82568,2939.00722,0.82578,2939.28321,0.82588,2939.55919,0.82598,2939.83515,0.82608,2940.11109,0.82618,2940.38702,0.82628,2940.66293,0.82638,2940.93882,0.82648,2941.21470,0.82658,2941.49056,0.82668,2941.76640,0.82678,2942.04223,0.82688,2942.31804,0.82698,2942.59383,0.82708,2942.86961,0.82718,2943.14537,0.82728,2943.42111,0.82738,2943.69684,0.82748,2943.97255,0.82758,2944.24825,0.82768,2944.52392,0.82778,2944.79959,0.82788,2945.07523,0.82798,2945.35086,0.82808,2945.62647,0.82818,2945.90206,0.82828,2946.17764,0.82838,2946.45321,0.82848,2946.72875,0.82858,2947.00428,0.82868,2947.27979,0.82878,2947.55529,0.82888,2947.83077,0.82898,2948.10623,0.82908,2948.38167,0.82918,2948.65710,0.82928,2948.93252,0.82938,2949.20791,0.82948,2949.48329,0.82958,2949.75866,0.82968,2950.03400,0.82978,2950.30933,0.82988,2950.58465,0.82998,2950.85994,0.83008,2951.13522,0.83018,2951.41049,0.83028,2951.68574,0.83038,2951.96097,0.83048,2952.23618,0.83058,2952.51138,0.83068,2952.78656,0.83078,2953.06172,0.83088,2953.33687,0.83098,2953.61200,0.83108,2953.88712,0.83118,2954.16222,0.83128,2954.43730,0.83138,2954.71237,0.83148,2954.98742,0.83158,2955.26245,0.83168,2955.53746,0.83178,2955.81246,0.83188,2956.08745,0.83198,2956.36241,0.83208,2956.63736,0.83218,2956.91230,0.83228,2957.18721,0.83238,2957.46211,0.83248,2957.73700,0.83258,2958.01187,0.83268,2958.28672,0.83278,2958.56155,0.83288,2958.83637,0.83298,2959.11117,0.83308,2959.38596,0.83318,2959.66072,0.83328,2959.93548,0.83338,2960.21021,0.83348,2960.48493,0.83358,2960.75963,0.83368,2961.03432,0.83378,2961.30899,0.83388,2961.58364,0.83398,2961.85828,0.83408,2962.13290,0.83418,2962.40750,0.83428,2962.68209,0.83438,2962.95666,0.83448,2963.23121,0.83458,2963.50575,0.83468,2963.78027,0.83478,2964.05478,0.83488,2964.32926,0.83498,2964.60374,0.83508,2964.87819,0.83518,2965.15263,0.83528,2965.42705,0.83538,2965.70146,0.83548,2965.97585,0.83558,2966.25022,0.83568,2966.52458,0.83578,2966.79892,0.83588,2967.07324,0.83598,2967.34755,0.83608,2967.62184,0.83618,2967.89611,0.83628,2968.17037,0.83638,2968.44461,0.83648,2968.71883,0.83658,2968.99304,0.83668,2969.26723,0.83678,2969.54141,0.83688,2969.81557,0.83698,2970.08971,0.83708,2970.36384,0.83718,2970.63795,0.83728,2970.91204,0.83738,2971.18612,0.83748,2971.46018,0.83758,2971.73422,0.83768,2972.00825,0.83778,2972.28226,0.83788,2972.55625,0.83798,2972.83023,0.83808,2973.10419,0.83818,2973.37813,0.83828,2973.65206,0.83838,2973.92598,0.83848,2974.19987,0.83858,2974.47375,0.83868,2974.74761,0.83878,2975.02146,0.83888,2975.29529,0.83898,2975.56910,0.83908,2975.84290,0.83918,2976.11668,0.83928,2976.39044,0.83938,2976.66419,0.83948,2976.93792,0.83958,2977.21164,0.83968,2977.48534,0.83978,2977.75902,0.83988,2978.03268,0.83998,2978.30633,0.84008,2978.57997,0.84018,2978.85358,0.84028,2979.12718,0.84038,2979.40077,0.84048,2979.67433,0.84058,2979.94788,0.84068,2980.22142,0.84078,2980.49494,0.84088,2980.76844,0.84098,2981.04192,0.84108,2981.31539,0.84118,2981.58884,0.84128,2981.86228,0.84138,2982.13570,0.84148,2982.40910,0.84158,2982.68249,0.84168,2982.95586,0.84178,2983.22921,0.84188,2983.50255,0.84198,2983.77587,0.84208,2984.04917,0.84218,2984.32246,0.84228,2984.59573,0.84238,2984.86899,0.84248,2985.14223,0.84258,2985.41545,0.84268,2985.68866,0.84278,2985.96185,0.84288,2986.23502,0.84298,2986.50818,0.84308,2986.78132,0.84318,2987.05444,0.84328,2987.32755,0.84338,2987.60064,0.84348,2987.87372,0.84358,2988.14678,0.84368,2988.41982,0.84378,2988.69284,0.84388,2988.96585,0.84398,2989.23885,0.84408,2989.51182,0.84418,2989.78478,0.84428,2990.05773,0.84438,2990.33066,0.84448,2990.60357,0.84458,2990.87646,0.84468,2991.14934,0.84478,2991.42220,0.84488,2991.69505,0.84498,2991.96788,0.84508,2992.24069,0.84518,2992.51349,0.84528,2992.78627,0.84538,2993.05903,0.84548,2993.33178,0.84558,2993.60451,0.84568,2993.87723,0.84578,2994.14993,0.84588,2994.42261,0.84598,2994.69528,0.84608,2994.96793,0.84618,2995.24056,0.84628,2995.51318,0.84638,2995.78578,0.84648,2996.05836,0.84658,2996.33093,0.84668,2996.60348,0.84678,2996.87602,0.84688,2997.14853,0.84698,2997.42104,0.84708,2997.69352,0.84718,2997.96599,0.84728,2998.23845,0.84738,2998.51088,0.84748,2998.78330,0.84758,2999.05571,0.84768,2999.32810,0.84778,2999.60047,0.84788,2999.87282,0.84798,3000.14516,0.84808,3000.41749,0.84818,3000.68979,0.84828,3000.96208,0.84838,3001.23436,0.84848,3001.50662,0.84858,3001.77886,0.84868,3002.05108,0.84878,3002.32329,0.84888,3002.59548,0.84898,3002.86766,0.84908,3003.13982,0.84918,3003.41196,0.84928,3003.68409,0.84938,3003.95620,0.84948,3004.22829,0.84958,3004.50037,0.84968,3004.77243,0.84978,3005.04448,0.84988,3005.31651,0.84998,3005.58852,0.85009,3005.86052,0.85019,3006.13250,0.85029,3006.40446,0.85039,3006.67641,0.85049,3006.94834,0.85059,3007.22026,0.85069,3007.49216,0.85079,3007.76404,0.85089,3008.03590,0.85099,3008.30775,0.85109,3008.57959,0.85119,3008.85141,0.85129,3009.12321,0.85139,3009.39499,0.85149,3009.66676,0.85159,3009.93851,0.85169,3010.21025,0.85179,3010.48197,0.85189,3010.75367,0.85199,3011.02536,0.85209,3011.29703,0.85219,3011.56868,0.85229,3011.84032,0.85239,3012.11194,0.85249,3012.38355,0.85259,3012.65514,0.85269,3012.92671,0.85279,3013.19827,0.85289,3013.46981,0.85299,3013.74133,0.85309,3014.01284,0.85319,3014.28433,0.85329,3014.55581,0.85339,3014.82727,0.85349,3015.09871,0.85359,3015.37014,0.85369,3015.64155,0.85379,3015.91294,0.85389,3016.18432,0.85399,3016.45568,0.85409,3016.72703,0.85419,3016.99836,0.85429,3017.26967,0.85439,3017.54097,0.85449,3017.81225,0.85459,3018.08351,0.85469,3018.35476,0.85479,3018.62599,0.85489,3018.89721,0.85499,3019.16841,0.85509,3019.43959,0.85519,3019.71076,0.85529,3019.98191,0.85539,3020.25304,0.85549,3020.52416,0.85559,3020.79526,0.85569,3021.06635,0.85579,3021.33742,0.85589,3021.60847,0.85599,3021.87951,0.85609,3022.15053,0.85619,3022.42153,0.85629,3022.69252,0.85639,3022.96349,0.85649,3023.23445,0.85659,3023.50539,0.85669,3023.77631,0.85679,3024.04722,0.85689,3024.31811,0.85699,3024.58898,0.85709,3024.85984,0.85719,3025.13068,0.85729,3025.40151,0.85739,3025.67232,0.85749,3025.94311,0.85759,3026.21389,0.85769,3026.48465,0.85779,3026.75539,0.85789,3027.02612,0.85799,3027.29683,0.85809,3027.56753,0.85819,3027.83821,0.85829,3028.10887,0.85839,3028.37952,0.85849,3028.65015,0.85859,3028.92077,0.85869,3029.19137,0.85879,3029.46195,0.85889,3029.73252,0.85899,3030.00307,0.85909,3030.27360,0.85919,3030.54412,0.85929,3030.81462,0.85939,3031.08511,0.85949,3031.35558,0.85959,3031.62603,0.85969,3031.89647,0.85979,3032.16689,0.85989,3032.43729,0.85999,3032.70768,0.86009,3032.97805,0.86019,3033.24841,0.86029,3033.51875,0.86039,3033.78907,0.86049,3034.05938,0.86059,3034.32967,0.86069,3034.59995,0.86079,3034.87020,0.86089,3035.14045,0.86099,3035.41067,0.86109,3035.68088,0.86119,3035.95108,0.86129,3036.22126,0.86139,3036.49142,0.86149,3036.76156,0.86159,3037.03169,0.86169,3037.30181,0.86179,3037.57190,0.86189,3037.84199,0.86199,3038.11205,0.86209,3038.38210,0.86219,3038.65213,0.86229,3038.92215,0.86239,3039.19215,0.86249,3039.46213,0.86259,3039.73210,0.86269,3040.00205,0.86279,3040.27199,0.86289,3040.54191,0.86299,3040.81181,0.86309,3041.08170,0.86319,3041.35157,0.86329,3041.62142,0.86339,3041.89126,0.86349,3042.16108,0.86359,3042.43089,0.86369,3042.70068,0.86379,3042.97046,0.86389,3043.24021,0.86399,3043.50996,0.86409,3043.77968,0.86419,3044.04939,0.86429,3044.31908,0.86439,3044.58876,0.86449,3044.85842,0.86459,3045.12807,0.86469,3045.39770,0.86479,3045.66731,0.86489,3045.93691,0.86499,3046.20649,0.86509,3046.47605,0.86519,3046.74560,0.86529,3047.01513,0.86539,3047.28465,0.86549,3047.55415,0.86559,3047.82363,0.86569,3048.09310,0.86579,3048.36255,0.86589,3048.63199,0.86599,3048.90140,0.86609,3049.17081,0.86619,3049.44019,0.86629,3049.70957,0.86639,3049.97892,0.86649,3050.24826,0.86659,3050.51758,0.86669,3050.78689,0.86679,3051.05618,0.86689,3051.32545,0.86699,3051.59471,0.86709,3051.86395,0.86719,3052.13318,0.86729,3052.40239,0.86739,3052.67158,0.86749,3052.94076,0.86759,3053.20992,0.86769,3053.47907,0.86779,3053.74820,0.86789,3054.01731,0.86799,3054.28641,0.86809,3054.55549,0.86819,3054.82455,0.86829,3055.09360,0.86839,3055.36264,0.86849,3055.63165,0.86859,3055.90065,0.86869,3056.16964,0.86879,3056.43861,0.86889,3056.70756,0.86899,3056.97649,0.86909,3057.24541,0.86919,3057.51432,0.86929,3057.78321,0.86939,3058.05208,0.86949,3058.32093,0.86959,3058.58977,0.86969,3058.85860,0.86979,3059.12741,0.86989,3059.39620,0.86999,3059.66497,0.87009,3059.93373,0.87019,3060.20248,0.87029,3060.47120,0.87039,3060.73991,0.87049,3061.00861,0.87059,3061.27729,0.87069,3061.54595,0.87079,3061.81460,0.87089,3062.08323,0.87099,3062.35184,0.87109,3062.62044,0.87119,3062.88903,0.87129,3063.15759,0.87139,3063.42614,0.87149,3063.69468,0.87159,3063.96320,0.87169,3064.23170,0.87179,3064.50019,0.87189,3064.76866,0.87199,3065.03711,0.87209,3065.30555,0.87219,3065.57397,0.87229,3065.84238,0.87239,3066.11077,0.87249,3066.37914,0.87259,3066.64750,0.87269,3066.91584,0.87279,3067.18417,0.87289,3067.45248,0.87299,3067.72077,0.87309,3067.98905,0.87319,3068.25731,0.87329,3068.52556,0.87339,3068.79379,0.87349,3069.06200,0.87359,3069.33020,0.87369,3069.59838,0.87379,3069.86655,0.87389,3070.13470,0.87399,3070.40283,0.87409,3070.67095,0.87419,3070.93905,0.87429,3071.20713,0.87439,3071.47520,0.87449,3071.74326,0.87459,3072.01129,0.87469,3072.27932,0.87479,3072.54732,0.87489,3072.81531,0.87499,3073.08328,0.87509,3073.35124,0.87519,3073.61918,0.87529,3073.88711,0.87539,3074.15502,0.87549,3074.42291,0.87559,3074.69079,0.87569,3074.95865,0.87579,3075.22650,0.87589,3075.49433,0.87599,3075.76214,0.87609,3076.02994,0.87619,3076.29772,0.87629,3076.56548,0.87639,3076.83323,0.87649,3077.10097,0.87659,3077.36868,0.87669,3077.63638,0.87679,3077.90407,0.87689,3078.17174,0.87699,3078.43939,0.87709,3078.70703,0.87719,3078.97465,0.87729,3079.24226,0.87739,3079.50985,0.87749,3079.77742,0.87759,3080.04498,0.87769,3080.31252,0.87779,3080.58004,0.87789,3080.84755,0.87799,3081.11505,0.87809,3081.38253,0.87819,3081.64999,0.87829,3081.91743,0.87839,3082.18486,0.87849,3082.45228,0.87859,3082.71967,0.87869,3082.98706,0.87879,3083.25442,0.87889,3083.52177,0.87899,3083.78911,0.87909,3084.05642,0.87919,3084.32373,0.87929,3084.59101,0.87939,3084.85828,0.87949,3085.12554,0.87959,3085.39277,0.87969,3085.66000,0.87979,3085.92720,0.87989,3086.19439,0.87999,3086.46157,0.88009,3086.72872,0.88019,3086.99587,0.88029,3087.26299,0.88039,3087.53010,0.88049,3087.79720,0.88059,3088.06428,0.88069,3088.33134,0.88079,3088.59839,0.88089,3088.86542,0.88099,3089.13243,0.88109,3089.39943,0.88119,3089.66641,0.88129,3089.93338,0.88139,3090.20033,0.88149,3090.46727,0.88159,3090.73419,0.88169,3091.00109,0.88179,3091.26798,0.88189,3091.53485,0.88199,3091.80170,0.88209,3092.06854,0.88219,3092.33537,0.88229,3092.60217,0.88239,3092.86897,0.88249,3093.13574,0.88259,3093.40250,0.88269,3093.66925,0.88279,3093.93597,0.88289,3094.20269,0.88299,3094.46938,0.88309,3094.73606,0.88319,3095.00273,0.88329,3095.26937,0.88339,3095.53601,0.88349,3095.80262,0.88359,3096.06922,0.88369,3096.33581,0.88379,3096.60238,0.88389,3096.86893,0.88399,3097.13547,0.88409,3097.40199,0.88419,3097.66849,0.88429,3097.93498,0.88439,3098.20146,0.88449,3098.46791,0.88459,3098.73436,0.88469,3099.00078,0.88479,3099.26719,0.88489,3099.53359,0.88499,3099.79996,0.88509,3100.06633,0.88519,3100.33267,0.88529,3100.59900,0.88539,3100.86532,0.88549,3101.13162,0.88559,3101.39790,0.88569,3101.66416,0.88579,3101.93042,0.88589,3102.19665,0.88599,3102.46287,0.88609,3102.72907,0.88619,3102.99526,0.88629,3103.26143,0.88639,3103.52759,0.88649,3103.79373,0.88659,3104.05985,0.88669,3104.32596,0.88679,3104.59205,0.88689,3104.85813,0.88699,3105.12419,0.88709,3105.39023,0.88719,3105.65626,0.88729,3105.92227,0.88739,3106.18827,0.88749,3106.45425,0.88759,3106.72022,0.88769,3106.98617,0.88779,3107.25210,0.88789,3107.51802,0.88799,3107.78392,0.88809,3108.04980,0.88819,3108.31567,0.88829,3108.58153,0.88839,3108.84737,0.88849,3109.11319,0.88859,3109.37899,0.88869,3109.64479,0.88879,3109.91056,0.88889,3110.17632,0.88899,3110.44206,0.88909,3110.70779,0.88919,3110.97350,0.88929,3111.23920,0.88939,3111.50488,0.88949,3111.77054,0.88959,3112.03619,0.88969,3112.30182,0.88979,3112.56744,0.88989,3112.83304,0.88999,3113.09862,0.89009,3113.36419,0.89019,3113.62974,0.89029,3113.89528,0.89039,3114.16080,0.89049,3114.42631,0.89059,3114.69180,0.89069,3114.95727,0.89079,3115.22273,0.89089,3115.48817,0.89099,3115.75360,0.89109,3116.01901,0.89119,3116.28440,0.89129,3116.54978,0.89139,3116.81514,0.89149,3117.08049,0.89159,3117.34582,0.89169,3117.61114,0.89179,3117.87644,0.89189,3118.14172,0.89199,3118.40699,0.89209,3118.67224,0.89219,3118.93748,0.89229,3119.20270,0.89239,3119.46790,0.89249,3119.73309,0.89259,3119.99827,0.89269,3120.26342,0.89279,3120.52857,0.89289,3120.79369,0.89299,3121.05880,0.89309,3121.32390,0.89319,3121.58897,0.89329,3121.85404,0.89339,3122.11908,0.89349,3122.38411,0.89359,3122.64913,0.89369,3122.91413,0.89379,3123.17911,0.89389,3123.44408,0.89399,3123.70903,0.89409,3123.97397,0.89419,3124.23889,0.89429,3124.50379,0.89439,3124.76868,0.89449,3125.03356,0.89459,3125.29841,0.89469,3125.56325,0.89479,3125.82808,0.89489,3126.09289,0.89499,3126.35768,0.89509,3126.62246,0.89519,3126.88723,0.89529,3127.15197,0.89539,3127.41670,0.89549,3127.68142,0.89559,3127.94612,0.89569,3128.21080,0.89579,3128.47547,0.89589,3128.74012,0.89599,3129.00476,0.89609,3129.26938,0.89619,3129.53398,0.89629,3129.79857,0.89639,3130.06315,0.89649,3130.32770,0.89659,3130.59225,0.89669,3130.85677,0.89679,3131.12128,0.89689,3131.38578,0.89699,3131.65026,0.89709,3131.91472,0.89719,3132.17917,0.89729,3132.44360,0.89739,3132.70801,0.89749,3132.97241,0.89759,3133.23680,0.89769,3133.50117,0.89779,3133.76552,0.89789,3134.02986,0.89799,3134.29418,0.89809,3134.55848,0.89819,3134.82277,0.89829,3135.08705,0.89839,3135.35130,0.89849,3135.61555,0.89859,3135.87977,0.89869,3136.14398,0.89879,3136.40818,0.89889,3136.67236,0.89899,3136.93652,0.89909,3137.20067,0.89919,3137.46480,0.89929,3137.72892,0.89939,3137.99302,0.89949,3138.25711,0.89959,3138.52118,0.89969,3138.78523,0.89979,3139.04927,0.89989,3139.31329,0.89999,3139.57730,0.90009,3139.84129,0.90019,3140.10526,0.90029,3140.36922,0.90039,3140.63317,0.90049,3140.89709,0.90059,3141.16101,0.90069,3141.42490,0.90079,3141.68878,0.90089,3141.95265,0.90099,3142.21650,0.90109,3142.48033,0.90119,3142.74415,0.90129,3143.00795,0.90139,3143.27174,0.90149,3143.53551,0.90159,3143.79926,0.90169,3144.06300,0.90179,3144.32673,0.90189,3144.59044,0.90199,3144.85413,0.90209,3145.11780,0.90219,3145.38147,0.90229,3145.64511,0.90239,3145.90874,0.90249,3146.17235,0.90259,3146.43595,0.90269,3146.69953,0.90279,3146.96310,0.90289,3147.22665,0.90299,3147.49019,0.90309,3147.75371,0.90319,3148.01721,0.90329,3148.28070,0.90339,3148.54417,0.90349,3148.80763,0.90359,3149.07107,0.90369,3149.33450,0.90379,3149.59791,0.90389,3149.86130,0.90399,3150.12468,0.90409,3150.38804,0.90419,3150.65139,0.90429,3150.91472,0.90439,3151.17804,0.90449,3151.44134,0.90459,3151.70462,0.90469,3151.96789,0.90479,3152.23114,0.90489,3152.49438,0.90499,3152.75760,0.90509,3153.02081,0.90519,3153.28400,0.90529,3153.54717,0.90539,3153.81033,0.90549,3154.07348,0.90559,3154.33661,0.90569,3154.59972,0.90579,3154.86281,0.90589,3155.12589,0.90599,3155.38896,0.90609,3155.65201,0.90619,3155.91504,0.90629,3156.17806,0.90639,3156.44106,0.90649,3156.70405,0.90659,3156.96702,0.90669,3157.22998,0.90679,3157.49292,0.90689,3157.75584,0.90699,3158.01875,0.90709,3158.28165,0.90719,3158.54452,0.90729,3158.80738,0.90739,3159.07023,0.90749,3159.33306,0.90759,3159.59588,0.90769,3159.85868,0.90779,3160.12146,0.90789,3160.38423,0.90799,3160.64698,0.90809,3160.90972,0.90819,3161.17244,0.90829,3161.43514,0.90839,3161.69783,0.90849,3161.96051,0.90859,3162.22317,0.90869,3162.48581,0.90879,3162.74844,0.90889,3163.01105,0.90899,3163.27364,0.90909,3163.53622,0.90919,3163.79879,0.90929,3164.06134,0.90939,3164.32387,0.90949,3164.58639,0.90959,3164.84889,0.90969,3165.11138,0.90979,3165.37385,0.90989,3165.63631,0.90999,3165.89875,0.91009,3166.16117,0.91019,3166.42358,0.91029,3166.68597,0.91039,3166.94835,0.91049,3167.21071,0.91059,3167.47306,0.91069,3167.73539,0.91079,3167.99771,0.91089,3168.26001,0.91099,3168.52229,0.91109,3168.78456,0.91119,3169.04681,0.91129,3169.30905,0.91139,3169.57127,0.91149,3169.83348,0.91159,3170.09567,0.91169,3170.35784,0.91179,3170.62000,0.91189,3170.88214,0.91199,3171.14427,0.91209,3171.40638,0.91219,3171.66848,0.91229,3171.93056,0.91239,3172.19263,0.91249,3172.45468,0.91259,3172.71671,0.91269,3172.97873,0.91279,3173.24074,0.91289,3173.50272,0.91299,3173.76470,0.91309,3174.02665,0.91319,3174.28859,0.91329,3174.55052,0.91339,3174.81243,0.91349,3175.07432,0.91359,3175.33620,0.91369,3175.59807,0.91379,3175.85991,0.91389,3176.12175,0.91399,3176.38356,0.91409,3176.64536,0.91419,3176.90715,0.91429,3177.16892,0.91439,3177.43067,0.91449,3177.69241,0.91459,3177.95414,0.91469,3178.21584,0.91479,3178.47754,0.91489,3178.73921,0.91499,3179.00087,0.91509,3179.26252,0.91519,3179.52415,0.91529,3179.78576,0.91539,3180.04736,0.91549,3180.30895,0.91559,3180.57051,0.91569,3180.83207,0.91579,3181.09360,0.91589,3181.35512,0.91599,3181.61663,0.91609,3181.87812,0.91619,3182.13959,0.91629,3182.40105,0.91639,3182.66250,0.91649,3182.92392,0.91659,3183.18534,0.91669,3183.44673,0.91679,3183.70811,0.91689,3183.96948,0.91699,3184.23083,0.91709,3184.49217,0.91719,3184.75348,0.91729,3185.01479,0.91739,3185.27608,0.91749,3185.53735,0.91759,3185.79861,0.91769,3186.05985,0.91779,3186.32107,0.91789,3186.58228,0.91799,3186.84348,0.91809,3187.10466,0.91819,3187.36582,0.91829,3187.62697,0.91839,3187.88811,0.91849,3188.14922,0.91859,3188.41032,0.91869,3188.67141,0.91879,3188.93248,0.91889,3189.19354,0.91899,3189.45458,0.91909,3189.71560,0.91919,3189.97661,0.91929,3190.23760,0.91939,3190.49858,0.91949,3190.75954,0.91959,3191.02049,0.91969,3191.28142,0.91979,3191.54234,0.91989,3191.80324,0.91999,3192.06412,0.92009,3192.32499,0.92019,3192.58585,0.92029,3192.84669,0.92039,3193.10751,0.92049,3193.36832,0.92059,3193.62911,0.92069,3193.88989,0.92079,3194.15065,0.92089,3194.41139,0.92099,3194.67212,0.92109,3194.93284,0.92119,3195.19354,0.92129,3195.45422,0.92139,3195.71489,0.92149,3195.97554,0.92159,3196.23618,0.92169,3196.49680,0.92179,3196.75741,0.92189,3197.01800,0.92199,3197.27857,0.92209,3197.53913,0.92219,3197.79968,0.92229,3198.06021,0.92239,3198.32072,0.92249,3198.58122,0.92259,3198.84170,0.92269,3199.10217,0.92279,3199.36262,0.92289,3199.62305,0.92299,3199.88348,0.92309,3200.14388,0.92319,3200.40427,0.92329,3200.66464,0.92339,3200.92500,0.92349,3201.18535,0.92359,3201.44567,0.92369,3201.70599,0.92379,3201.96628,0.92389,3202.22657,0.92399,3202.48683,0.92409,3202.74708,0.92419,3203.00732,0.92429,3203.26754,0.92439,3203.52774,0.92449,3203.78793,0.92459,3204.04810,0.92469,3204.30826,0.92479,3204.56841,0.92489,3204.82853,0.92499,3205.08864,0.92509,3205.34874,0.92519,3205.60882,0.92529,3205.86889,0.92539,3206.12894,0.92549,3206.38897,0.92559,3206.64899,0.92569,3206.90899,0.92579,3207.16898,0.92589,3207.42896,0.92599,3207.68891,0.92609,3207.94886,0.92619,3208.20878,0.92629,3208.46869,0.92639,3208.72859,0.92649,3208.98847,0.92659,3209.24833,0.92669,3209.50818,0.92679,3209.76802,0.92689,3210.02784,0.92699,3210.28764,0.92709,3210.54743,0.92719,3210.80720,0.92729,3211.06696,0.92739,3211.32670,0.92749,3211.58643,0.92759,3211.84614,0.92769,3212.10583,0.92779,3212.36551,0.92789,3212.62518,0.92799,3212.88483,0.92809,3213.14446,0.92819,3213.40408,0.92829,3213.66368,0.92839,3213.92327,0.92849,3214.18284,0.92859,3214.44240,0.92869,3214.70194,0.92879,3214.96146,0.92889,3215.22097,0.92899,3215.48047,0.92909,3215.73995,0.92919,3215.99941,0.92929,3216.25886,0.92939,3216.51830,0.92949,3216.77772,0.92959,3217.03712,0.92969,3217.29651,0.92979,3217.55588,0.92989,3217.81523,0.92999,3218.07458,0.93009,3218.33390,0.93019,3218.59321,0.93029,3218.85251,0.93039,3219.11179,0.93049,3219.37105,0.93059,3219.63030,0.93069,3219.88954,0.93079,3220.14875,0.93089,3220.40796,0.93099,3220.66714,0.93109,3220.92632,0.93119,3221.18547,0.93129,3221.44462,0.93139,3221.70374,0.93149,3221.96285,0.93159,3222.22195,0.93169,3222.48103,0.93179,3222.74009,0.93189,3222.99914,0.93199,3223.25818,0.93209,3223.51720,0.93219,3223.77620,0.93229,3224.03519,0.93239,3224.29416,0.93249,3224.55312,0.93259,3224.81206,0.93269,3225.07098,0.93279,3225.32990,0.93289,3225.58879,0.93299,3225.84767,0.93309,3226.10654,0.93319,3226.36539,0.93329,3226.62422,0.93339,3226.88304,0.93349,3227.14185,0.93359,3227.40063,0.93369,3227.65941,0.93379,3227.91816,0.93389,3228.17691,0.93399,3228.43563,0.93409,3228.69435,0.93419,3228.95304,0.93429,3229.21172,0.93439,3229.47039,0.93449,3229.72904,0.93459,3229.98768,0.93469,3230.24630,0.93479,3230.50490,0.93489,3230.76349,0.93499,3231.02206,0.93509,3231.28062,0.93519,3231.53916,0.93529,3231.79769,0.93539,3232.05620,0.93549,3232.31470,0.93559,3232.57318,0.93569,3232.83165,0.93579,3233.09010,0.93589,3233.34854,0.93599,3233.60696,0.93609,3233.86536,0.93619,3234.12375,0.93629,3234.38213,0.93639,3234.64049,0.93649,3234.89883,0.93659,3235.15716,0.93669,3235.41548,0.93679,3235.67377,0.93689,3235.93206,0.93699,3236.19032,0.93709,3236.44858,0.93719,3236.70681,0.93729,3236.96504,0.93739,3237.22324,0.93749,3237.48143,0.93759,3237.73961,0.93769,3237.99777,0.93779,3238.25592,0.93789,3238.51405,0.93799,3238.77216,0.93809,3239.03026,0.93819,3239.28834,0.93829,3239.54641,0.93839,3239.80447,0.93849,3240.06250,0.93859,3240.32053,0.93869,3240.57854,0.93879,3240.83653,0.93889,3241.09450,0.93899,3241.35247,0.93909,3241.61041,0.93919,3241.86834,0.93929,3242.12626,0.93939,3242.38416,0.93949,3242.64205,0.93959,3242.89992,0.93969,3243.15777,0.93979,3243.41561,0.93989,3243.67344,0.93999,3243.93125,0.94009,3244.18904,0.94019,3244.44682,0.94029,3244.70458,0.94039,3244.96233,0.94049,3245.22006,0.94059,3245.47778,0.94069,3245.73548,0.94079,3245.99317,0.94089,3246.25084,0.94099,3246.50850,0.94109,3246.76614,0.94119,3247.02377,0.94129,3247.28138,0.94139,3247.53897,0.94149,3247.79655,0.94159,3248.05412,0.94169,3248.31167,0.94179,3248.56920,0.94189,3248.82672,0.94199,3249.08423,0.94209,3249.34172,0.94219,3249.59919,0.94229,3249.85665,0.94239,3250.11409,0.94249,3250.37152,0.94259,3250.62893,0.94269,3250.88633,0.94279,3251.14371,0.94289,3251.40108,0.94299,3251.65843,0.94309,3251.91577,0.94319,3252.17309,0.94329,3252.43039,0.94339,3252.68768,0.94349,3252.94496,0.94359,3253.20222,0.94369,3253.45947,0.94379,3253.71670,0.94389,3253.97391,0.94399,3254.23111,0.94409,3254.48829,0.94419,3254.74546,0.94429,3255.00262,0.94439,3255.25976,0.94449,3255.51688,0.94459,3255.77399,0.94469,3256.03108,0.94479,3256.28816,0.94489,3256.54522,0.94499,3256.80227,0.94509,3257.05930,0.94519,3257.31632,0.94529,3257.57332,0.94539,3257.83031,0.94549,3258.08728,0.94559,3258.34423,0.94569,3258.60117,0.94579,3258.85810,0.94589,3259.11501,0.94599,3259.37191,0.94609,3259.62879,0.94619,3259.88565,0.94629,3260.14250,0.94639,3260.39934,0.94649,3260.65616,0.94659,3260.91296,0.94669,3261.16975,0.94679,3261.42652,0.94689,3261.68328,0.94699,3261.94002,0.94709,3262.19675,0.94719,3262.45347,0.94729,3262.71016,0.94739,3262.96685,0.94749,3263.22351,0.94759,3263.48017,0.94769,3263.73680,0.94779,3263.99343,0.94789,3264.25003,0.94799,3264.50662,0.94809,3264.76320,0.94819,3265.01976,0.94829,3265.27631,0.94839,3265.53284,0.94849,3265.78936,0.94859,3266.04586,0.94869,3266.30234,0.94879,3266.55881,0.94889,3266.81527,0.94899,3267.07171,0.94909,3267.32813,0.94919,3267.58454,0.94929,3267.84094,0.94939,3268.09732,0.94949,3268.35368,0.94959,3268.61003,0.94969,3268.86636,0.94979,3269.12268,0.94989,3269.37898,0.94999,3269.63527,0.95010,3269.89155,0.95020,3270.14780,0.95030,3270.40405,0.95040,3270.66028,0.95050,3270.91649,0.95060,3271.17269,0.95070,3271.42887,0.95080,3271.68503,0.95090,3271.94119,0.95100,3272.19732,0.95110,3272.45345,0.95120,3272.70955,0.95130,3272.96564,0.95140,3273.22172,0.95150,3273.47778,0.95160,3273.73383,0.95170,3273.98986,0.95180,3274.24587,0.95190,3274.50188,0.95200,3274.75786,0.95210,3275.01383,0.95220,3275.26979,0.95230,3275.52573,0.95240,3275.78165,0.95250,3276.03756,0.95260,3276.29346,0.95270,3276.54934,0.95280,3276.80520,0.95290,3277.06105,0.95300,3277.31689,0.95310,3277.57270,0.95320,3277.82851,0.95330,3278.08430,0.95340,3278.34007,0.95350,3278.59583,0.95360,3278.85157,0.95370,3279.10730,0.95380,3279.36302,0.95390,3279.61871,0.95400,3279.87440,0.95410,3280.13007,0.95420,3280.38572,0.95430,3280.64136,0.95440,3280.89698,0.95450,3281.15259,0.95460,3281.40818,0.95470,3281.66376,0.95480,3281.91932,0.95490,3282.17487,0.95500,3282.43040,0.95510,3282.68592,0.95520,3282.94142,0.95530,3283.19691,0.95540,3283.45238,0.95550,3283.70783,0.95560,3283.96328,0.95570,3284.21870,0.95580,3284.47411,0.95590,3284.72951,0.95600,3284.98489,0.95610,3285.24026,0.95620,3285.49561,0.95630,3285.75095,0.95640,3286.00627,0.95650,3286.26157,0.95660,3286.51686,0.95670,3286.77214,0.95680,3287.02740,0.95690,3287.28265,0.95700,3287.53788,0.95710,3287.79309,0.95720,3288.04829,0.95730,3288.30348,0.95740,3288.55865,0.95750,3288.81380,0.95760,3289.06894,0.95770,3289.32407,0.95780,3289.57918,0.95790,3289.83427,0.95800,3290.08935,0.95810,3290.34442,0.95820,3290.59947,0.95830,3290.85450,0.95840,3291.10952,0.95850,3291.36453,0.95860,3291.61952,0.95870,3291.87449,0.95880,3292.12945,0.95890,3292.38440,0.95900,3292.63933,0.95910,3292.89424,0.95920,3293.14914,0.95930,3293.40402,0.95940,3293.65889,0.95950,3293.91375,0.95960,3294.16859,0.95970,3294.42341,0.95980,3294.67822,0.95990,3294.93301,0.96000,3295.18779,0.96010,3295.44256,0.96020,3295.69730,0.96030,3295.95204,0.96040,3296.20676,0.96050,3296.46146,0.96060,3296.71615,0.96070,3296.97082,0.96080,3297.22548,0.96090,3297.48013,0.96100,3297.73475,0.96110,3297.98937,0.96120,3298.24397,0.96130,3298.49855,0.96140,3298.75312,0.96150,3299.00767,0.96160,3299.26221,0.96170,3299.51673,0.96180,3299.77124,0.96190,3300.02574,0.96200,3300.28021,0.96210,3300.53468,0.96220,3300.78913,0.96230,3301.04356,0.96240,3301.29798,0.96250,3301.55238,0.96260,3301.80677,0.96270,3302.06114,0.96280,3302.31550,0.96290,3302.56984,0.96300,3302.82417,0.96310,3303.07849,0.96320,3303.33278,0.96330,3303.58707,0.96340,3303.84134,0.96350,3304.09559,0.96360,3304.34983,0.96370,3304.60405,0.96380,3304.85826,0.96390,3305.11245,0.96400,3305.36663,0.96410,3305.62079,0.96420,3305.87494,0.96430,3306.12907,0.96440,3306.38319,0.96450,3306.63730,0.96460,3306.89139,0.96470,3307.14546,0.96480,3307.39952,0.96490,3307.65356,0.96500,3307.90759,0.96510,3308.16160,0.96520,3308.41560,0.96530,3308.66958,0.96540,3308.92355,0.96550,3309.17751,0.96560,3309.43144,0.96570,3309.68537,0.96580,3309.93928,0.96590,3310.19317,0.96600,3310.44705,0.96610,3310.70091,0.96620,3310.95476,0.96630,3311.20860,0.96640,3311.46241,0.96650,3311.71622,0.96660,3311.97001,0.96670,3312.22378,0.96680,3312.47754,0.96690,3312.73128,0.96700,3312.98501,0.96710,3313.23873,0.96720,3313.49243,0.96730,3313.74611,0.96740,3313.99978,0.96750,3314.25343,0.96760,3314.50707,0.96770,3314.76070,0.96780,3315.01431,0.96790,3315.26790,0.96800,3315.52148,0.96810,3315.77504,0.96820,3316.02859,0.96830,3316.28213,0.96840,3316.53565,0.96850,3316.78915,0.96860,3317.04264,0.96870,3317.29612,0.96880,3317.54958,0.96890,3317.80302,0.96900,3318.05645,0.96910,3318.30987,0.96920,3318.56327,0.96930,3318.81665,0.96940,3319.07002,0.96950,3319.32338,0.96960,3319.57672,0.96970,3319.83005,0.96980,3320.08336,0.96990,3320.33665,0.97000,3320.58993,0.97010,3320.84320,0.97020,3321.09645,0.97030,3321.34968,0.97040,3321.60291,0.97050,3321.85611,0.97060,3322.10930,0.97070,3322.36248,0.97080,3322.61564,0.97090,3322.86879,0.97100,3323.12192,0.97110,3323.37504,0.97120,3323.62814,0.97130,3323.88122,0.97140,3324.13430,0.97150,3324.38735,0.97160,3324.64039,0.97170,3324.89342,0.97180,3325.14643,0.97190,3325.39943,0.97200,3325.65241,0.97210,3325.90538,0.97220,3326.15833,0.97230,3326.41127,0.97240,3326.66419,0.97250,3326.91710,0.97260,3327.17000,0.97270,3327.42287,0.97280,3327.67574,0.97290,3327.92859,0.97300,3328.18142,0.97310,3328.43424,0.97320,3328.68704,0.97330,3328.93983,0.97340,3329.19260,0.97350,3329.44536,0.97360,3329.69811,0.97370,3329.95084,0.97380,3330.20355,0.97390,3330.45625,0.97400,3330.70893,0.97410,3330.96160,0.97420,3331.21426,0.97430,3331.46690,0.97440,3331.71952,0.97450,3331.97213,0.97460,3332.22473,0.97470,3332.47731,0.97480,3332.72988,0.97490,3332.98243,0.97500,3333.23496,0.97510,3333.48748,0.97520,3333.73999,0.97530,3333.99248,0.97540,3334.24496,0.97550,3334.49742,0.97560,3334.74987,0.97570,3335.00230,0.97580,3335.25471,0.97590,3335.50712,0.97600,3335.75950,0.97610,3336.01188,0.97620,3336.26423,0.97630,3336.51658,0.97640,3336.76890,0.97650,3337.02122,0.97660,3337.27351,0.97670,3337.52580,0.97680,3337.77807,0.97690,3338.03032,0.97700,3338.28256,0.97710,3338.53478,0.97720,3338.78699,0.97730,3339.03919,0.97740,3339.29136,0.97750,3339.54353,0.97760,3339.79568,0.97770,3340.04781,0.97780,3340.29993,0.97790,3340.55204,0.97800,3340.80413,0.97810,3341.05620,0.97820,3341.30826,0.97830,3341.56031,0.97840,3341.81234,0.97850,3342.06436,0.97860,3342.31636,0.97870,3342.56835,0.97880,3342.82032,0.97890,3343.07227,0.97900,3343.32422,0.97910,3343.57614,0.97920,3343.82805,0.97930,3344.07995,0.97940,3344.33184,0.97950,3344.58370,0.97960,3344.83556,0.97970,3345.08739,0.97980,3345.33922,0.97990,3345.59103,0.98000,3345.84282,0.98010,3346.09460,0.98020,3346.34636,0.98030,3346.59811,0.98040,3346.84985,0.98050,3347.10157,0.98060,3347.35327,0.98070,3347.60496,0.98080,3347.85664,0.98090,3348.10830,0.98100,3348.35994,0.98110,3348.61157,0.98120,3348.86319,0.98130,3349.11479,0.98140,3349.36638,0.98150,3349.61795,0.98160,3349.86951,0.98170,3350.12105,0.98180,3350.37258,0.98190,3350.62409,0.98200,3350.87559,0.98210,3351.12707,0.98220,3351.37854,0.98230,3351.62999,0.98240,3351.88143,0.98250,3352.13285,0.98260,3352.38426,0.98270,3352.63566,0.98280,3352.88704,0.98290,3353.13840,0.98300,3353.38975,0.98310,3353.64108,0.98320,3353.89240,0.98330,3354.14371,0.98340,3354.39500,0.98350,3354.64628,0.98360,3354.89754,0.98370,3355.14878,0.98380,3355.40002,0.98390,3355.65123,0.98400,3355.90244,0.98410,3356.15362,0.98420,3356.40480,0.98430,3356.65595,0.98440,3356.90710,0.98450,3357.15822,0.98460,3357.40934,0.98470,3357.66044,0.98480,3357.91152,0.98490,3358.16259,0.98500,3358.41364,0.98510,3358.66468,0.98520,3358.91571,0.98530,3359.16672,0.98540,3359.41771,0.98550,3359.66870,0.98560,3359.91966,0.98570,3360.17061,0.98580,3360.42155,0.98590,3360.67247,0.98600,3360.92338,0.98610,3361.17427,0.98620,3361.42515,0.98630,3361.67601,0.98640,3361.92686,0.98650,3362.17769,0.98660,3362.42851,0.98670,3362.67931,0.98680,3362.93010,0.98690,3363.18088,0.98700,3363.43164,0.98710,3363.68238,0.98720,3363.93311,0.98730,3364.18383,0.98740,3364.43453,0.98750,3364.68521,0.98760,3364.93588,0.98770,3365.18654,0.98780,3365.43718,0.98790,3365.68781,0.98800,3365.93842,0.98810,3366.18902,0.98820,3366.43960,0.98830,3366.69017,0.98840,3366.94072,0.98850,3367.19126,0.98860,3367.44179,0.98870,3367.69229,0.98880,3367.94279,0.98890,3368.19327,0.98900,3368.44373,0.98910,3368.69418,0.98920,3368.94462,0.98930,3369.19504,0.98940,3369.44545,0.98950,3369.69584,0.98960,3369.94621,0.98970,3370.19658,0.98980,3370.44692,0.98990,3370.69726,0.99000,3370.94757,0.99010,3371.19788,0.99020,3371.44817,0.99030,3371.69844,0.99040,3371.94870,0.99050,3372.19894,0.99060,3372.44917,0.99070,3372.69939,0.99080,3372.94959,0.99090,3373.19978,0.99100,3373.44995,0.99110,3373.70010,0.99120,3373.95024,0.99130,3374.20037,0.99140,3374.45048,0.99150,3374.70058,0.99160,3374.95066,0.99170,3375.20073,0.99180,3375.45079,0.99190,3375.70082,0.99200,3375.95085,0.99210,3376.20086,0.99220,3376.45085,0.99230,3376.70083,0.99240,3376.95080,0.99250,3377.20075,0.99260,3377.45068,0.99270,3377.70061,0.99280,3377.95051,0.99290,3378.20040,0.99300,3378.45028,0.99310,3378.70015,0.99320,3378.94999,0.99330,3379.19983,0.99340,3379.44965,0.99350,3379.69945,0.99360,3379.94924,0.99370,3380.19901,0.99380,3380.44877,0.99390,3380.69852,0.99400,3380.94825,0.99410,3381.19797,0.99420,3381.44767,0.99430,3381.69735,0.99440,3381.94703,0.99450,3382.19668,0.99460,3382.44633,0.99470,3382.69595,0.99480,3382.94557,0.99490,3383.19517,0.99500,3383.44475,0.99510,3383.69432,0.99520,3383.94388,0.99530,3384.19342,0.99540,3384.44294,0.99550,3384.69245,0.99560,3384.94195,0.99570,3385.19143,0.99580,3385.44090,0.99590,3385.69035,0.99600,3385.93979,0.99610,3386.18921,0.99620,3386.43862,0.99630,3386.68801,0.99640,3386.93739,0.99650,3387.18676,0.99660,3387.43611,0.99670,3387.68544,0.99680,3387.93476,0.99690,3388.18407,0.99700,3388.43336,0.99710,3388.68264,0.99720,3388.93190,0.99730,3389.18115,0.99740,3389.43038,0.99750,3389.67960,0.99760,3389.92880,0.99770,3390.17799,0.99780,3390.42717,0.99790,3390.67633,0.99800,3390.92547,0.99810,3391.17460,0.99820,3391.42372,0.99830,3391.67282,0.99840,3391.92191,0.99850,3392.17098,0.99860,3392.42004,0.99870,3392.66908,0.99880,3392.91811,0.99890,3393.16712,0.99900,3393.41612,0.99910,3393.66511,0.99920,3393.91408,0.99930,3394.16303,0.99940,3394.41197,0.99950,3394.66090,0.99960,3394.90981,0.99970,3395.15871,0.99980,3395.40759,0.99990,3395.65646,1.00000,3395.90531])
    data = data.reshape(-1,2)
    if type == "z":
        y = np.interp(x,data[:,0],data[:,1])*Mpc  # in meter
    elif type == "dis":
        y = np.interp(x,data[:,1],data[:,0])
    return y
    
def bindingenergy(pA,pB,p=10):
    """binding energy calculation between galaxies by A.D.
    defined in the caustic source Libraries/lib-tree/energies_z.c .
    Explained in Ana,2011,MNRAS,412,800 
    ra (radian), dec(radian) , z are needed for pA and pB"""
    
    costheta = np.cos(pA[1])*np.cos(pB[1])*np.cos(pA[0]-pB[0]) + np.sin(pA[1])*np.sin(pB[1])

    rA = pA[3]
    rB = pB[3]
    # criterion of the linking range, 100 Mpc and 10000km/s    
    if rA > 100*Mpc and rB > 100*Mpc and ((rA+rB)*np.sqrt(1-costheta**2)/2. < 10*Mpc) and (np.abs(pA[2]-pB[2])*light_speed < 1E+4) :          
        
        pi = (rA**2 - rB**2)/np.sqrt(rA**2 + rB**2 + 2*rA*rB*costheta) 
        rp = np.sqrt(np.abs((rA**2 + rB**2 - 2*rA*rB*costheta)-pi**2))
        zl = cdistance3((rA+rB)/2/Mpc, type="dis")   
        Rp = rp/(1+zl)
        Pi = pi/(1+zl)*H0*1000/Mpc  # in m/s
        Eb = (-G * p*mgal/(Rp+1E+15) + Pi**2./4.) *p*mgal/1E+22/msun   # unit: of (km/s)^2 10^11 Msol/h
        #~ print(Eb, -G * mtyp**2/Rp/1E+22/msun, Pi**2./4.*mtyp/1E+22/msun)
        ### 1E+15 here is an epsilon to avoid infinity caused by close pairs
    else:
        Eb = 1E+7
    if np.isnan(Eb):
        print(pA,pB,zl,rp,pi)
    return Eb
    
def minbinde(E_set,sig=3):
    """select minimum of binding energy, filter outliers"""
    from scipy.stats import sigmaclip
    E_set = E_set[E_set!=1e+7]    
    E_nset = sigmaclip(E_set,low=sig,high=sig)[0]  # 3-sigma clip
    return [np.min(E_nset),np.max(E_nset)]
    
def genpair(fname,p=10):
    """generate fixed epairs"""
    odata = np.loadtxt(fname,usecols=[0,1,2])  # only adopt the first 3 columns, in case the input data contains more columns
    # transfer degree into radian
    odata[:,:2] = odata[:,:2]*np.pi/180.0   # in radian
    # transfer velocity into redshift 1+z = sqrt((1+v/c)/(1-v/c))
    odata[:,2] = np.sqrt((1+odata[:,2]/light_speed)/(1-odata[:,2]/light_speed))-1
    d_c = cdistance3(odata[:,2])
    
    num_gr = len(odata)
    ydata = np.c_[np.arange(1,num_gr+1),odata,d_c]  
    print(">>>>>> Generating pairwise binding energies...")

    #### output in binary format
    fpair = open(fname+".gpwbe",'wb')
    for i in range(num_gr-1):
        for j in range(i+1,num_gr):
            ##~ print(i,j,)
            E_b = bindingenergy(ydata[i,1:],ydata[j,1:],p=p)
            #~ Eb_set.append(E_b)
            E_bs = struct.pack('f', E_b)
            fpair.write(E_bs)
        ### status bar
        if (i*100.0/num_gr)%10<(100./(num_gr+1)):
            print("{:.0f}%..".format(i*100/num_gr),flush=True,end="")
                
    fpair.close()

    print("Done")

def gentree(fname,method='single',cut=""):
    """generate tree with scipy function"""
    """z: distance, in Mpc"""

    ### for binary output
    fpair = open(fname+'.gpwbe','rb').read()
    print("There are ",int(len(fpair)/4),"pairwise energy values.")
    distArray = struct.unpack(int(len(fpair)/4)*"f", fpair)
    blim = np.min(distArray)
    print("Minimal Binding Energy:",blim)

    Ztree = hac.linkage(distArray-blim, method)  # method in ['single','complete','average'], #'weighted'
    #~ print(Ztree)
    Ztree[:,2]=Ztree[:,2]+blim

    print(">>>>>> Generating tree data (*.gtre) ...")
    ## save dendrogram file gtre
    ## add additional informaiton of nodes, like velocity dispersion etc.
    n = len(Ztree) + 1 # the number of leaves
    vpd = vpdict(fname)
    fv = open(fname+".gtre",'w')
    fv.write("# nodeid lkid rkid bind_energy(ascending) number_of_nodes v Vdis r_avg r_h distance ra dec eta deltav\n")
    sign_skip = 0
    for i in range(n-1):
        set_id = [int(Ztree[i,0]),int(Ztree[i,1])]
        num  = Ztree[i,3]
        
        if sign_skip == 0:  # skip calculation with bEng larger than threshold, accelerate calculations on the root side .
            
            set_v = []
            set_pos = []
            k = i            
                
            while len(set_id) < num:
                for nid in set_id:
                    if nid>=n:
                        set_id.remove(nid)
                        set_id = set_id + [int(Ztree[nid-n,0]),int(Ztree[nid-n,1])]
            
            for nid in set_id:
                #~ print vpd[nid],
                set_v.append(vpd[nid][2])  
                set_pos.append(vpd[nid][:2])
            set_pos = np.array(set_pos)
            set_v = np.array(set_v)
            centroid = harmonic_r(set_pos)
            ## 
            z = np.mean(set_v)/light_speed
            dist = cdistance3([z])[0]/Mpc # in Mpc
            
            vdis = np.std(set_v)
            davg = strucrad(set_pos)*dist
        
        if Ztree[i,2]>1e+6:
            sign_skip = 1 
            
        output = [i+n,Ztree[i,0],Ztree[i,1],Ztree[i,2],num,np.mean(set_v),vdis, davg,
                  centroid[2]*dist,dist,centroid[0]*180/np.pi,centroid[1]*180/np.pi,vdis/davg/np.sqrt(num),(2*vdis/davg/H0)**2/num*6]  # projected radius in Mpc
        fv.write("{:5g} {:5g} {:5g} {:12.2f} {:5g} {:.2f} {:.2f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.2f} {:.2f}\n".format(*output))
        #~ fv.write(" ".join(map(str,output))+"\n")
        ### status bar
        if (i*100./n)%10<(100./(n+1)):
            print(str(int(i*100./n))+"%..",end="",flush=True)
            
    fv.close()
    print("Done")

def eprofile(fname,thr=10):
    """generate and plot binding energy profile according to gtre file"""
    """node ID start from n"""
    from scipy.signal import argrelextrema
    
    linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,3,4])
    length = len(linkage_matrix)+1
    blim = min(linkage_matrix[:,2])
    linkage_matrix[:,2] = (linkage_matrix[:,2]-blim)   # with binding energy
                
    btree = hac.dendrogram(linkage_matrix,get_leaves=True)
    linkage_matrix[:,2] = linkage_matrix[:,2]+blim #   # recover the right binding energy
    treep = [] # binding energy profile of leaves
    treen = [] # binding energy profile of bottom nodes
    treen_set = []
    for i in range(length):
        E_ind = np.where(linkage_matrix[:,:2]==btree["leaves"][i])[0][0]         
        treep.append([i*10+5,linkage_matrix[E_ind,2]])
        #### trace from bottom nodes instead of leaves.
        if E_ind + length not in treen_set: 
            treen.append([i*10+5,linkage_matrix[E_ind,2]])
            treen_set.append(E_ind + length)     #   leaf_id  , the same as in gtre     
    
    tree_prof = np.array(treep)
    treen = np.array(treen)
    treen_set = np.array(treen_set)
    ######### smooth profile with np.convolve
    f_smooth = [0.2]*5 
    #~ f_smooth = [0.25,0.25,0.0,0.25,0.25]
    tree_smooth_x = np.convolve(treen[:,0],f_smooth,"same")
    tree_smooth_y = np.convolve(treen[:,1],f_smooth,"same")
    
    ### save smoothed profile
    np.savetxt(fname+".gepr",np.c_[treen_set, tree_smooth_x, tree_smooth_y], fmt='%5i %5i %.12e')  
        
    min_ind_s = argrelextrema(tree_smooth_y,np.less,order=2)[0]  ## larger order for saddle points
    # plot energy profile
    plt.figure()
    plt.plot(treen[:,0],treen[:,1],'go',ms=2,)
    plt.plot(tree_smooth_x[2:-2],tree_smooth_y[2:-2],'b-')
    #~ energy_c = -4.3/48. * np.sqrt(length)      # critical binding energy
    plt.axhline(y=thr,color="k")
    minf = min_ind_s[tree_smooth_y[min_ind_s] < 0]
    #~ minf = min_ind_s
    plt.plot(treen[minf,0],tree_smooth_y[minf],'ro',ms=4)
    print(">>>>>> Energy profile Analysis:")
    print("saddle points:",len(min_ind_s),",below Zero Binding Energy:",len(minf))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    plt.xlim([0,length*10])
    plt.ylim(minbinde(treen[:,1],sig=5))

    plt.title("binding energy profile")
    #~ plt.show()
    plt.savefig(fname+"_dtreep.pdf")
    plt.close()

def keynodes(fname,outfile,thr=50,yulim=1500,debug=0,memLim=9,trim="dEta",vdisLim=200):
    """analyze the tree with the blooming tree approach """
    """color candidates, save to *.glis"""
    """plot dendrogram according to gtre file, with velocity dispersion as y-axis"""
    """BUG: for the nodes with the same velocity dispersion, the position of key notes 
    may drift horizontally. """
    
    linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,6,4])
    length = len(linkage_matrix)+1  # number of particles
    ind_sort = linkage_matrix[:,2].argsort().argsort() # the index of the sorted list for the original list
    
    linkage_matrix[linkage_matrix[:,2].argsort()]    
        
    print(">>>>>> Tree Analysis:")
    min_ind,node_cl = minode(fname,thr=thr,memLim=memLim,vdisLim=vdisLim)   # get nodes of substructures, threshold 1
    print("There are {0} structures found with the threshold {1} in the tree.".format(len(node_cl),thr))
        
    ########## new membership ################
    ndata = np.c_[np.loadtxt(fname,usecols=range(3)),np.zeros([length,1])]  # add new column of membership
            
    color_set = ["0.7"]*(2*length-1)
    i=0
    for i in range(len(node_cl)):
        [nodes,leaves] = find_children(linkage_matrix,node_cl[i]+length)
        ndata[np.array(leaves),3] = i+1  # start from 1
        for lid in nodes:
            color_set[lid]=color_cl[i%18]  # node id start from 0, for basic structures   

    np.savetxt(outfile+".glis",ndata,fmt="%12.6f %12.6f %8i %4i")
    if len(node_cl) >0 :
        fsum = open(outfile+".gsum",'w')
        fsum.write("#Cl_ID, sub_ID, num, RA, DEC, z, vdis, knode \n")
        nsub = set(ndata[:,-1])
        for k in nsub:
            if k !=0:
                mem = ndata[ndata[:,-1]==k]
                out = [0,k,len(mem),np.median(mem[:,0]),np.median(mem[:,1]),np.median(mem[:,2])/light_speed,np.round(np.std(mem[:,2])),node_cl[int(k-1)]+length]
                fsum.write("{:.0f} {:.0f} {:.0f} {:.6f} {:.6f} {:.4f} {:.0f} {:.0f}\n".format(*out))
        fsum.close()    
    
    ####################################################################
    ### Just for debug. Use pltree function for regular outputs 
    ####################################################################
    if (debug == 1) or (debug=="T"):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.title("dendrogram")
        plt.xlabel("Leaves")
        plt.ylabel("$\sigma_v$ (km/s)")
        mpl.rcParams['lines.linewidth'] = 0.2
        btree = hac.dendrogram(linkage_matrix,link_color_func=lambda x: color_set[x],no_labels=True)

        # coordinates of nodes
        coord = np.c_[np.array(btree['icoord'])[:,1:3],np.array(btree['dcoord'])[:,1]]
        coord = coord[np.argsort(coord[:,2])]   # sort the tree distance
        bx = (coord[:,0]+coord[:,1])/2.0
        by = coord[:,2]

        #### add ID on the x axis
        mdata= np.loadtxt(fname+".gegpr")  # tree profile
        for k in range(len(min_ind)):
            plt.plot(mdata[min_ind[k],1],1,'^')
            plt.annotate("%3i" % mdata[min_ind[k],0], (mdata[min_ind[k],1], 10*(k%2)), xytext=(0, -5),
                textcoords='offset points', va='bottom', ha='center',size=2)
        
        
        if len(node_cl)>0:
            plt.plot(bx[ind_sort[np.array(node_cl)]],by[ind_sort[np.array(node_cl)]],'s')  # the maker of clusters

        for j in range(len(node_cl)):
            plt.text(bx[ind_sort[np.array(node_cl[j])]]-30,by[ind_sort[np.array(node_cl[j])]]+30, j+1,fontsize=10)

        if yulim!="":
            plt.ylim([0,yulim])        
        else:
            ply.ylim([0,1500])
        
        ax.axes.get_xaxis().set_visible(False)

        #~ plt.show()
        plt.savefig(fname+"_"+trim+"_thr"+str(thr)+"_M"+str(memLim)+'.pdf')
        plt.close(fig)

def keynodes0(fname,outfile,thr=10,yulim=1500,debug=0,memLim=9,trim="vdis"):
    """analyze the tree with binding energy profile"""
    """color candidates, save to *.glis"""
    """plot dendrogram according to gtre file, with velocity dispersion as y-axis"""
    """BUG: for the nodes with the same velocity dispersion, the position of key notes 
    may drift horizontally. """
    
    linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,6,4])
    length = len(linkage_matrix)+1  # number of particles
    ind_sort = linkage_matrix[:,2].argsort().argsort() # the index of the sorted list for the original list
    
    linkage_matrix[linkage_matrix[:,2].argsort()]
    
    if trim == "vdis":
        indx = 6        
    elif trim == "bEne":
        indx = 3
    elif trim == "node":
        indx = 0
    elif trim == "eta":
        indx = -2
    else:
        indx = 6
    print(">>>>>> Tree Analysis:")
    n_node =  minode3(fname,thr=thr,memLim=memLim,indx=indx)

    if len(n_node)>0:
        node_cl = np.array(n_node)[:,0]   # get nodes of substructures, threshold 1
        print("There are {0} structures found with the threshold {2}={1} (memLim > {3})in the tree.".format(len(node_cl),thr,trim,memLim))
    else:
        node_cl = []
        print("There are {0} structures found with the threshold {2}={1} (memLim > {3})in the tree.".format(len(node_cl),thr,trim,memLim))

    ########## new membership ################
    ndata = np.c_[np.loadtxt(fname,usecols=range(3)),np.zeros([length,1])]  # add new column of membership
            
    i=0
    for i in range(len(node_cl)):
        [nodes,leaves] = find_children(linkage_matrix,node_cl[i])
        # print node_cl[i]+length,nodes
        ndata[np.array(leaves),-1] = i+1  # start from 1

    np.savetxt(outfile+".glis",ndata,fmt="%18.6f"*3 + "%4i", header="#")

    fsum = open(outfile+".gsum",'w')
    fsum.write("#Cl_ID, sub_ID, num, RA, DEC, z, vdis, knode \n")
    nsub = set(ndata[:,-1])
    for k in nsub:
        if k !=0:
            mem = ndata[ndata[:,-1]==k]
            out = [0,k,len(mem),np.median(mem[:,0]),np.median(mem[:,1]),np.median(mem[:,2])/light_speed,np.round(np.std(mem[:,2])),node_cl[int(k-1)]]
            fsum.write("{:.0f} {:.0f} {:.0f} {:.6f} {:.6f} {:.4f} {:.0f} {:.0f}\n".format(*out))
    fsum.close()    

def keynodes3(fname,outfile,thr,memLim=9):
    """analyze the tree with sigma plateau threshold"""
    """save to *.slist"""
    
    linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,3,4])
    length = len(linkage_matrix)+1  # number of particles

    print(">>>>>> Tree Analysis:")
    
    ncols = 3 
    ########## new membership ################
    ndata = np.c_[np.loadtxt(fname,usecols=range(ncols)),np.zeros([length,1]),np.zeros([length,1])]  # add new column of membership
        
    ### cluster member assignment
    [nodes,leaves] = find_children(linkage_matrix,thr[2]+length)
    print("cluster:",len(leaves))
    print("v_disp thr:",thr[6])
    print("E thr:",thr[4])
    ndata[np.array(leaves),-2] = 1  # start from 1
    
    ### sub member assignment  
    node_cl = [] 
    knode = minode3(fname,thr=thr[5],indx=3,memLim=memLim)

    for node in knode:
        if node[0] in nodes:    ## check node in cluster or not 
            node_cl.append(node[0])
            
    print("sub:")
    print("v_disp thr:",thr[7])
    print("E thr:",thr[5])
    for i in range(len(node_cl)):
        [nodes,leaves] = find_children(linkage_matrix,node_cl[i])
        print(i+1,len(leaves))
        ndata[np.array(leaves),-1] = i+1  # start from 1
        
    np.savetxt(fname+"_sigpl.glis",ndata,fmt="%18.6f "*ncols + "%4i "*2, header="#")

    #############################
    fsum = open(fname+"_sigpl.gsum",'w')
    fsum.write("#Cl_ID, sub_ID, num, RA, DEC, z, vdis, knode \n")
    for i in range(2):
        nsub = set(ndata[:,i+ncols])
        #~ print(i,nsub)
        for k in nsub:
            if k !=0:
                mem = ndata[ndata[:,i+ncols]==k]
                out = [i,k,len(mem),np.median(mem[:,0]),np.median(mem[:,1]),np.median(mem[:,2])/light_speed,np.round(np.std(mem[:,2])),node_cl[int(k-1)]]
                fsum.write("{:.0f} {:.0f} {:.0f} {:.6f} {:.6f} {:.4f} {:.0f} {:.0f}\n".format(*out))
    fsum.close()    
        
def profvdis(fname,knode,thr=0):
    """plot velocity dispersion profile to check sigma plateau"""
    import scipy
    import matplotlib.gridspec as gridspec
    from scipy.interpolate import interp1d
    
    data = np.loadtxt(fname+".gtre")
    n_leaf = len(data)+1
    node = knode
    
    data[np.isnan(data)] = 1e+4  #### fix nan values 
    
    E_set = []  #  binding energy
    v_set = []  # velocity dispersion
    eta_set = []
    id_set = [] # set of sequence, begin from 0
    ######## search nodes above given node
    while (node < 2*n_leaf-2): # and (data[ind,6] < 4000):
        #~ print(k,np.where(data[:,1:3]==k))
        ind = np.where(data[:,1:3]==node)[0][0]
        id_set.append(ind)
        E_set.append(data[ind,3]) 
        v_set.append(data[ind,6])
        eta_set.append(data[ind,6]/data[ind,7]/np.sqrt(data[ind,4]))
        node = ind + n_leaf
        
    if len(id_set) > 0:
        id_set = np.array(id_set)
        E_set = np.array(E_set)
        v_set = np.array(v_set)
        eta_set = np.array(eta_set)
        tailcut = np.argwhere(v_set>1000)
        if len(tailcut)>0:
            tailcut = tailcut[0][0]
        else:
            tailcut = -1
        
        nbin  = 100 

        plt.figure()
        gs = gridspec.GridSpec(1, 4)
        gs.update(wspace=0, hspace=0)
        ax1 = plt.subplot(gs[0:3])
        ax1.plot(id_set[:tailcut],v_set[:tailcut],'-o',ms=2,alpha=0.6)            ## id distribution 
        ax1.invert_xaxis()
        ax2 = plt.subplot(gs[3],sharey=ax1)   

        ### resample the id range to make an equal distribution for histogram 

        size_sam = 10000
        idrang = np.linspace(np.min(id_set[:tailcut]),np.max(id_set[:tailcut]),size_sam)
        fmb = interp1d(id_set,v_set, kind='linear')
        num,bin,patches = ax2.hist(fmb(idrang),bins= nbin, orientation='horizontal') 
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax1.set_xlabel("node ID")
        ax2.set_xlabel("node number")
        ax1.set_ylabel("$\sigma_v$ (km/s)")        
        
        #### calculate mixture gaussian components,  GMM 
        from sklearn import mixture
        
        sample = fmb(idrang).reshape(-1,1) # v_set[:-tailcut].reshape(-1,1)
        print(np.min(sample),np.max(sample))
        bic0 = 1E+10
        for i in range(1,10):
            clf = mixture.GaussianMixture(n_components=i, covariance_type='diag').fit(sample)
            aic = clf.aic(sample)
            bic = clf.bic(sample)
            ###~ print(i,bic,aic)
            ## why negative ??
            if bic < bic0:
                bic0 = bic 
                clf0 = clf
                i0 = i
        m = clf0.means_
        w = clf0.weights_
        c = np.sqrt(clf0.covariances_)
        m = m[w>0.01]
        c = c[w>0.01]
        w = w[w>0.01]
        indx = np.argsort(w)   #  large weight with a relative small deviation
        
        print(i0, m.T, c.T, w, indx, sep="\n")
        ############# get keyid  ###########
        
        ind_pl = -1 # index of plateau
        ind_cl_set = np.where(v_set < (m[indx[ind_pl]] + c[indx[ind_pl]]) )[0]  ### trick  ????
        ind_sub_set = np.where(v_set > (m[indx[ind_pl]] - c[indx[ind_pl]]) )[0]
        
        ind_end = ind_cl_set[-1]   ### Note: to supress occasional fluctuation at two ends.
        ind_start = ind_sub_set[1] ###
        print("Main branch key posi:",ind_start,ind_end)
        ind_sub = np.argmin(v_set[ind_start:ind_end]) + ind_start
        ind_cl = ind_end   #np.argmax(v_set[ind_sub:ind_end]) + ind_sub
        
        ax1.plot(id_set[ind_cl],v_set[ind_cl],'sr',alpha=0.6)
        ax1.plot(id_set[ind_sub],v_set[ind_sub],'^r',alpha=0.6)

        mbin = (bin[1:] + bin[:-1])/2
        
        ax1.axhline(y=m[indx[-1]],color="k",alpha=0.6,zorder=0)
        ax1.axhline(y=m[indx[-1]]+c[indx[-1]],ls="--",color="k",lw=1,alpha=0.4)
        ax1.axhline(y=m[indx[-1]]-c[indx[-1]],ls="--",color="k",lw=1,alpha=0.4)
        ax1.axvline(x=id_set[ind_end],ls="--",color="k",lw=1,alpha=0.4)
        ax1.axvline(x=id_set[ind_start],ls="--",color="k",lw=1,alpha=0.4)
        bink = np.linspace(bin[0],bin[-1],nbin)
        for j in range(len(m)):        
            if w[j]>0.1:
                model = scipy.stats.norm(m[j],c[j][0]).pdf(bink)*w[j]*(bin[1]-bin[0])*size_sam 
                ax2.plot(model,bink, '-',lw=2, alpha=0.6)   # reverse x,y because of horizontal orientation
   
        plt.savefig(fname+"_sigpl_k"+str(knode)+"_vhst.png",dpi=300)
        plt.close()
                        
        ######## check other profile 
        plt.figure()
        plt.plot(id_set[ind_cl],E_set[ind_cl],'sr')
        plt.plot(id_set[ind_sub],E_set[ind_sub],'^r')
        plt.plot(id_set[:-int(len(id_set)/2)],E_set[:-int(len(id_set)/2)],'-o',ms=2,alpha=0.6)
        plt.gca().invert_xaxis()
        plt.xlabel("node ID")
        plt.ylabel("binding energy")
        plt.savefig(fname+"_k"+str(knode)+"_E.png",dpi=150)

        plt.figure()
        plt.plot(id_set[:-int(len(id_set)/4)],eta_set[:-int(len(id_set)/4)],'-o',ms=2,alpha=0.6)
        plt.plot(id_set[ind_cl],eta_set[ind_cl],'sr')
        plt.plot(id_set[ind_sub],eta_set[ind_sub],'^r')
        plt.gca().invert_xaxis()
        plt.xlabel("node ID")
        plt.ylabel("$\eta$")
        plt.savefig(fname+"_k"+str(knode)+"_eta.png",dpi=150)
        
        print("Main branch key ID: ",id_set[ind_cl]+n_leaf,id_set[ind_sub]+n_leaf)
        return  [m[indx[-1]][0],c[indx[-1]][0],id_set[ind_cl],id_set[ind_sub],E_set[ind_cl],E_set[ind_sub],v_set[ind_cl],v_set[ind_sub],w[indx[-1]],w[indx[-1]]/c[indx[-1]]]

def mainbranch(fname,mtyp):
    """find the key node of main branch and plot the profile"""
    """from top to the leaf"""
        
    data = np.loadtxt(fname+".gtre",usecols=[0,1,2,4,6])
    length = len(data)+1
    
    id_set = []
    k = length * 2 - 2
    while np.max(data[k-length,1:3]) > length - 1 :
        mem = 1
        
        for i in [1,2]:
            ind = np.where(data[:,0] == data[k-length,i])
            if len(ind) > 0:
                if data[ind[0],3] > mem:
                    mem = data[ind[0],3] 
                    indk = ind[0] + length

        id_set.append(int(data[indk-length,0][0]))
        k = indk[0]
        
    print("Main branch node:",k)
    thr = profvdis(fname,k)
    np.savetxt(fname+"_sigpl.gthr",thr+[mtyp],fmt="%14.6f",header="#v_disp,  v_d_e,  ID_cl, ID_sub, E_cl,  E_sub, v_cl, v_sub, w, w/c, mtyp")
       
    return [thr,id_set]
    
def plot1d2(fname,zrange="",memLim=9,yulim="",vdisPlot=200):
    """plot 1D velocity histogram of the tree analysis result"""
    import itertools
    pdata = np.loadtxt(fname+'.glis')  
    nbin = 50  
    pdata[:,2] = pdata[:,2]/light_speed
    
    if zrange == "":
        uplim = np.percentile(pdata[:,2],99)
        dnlim = np.percentile(pdata[:,2],1)
    else:
        uplim = zrange[1]
        dnlim = zrange[0]
        
    data_sub = np.loadtxt(fname+".gsum",usecols=[1,2,5,6],ndmin=2)
    data_subf = data_sub[(data_sub[:,1]>memLim)*(data_sub[:,3]>vdisPlot)*(data_sub[:,2]>dnlim)*(data_sub[:,2]<uplim)]   #  filter subs
    print(len(data_subf))
    
    set_grps=[]
    if len(data_subf) > 0 :
        
        for i in range(len(data_subf)):
            tmp_set = pdata[pdata[:,-1]==data_subf[i,0]][:,2]
            set_grps.append(tmp_set)  
        print("The 1D2 plot contains {} strs in the range [{:.4f},{:.4f}]".format(len(set_grps),dnlim,uplim))
        
        v_tot = list(itertools.chain(*set_grps))
        if zrange == "":
            uplim = max(v_tot)
            dnlim = min(v_tot) 
        
    bins = np.arange(nbin)*(uplim-dnlim)*1.0/(nbin-1) + dnlim
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)  
    fig.subplots_adjust(hspace=0)
    # Turn off axis lines and ticks of the big subplot)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.tick_params(labelcolor='w', which='both',top=False, bottom=False, left=False, right=False)
    
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharey=ax1)  
    
    if len(set_grps) >0 : 
        for i in range(len(set_grps)):
            set_subs=set_grps[i]  # just the velocity
            ax1.hist(set_subs,bins,color=color_cl[i%18],alpha=0.6,edgecolor="k",label=int(data_subf[i,0]))
            ax2.plot(bins,norm.pdf(bins,np.mean(set_subs),np.std(set_subs))*(uplim-dnlim)/(nbin-1)*len(set_subs),'-',color=color_cl[i%18],lw=2)  ## equal area normal distribution

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.legend(fontsize='6',bbox_to_anchor=(1,1))
    ax1.set_xlim([dnlim*0.95,uplim*1.05])
    ax2.set_xlim([dnlim*0.95,uplim*1.05])
    if yulim != "":
        ax1.set_ylim([0,yulim])
        ax2.set_ylim([0,yulim])
    ax2.set_xlabel("z")
    plt.savefig(fname+'_1D2_m'+str(memLim)+'.pdf')
    plt.close()
    
def pltree(fname,outfile,thr="",method="single",yaxis="node",rmem="F",memLim=9,knode=""):
    """plot dendrogram"""
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title("Dendrogram Tree ("+method+")")
    plt.xlabel("Leaves")
    mpl.rcParams['lines.linewidth'] = 0.4
    
    if yaxis=="node":
        linkage_matrix = np.loadtxt(fname+'.gtre',usecols=[1,2,3,4])
        length = len(linkage_matrix)+1
        linkage_matrix[:,2] = np.arange(length-1) + length-1 #  discard leaves, start with nodes ID 
        plt.ylabel("Node sequence")
        
    elif yaxis == "vdis" or yaxis == "dEta" :        
        linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,6,4])
        length = len(linkage_matrix)+1    
        plt.ylabel("$\sigma_v$ (km/s)")
        
    elif yaxis == "sigpl" :        
        linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,6,4])
        length = len(linkage_matrix)+1
        thr = np.loadtxt(fname+"_sigpl.gthr")[0]
        plt.ylabel("$\sigma_v$ (km/s)")            
            
    elif yaxis == "eta" :        
        linkage_matrix = np.loadtxt(fname+".gtre",usecols=[1,2,-2,4])
        linkage_matrix[:,2][linkage_matrix[:,2]>1000]=1000
        linkage_matrix[:,2] = 1001-linkage_matrix[:,2]
        linkage_matrix = np.nan_to_num(linkage_matrix, nan=0.0, posinf=1e+8)  # replace nan values
        length = len(linkage_matrix)+1
        plt.ylabel("$\eta$ (km/s/Mpc)")            
        
    elif yaxis == "bEne" :
        linkage_matrix = np.loadtxt(fname+'.gtre',usecols=[1,2,3,4])
        length = len(linkage_matrix)+1
        E_range = minbinde(linkage_matrix[:,2])            
        linkage_matrix[:,2] = linkage_matrix[:,2]-E_range[0]+1e-6  # should be positive
        linkage_matrix[:,2][linkage_matrix[:,2]<0]=1e-6
        plt.ylabel("Binding Energy (10$^{11}$ km$^2$ s$^{-2}$ M$_\odot$ h$^{-1}$)")

    
    if os.path.exists(fname+".pid") and rmem=="T":
        """if *.pid file exists, color tree with real membership"""
        
        cID = fname.split('.')[1]
        pid = np.loadtxt(fname +".pid",dtype=np.string_, usecols=[1], converters = {1: lambda s: s.split('-')})

        # check filtered particle is member or not
        clset = dict()
        for i in range(len(pid)):
            # for target cluster
            if pid[i][0] == str(int(cID)):
                if pid[i][1] in clset:
                    clset[pid[i][1]].append(i)
                elif pid[i][1] != "N":
                    clset[pid[i][1]]=[i]
                    
            elif (pid[i][1] != "N") and (pid[i][0] != str(int(cID))):
                
                if pid[i][0] in clset:
                    clset[pid[i][0]].append(i)
                else:
                    clset[pid[i][0]]=[i]
    
        ### show membership of core and subs ###
        ### dendrogram plot, only node color is adjustable,[1
        color_set = ["0.7"]*(2*length-1)
        
        ind_c = 0 # color index
        for key in sorted(map(int,clset.keys())):    # color leaves with structure id 
            key = str(key)
            if len(clset[key]) >= memLim:
                print(key,color_cl[ind_c%18],len(clset[key]))
                for id in clset[key]:
                    lid = np.where(linkage_matrix[:,:2]==id)[0] # line sequence
                    if len(lid) > 0:
                        color_set[lid[0]+length]= color_cl[ind_c%18]            # node id
                ind_c = ind_c + 1                 
    elif rmem=="F" or rmem==0:
        """when real observation, color tree with found substructure and group"""
        data = np.loadtxt(outfile+".glis")
        color_set = ["0.7"]*(2*length-1)
        
        for i in range(len(data)):
            if data[i,-1]!=0 and np.sum(data[:,-1]==data[i,-1])>memLim:
                ind_str = np.where(linkage_matrix[:,:2]==i)[0][0]
                
                color_set[ind_str+length] = color_cl[(int(data[i,-1])-1)%18]  # node id start from 0, for basic structures

    elif rmem==-1:
        color_set = ["0.3"]*(2*length-1)
    
    btree = hac.dendrogram(linkage_matrix,link_color_func=lambda x: color_set[x])
    
    # coordinates of nodes
    coord = np.c_[np.array(btree['icoord'])[:,1:3],np.array(btree['dcoord'])[:,1]]    
    coord = coord[np.argsort(coord[:,2])]   # sort the tree distance 
    bx = (coord[:,0]+coord[:,1])/2.0
    by = coord[:,2]
    
    ###### id of structures
    ind_sort = linkage_matrix[:,2].argsort().argsort()
    if os.path.exists(fname+".gsum"):
        node_str = np.loadtxt(fname+".gsum",usecols=[-1,2,1],dtype="int32",ndmin=2)
        node_str[:,0] = node_str[:,0] - length
        for j in range(len(node_str)):
            if node_str[j,1]>memLim:
                plt.text(bx[ind_sort[node_str[j,0]]],by[ind_sort[node_str[j,0]]], node_str[j,2],fontsize=8)
    
    ax.axes.get_xaxis().set_visible(False)
    
    if thr !="" :
        zmscale = 3 #  zoom scale
        if yaxis=="node" :
            plt.ylim([0,zmscale*length])   
        elif yaxis == "vdis" or yaxis == "sigpl": 
            plt.axhline(y=thr,color="k",lw=1)      
            plt.ylim([0,zmscale*thr])   
        elif yaxis == "dEta":
            vdism = 500
            plt.ylim([0,zmscale*vdism ])   
            
        elif yaxis == "bEne":
            step = -E_range[0]+thr
            plt.axhline(y=step,color="k",lw=1)
            
            plt.ylim([0,zmscale*(step)])
            
            ylabel = np.ones(zmscale)*E_range[0] + np.arange(zmscale)*step
            plt.yticks(np.arange(zmscale)*step, np.round(ylabel,2) )        
            
        yaxis = yaxis+"_m"+str(memLim)+"_zoom"
    else:
        if yaxis == "vdis" or yaxis == "sigpl": 
            plt.ylim([0,3000])   
        elif yaxis == "bEne":
            zmscale = 5
            step = 100/zmscale
            plt.ylim([0,100])
            ylabel = np.ones(zmscale)*E_range[0] + np.arange(zmscale)*step
            plt.yticks(np.arange(zmscale)*step, np.round(ylabel,2) )           
            
    if knode !="":
        
        node_set,leaf_set = find_children(linkage_matrix,knode)
        node_seq = list(map(int,btree['ivl']))
        xset = []
        for id in leaf_set:
            xset.append(node_seq.index(id))
        
        xrange = [np.min(xset)-1,np.max(xset)+1]
    
        print("Tree is zoomed to",xrange)
        plt.xlim([xrange[0]*10,xrange[1]*10])
        yaxis = yaxis+"_k"+str(knode)
      
    hac.set_link_color_palette(None)
    plt.savefig(outfile+'_dtree_y-'+yaxis+'.pdf')
    plt.close(fig)
    
def plot2d(fname, zrange="",memLim=9,limi="",sign="sub"):
    """plot 2D projection of the tree analysis result"""
    
    pdata = np.loadtxt(fname+'.glis')
    
    fig = plt.figure()
    plt.plot(pdata[:,0], pdata[:,1],'k,',alpha =0.6, zorder=1)  
         
    if np.std(pdata[:,2]) > 10:
        pdata[:,2] = pdata[:,2]/light_speed
    if zrange!="":
        pdata = pdata[(pdata[:,2]>zrange[0])*(pdata[:,2]<zrange[1])]
        if len(pdata)==0:
            sys.exit("There are 0 records in the zrange")      
    
    if sign=="sub":
        indx = -1
    elif sign=="cl":
        indx = -2
        
    topnode_set = set(pdata[:,indx])
    n_grps = len(topnode_set)
    set_grps=[]
    if n_grps > 0 :
        
        for i in topnode_set:
            if np.sum(pdata[:,indx]==i) > memLim :
                set_grps.append(pdata[pdata[:,indx]==i][:,:3])            
        print("The procedure finds {} {}".format(len(set_grps)-1,sign))
    
    n_grps = len(set_grps)
    
    for j in range(1,n_grps):
        c = color_cl[(j-1)%18]
        plt.plot(set_grps[j][:,0],set_grps[j][:,1],'o',ms=4,alpha=0.6,color=c,label=str(j))
        plt.text(np.mean(set_grps[j][:,0]),np.mean(set_grps[j][:,1]),j,fontsize=8,color="k",alpha=0.8)

    plt.xlabel("RA")
    plt.ylabel("DEC")
    
    pos_c = [(min(pdata[:,0])+max(pdata[:,0]))/2,(min(pdata[:,1])+max(pdata[:,1]))/2]
 
    suffix = ""
    if limi=="":
        limi = max([max(pdata[:,0])-min(pdata[:,0]),max(pdata[:,1])-min(pdata[:,1])])/2
    else:        
        suffix = "_zoom"
    plt.xlim([-1*limi+pos_c[0],limi+pos_c[0]])
    plt.ylim([-1*limi+pos_c[1],limi+pos_c[1]])

    if not os.path.exists(fname[:-4] +".pid"):
        plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.savefig(fname+'_2D_m'+str(memLim)+"_"+sign+suffix+'.pdf')
    plt.close()

    
def plotzdt(fname,zrange="",memLim=9,vdisLim=200):
    """plot redshift diagram of the tree analysis result
    in Cartesian coordinate system"""

    pdata = np.loadtxt(fname+'.glis')       
    pdata[:,2] = pdata[:,2]/light_speed
    ddata = np.loadtxt(fname+".gsum",usecols=[1,2,5,6],ndmin=2)
        
    if zrange!="":
        z_down = min(map(float,zrange))
        z_up = max(map(float,zrange))
        suffix = "_zcut"
    else:        
        ddata = ddata[(ddata[:,1]>memLim)*(ddata[:,3]>vdisLim)]
        z_up = np.max(ddata[:,2])*1.1
        z_down = np.min(ddata[:,2])*0.75
        suffix = ""
    print("The redshift range of zDt plot is {:.4f} ~ {:.4f}".format(z_down,z_up))
    subsam = pdata[(pdata[:,2]>z_down)*(pdata[:,2]<z_up)]

    dc = cdistance2(subsam[:,2])/Mpc # in Mpc
    subsam[:,0]=(subsam[:,0]-np.mean(subsam[:,0]))*np.pi/180*dc
    subsam[:,1]=(subsam[:,1]-np.mean(subsam[:,1]))*np.pi/180*dc

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)  
    fig.subplots_adjust(hspace=0)
    # Turn off axissector lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharex=ax1) 
    ax_set = [ax1,ax2,'RA(Mpc)','DEC(Mpc)']
    
    # build group sets
    topnode_set = set(subsam[:,3])
    n_grps = len(topnode_set)
    if n_grps > 0 :
        set_grps=[]
        
        for i in topnode_set:
            if np.sum(subsam[:,3]==i) > memLim:
                set_grps.append(subsam[subsam[:,3]==i][:,:3])  
    
    n_grps = len(set_grps)
    
    for i in [0,1]:
        ax_set[i].plot(subsam[:,2],subsam[:,i],'k.',markersize=0.5)
        ax_set[i].set_ylabel(ax_set[i+2])
        
        for j in range(1,n_grps):
            c = color_cl[(j-1)%18]
            ax_set[i].plot(set_grps[j][:,2],set_grps[j][:,i],'o',ms=2,alpha=1,color=c,label="str "+str(j))
            
        ax_set[i].set_xlim([z_down,z_up])
         
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax.set_xlabel('z')
    plt.savefig(fname+'_zDt_m'+str(memLim)+suffix+'.pdf')
    plt.close()

    
def genreg(fname,zrange="",memLim=9,vdisPlot=300):
    """generate DS9 region file"""
    
    pdata = np.loadtxt(fname+'.glis')
        
    data_sub = np.loadtxt(fname+".gsum",usecols=[1,2,5,6],ndmin=2)
    if zrange =="":
        topnode_set = set(data_sub[(data_sub[:,1]>memLim)*(data_sub[:,3]>vdisPlot)][:,0])
        suffix = ""
    else:
        topnode_set = set(data_sub[(data_sub[:,1]>memLim)*(data_sub[:,3]>vdisPlot)*(data_sub[:,2]>zrange[0])*(data_sub[:,2]<zrange[1])][:,0])
        print(topnode_set)
        suffix = "_zcut"
        
    f=open(fname+'_m'+str(memLim)+suffix+'.reg','w')
    f.write("""
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    """)
    for k in range(len(set(pdata[:,-1]))):
        if k not in topnode_set:
            #~ print(k)
            pdata[:,-1] = pdata[:,-1]-k*(pdata[:,-1]==k)
        #~ else:
            #~ print(int(k),np.sum(pdata[:,-1]==k))
        
    for i in range(0,len(pdata[:,0])):
        line = ""
        if pdata[i,-1]!=0:
            color = color_cl[(int(pdata[i,-1])-1)%9]
            line = line+'box('+str(pdata[i,0])+','+str(pdata[i,1])+',0.03",0.03") # color='+color+' width=1\n'
        f.write(line)
    f.close()   
    print("Ds9 regions of {0} structures are generated.".format(len(set(pdata[:,-1]))-1))
    
def find_parents(data,node,depth=1e+5):
    """return parent nodes id (begin from 0) above the given Node"""
    """work with *.gtre data"""
    
    node_set = []  # nodes
    n_leaf = len(data)+1
    k = int(node)
    if k <= 2*n_leaf-1:
        node_set.append(k)
        ind = (np.where(data[:,1:3]==k))[0] 
        #~ ##print(k,ind)
        if ind.size > 0:
            if ind < (n_leaf-1) and ind > 0:
                nset = find_parents(data,data[ind,0])
                node_set = node_set + nset[0]

    return [node_set]
    
def subtree_vdis(nfile,outfile,memLim=9,vLim=5000,zrange="",vdisLim=0,cLim=1000):
    """plot simplified tree with substructures. Y-axis is the velocity dispersion."""
    data = np.loadtxt(outfile+".gsum",usecols=[1,-1,2,5,6],ndmin=2)
    data = data[data[:,2]>memLim]
    if len(data) > 2: 
            
        if zrange!="":
            data = data[(data[:,3]>zrange[0])*(data[:,3]<zrange[1])]
            suffix = "_zcut"
        else:
            suffix = ""
        
        data = data[data[:,-1]>vdisLim]
        dtree = np.loadtxt(nfile+".gtre")
        length = len(data)

        #### calculate distance between nodes
        idict = {} 
        parent_set = []
        for i in range(length):
            idict[i] = int(data[i,0])
            parent = find_parents(dtree,data[i,1])
            parent_set.append(parent[0])
        
        ytdist = []
        for i in range(length):
            for j in range(i+1,length):
                dis_set = set(parent_set[i])&set(parent_set[j])  # the number of same parent id
                disij = -len(dis_set)                            # the distance to the root 
                ytdist = ytdist + [disij]

        ndis = np.array(ytdist) - min(ytdist)+1
        linkage_matrix = hac.linkage(ndis, 'single')
        
        #####  replace y axis with velocity dispersion 
        vdis = []
        addi = []
        for i in range(len(linkage_matrix)):
            a = linkage_matrix[i,0]
            b = linkage_matrix[i,1]
            ###  find leaf
            while a > length-1:
                a = linkage_matrix[int(a-length),0]
            while b > length-1:
                b = linkage_matrix[int(b-length),0]   
                
            k_ind = min(set(parent_set[int(a)])&set(parent_set[int(b)]))
            vdis.append(dtree[k_ind-len(dtree)-1,6])  # Vdis
            addi.append(dtree[k_ind-len(dtree)-1,np.r_[0, 4:6,7:9]])
            
        vdis = np.array(vdis) 
        addi = np.array(addi) 
        vdis = vdis*(vdis<vLim)+(vdis>vLim)*vLim   # remove value above given vlimit
        linkage_matrix[:,2] = vdis
        fig, ax = plt.subplots()
        dn = hac.dendrogram(linkage_matrix,distance_sort=True,color_threshold=cLim)#,orientation='right')
        
        #####  save tree nodes and properties, id start from 1, consistent with plot
        id_up = linkage_matrix[:,:2]
        id_up[id_up<length] = id_up[id_up<length]+1
        np.savetxt(outfile+"_stree_vdis"+suffix+'_m'+str(memLim)+".gtre",np.c_[np.arange(length-1)+length,id_up,linkage_matrix[:,2:],addi],fmt="%7i %7i %7i %7.1f %7i %7i %7i %10.2f %7.3f %7.3f",header="node Lnode  Rnode  vdis  num  knode  N  cz  r_avg  r_h")

        ### adjust subid, start from 1 
        xsub = [idict[int(t.get_text())]  for t in ax.get_xticklabels()]
        ax.set_xticklabels(xsub)
        plt.xlabel("group ID",fontsize=12)
        plt.ylabel("$\sigma_v$ (km/s)",fontsize=12)
        plt.savefig(outfile+"_stree_vdis"+suffix+'_m'+str(memLim)+".pdf")
    
def main(fname,trim="dEta",thres=50, p=10, memLim=10,memPlot=10, zr="",vdisLim=100,vdisPlot=200):
    """The input data file should start with a summary line, which include: 
    the number of galaxies, the center of RA, DEC, redshift"""
            
    from timeit import default_timer as timer
    start = timer()
    
    if fname[-4:-2]==".g": 
        nfile = fname
        i = int(fname[-2:])
        fname = fname[:-4]
    else:
        i = 0
        while os.path.exists(fname+".g"+str(int(i)).zfill(2)):
            i = i+1
        nfile = fname+".g"+str(int(i)).zfill(2)
                
    ###### deal with the parameter file for the pairwise energy 
    logfile = nfile+".gpar"
    nh,nmag,delim=precheck(fname)
    
    if not os.path.exists(logfile):
        print(logfile,"does not exist. Default values are adopted.")
        sig_rad,sig_v = filterdata2(fname,zrange=zr,nfile=nfile,nhead=nh,sep=delim)        
        logf = open(logfile,'w')
        logf.write("# Input Parameters: \n")
        logf.write("position unit: "+sig_rad+"\n")
        logf.write("l.o.s unit: "+sig_v+"\n")
        logf.write("Typical Mass (p): " + str(p) + " x 10^11 solar masses "+"\n")
        logf.write("zrange: {}\n".format(zr))
        #~ logf.write("Minimal_Number_of_Members: " + str(memLim) )
        logf.close()        
    else:
        print(logfile,"exists. Previous settings are reserved if not updated.")
        logdata = open(logfile).read()
        print(logdata)
        
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
    if trim in ["bEne","vdis","node","dEta","eta"]:
        outfile = nfile+"_"+trim+"_thr"+str(thres)+"_vdisL"+str(vdisLim)+"_M"+str(memLim)
    else:   ## sigpl
        outfile = nfile+"_"+trim        
        
    ###### save parameters of the analysis     
    logf2 = open(outfile+".gpar",'w')
    logf2.write("# Analysis Parameters: \n")
    logf2.write("zr = {} \n".format(zr))
    logf2.write("trim = {} \n".format(trim))
    logf2.write("thres = {} \n".format(thres))
    logf2.write("memLim = {} \n".format(memLim))
    logf2.write("memPlot = {} \n".format(memPlot))
    logf2.write("vdisLim = {} \n".format(vdisLim))
    logf2.write("vdisPlot = {} \n".format(vdisPlot))
    logf2.close()       
    
    
    if not os.path.exists(nfile+".gtre"):        
        if not os.path.exists(nfile+".gpwbe"):
            genpair(nfile,p)   # generate pairwise energy
        else:
            print("Pairwise energy file "+nfile+".gpwbe exists")
        
        gentree(nfile,cut=0)   # generate binary tree for the whole field, do not cut        
        pltree(nfile,outfile,yaxis="node",rmem=-1) 
        pltree(nfile,outfile,yaxis="vdis",rmem=-1) 
        #~ pltree(nfile,outfile,yaxis="bEne",rmem=-1)
        #~ pltree(nfile,outfile,yaxis="eta",rmem=-1)
    else:
        print("Tree file "+nfile+".gtre exists")
        
    if not os.path.exists(outfile+".glis"): 
        print(">>>>>> chopping the tree ")
        #~ print(memLim, outfile+".ssub")  
        ### different trimming approach  ??? 
        if trim in ["bEne","vdis","node","eta"]:
            keynodes0(nfile,outfile,thr=thres,debug=0,yulim="",memLim=memLim,trim=trim)  
        elif trim == "dEta":
            if not os.path.exists(nfile+".gepr"): 
                eprofile(nfile)  # generate energy profile for the local minima , mainly for the cluster 
            keynodes(nfile,outfile,thr=thres,debug=0,yulim="",memLim=memLim,vdisLim=vdisLim)  
            
        elif trim == "sigpl":
            if not os.path.exists(outfile+".gthr"):
                thr,tmp = mainbranch(nfile,mtyp=p)
            else:
                thr = np.loadtxt(outfile+".gthr")
            
            print(thr)            
            keynodes3(nfile,outfile,thr=thr, memLim=memLim)  # trace the tree and find keynodes  
            
    else: 
        print("Structure list file "+outfile+".glis exists")
    
    if os.path.exists(outfile+".gsum"):
        
        print(">>>>>> glis file is ready. Plot Figures ")
        pltree(nfile,outfile,thr=thres,yaxis=trim,memLim=memPlot) 
        #~ pltree(nfile,outfile,thr=thres,yaxis="node",memLim=memPlot) 
        #~ pltree(nfile,outfile,thr=thres,yaxis="bEne",memLim=memPlot) 
        ### distribution
        plot1d2(outfile,zrange=zr,memLim=memPlot,yulim="",vdisPlot=vdisPlot)        # plot the velocity histogram
        plot2d(outfile,zrange=zr,memLim=memPlot,limi="")
        if trim == "sigpl":
            plot2d(outfile,zrange=zr,memLim=memPlot,limi="",sign="cl")
        elif trim =="dEta":                
            ### simplified hierarchical structure of subs
            subtree_vdis(nfile,outfile,memLim=memPlot,zrange=zr,vdisLim=vdisPlot,vLim=3000)
            
        plotzdt(outfile,zrange=zr,memLim=memPlot,vdisLim=200)  
        genreg(outfile,zrange=zr,memLim=memPlot,vdisPlot=vdisPlot)
    else:
        print(outfile+".gsum does not exists.")
        
    end = timer()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Total running time: {:.2f}s".format(end - start))
    
if __name__ == "__main__":
    print("""
Blooming Tree Hierarchical Structure Analysis Script (2025.03 v1.85g)
----------------------------------------------------------
developed by Dr. Heng Yu of Beijing Normal University, 
with Prof. Antonaldo Diaferio of Turin University, and 
his Caustic Group: http://www.dfg.unito.it/ricerca/caustic/
----------------------------------------------------------

Usage:  blmtree.py list test    # check the input catalog and summarize
        blmtree.py list dEta=20 # cut the tree with the blooming tree algorithm, with a given threshold 20
        blmtree.py list dEta=20 z0=0.1 z1=0.2 # filter input catalog with a given redshift range and analyze it
        blmtree.py list dEta=50 mplot=20 # set the lowest member limit of the ploted structure to 20, 10 by default. 
        blmtree.py list dEta=50 vdislim=200 vdisplot=250 # set the smallest velocity dispersion limit of the detected structure to 200 km/s, 
                                                         # set the smallest velocity dispersion limit of the ploted structure to 250 km/s. 
        
        Once the script runs, it will creat a new catalog list.g?? with an self-increasing number suffix.
        If you want to make use of previous outputs, please run the script as below:
        The script will adopt parameters from the corresponding configuration file .gpar

        blmtree.py list.g01
        blmtree.py list.g01 dEta=5 mem=15

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
)
    parlist = ["mem","p","mplot","vdis","bene","binde","deta","eta","z0","z1","vdislim","vdisplot"]
    
    if len(sys.argv) == 2:
        if os.path.exists(sys.argv[1]):
            if os.path.exists(sys.argv[1]+".gpar"):
                main(sys.argv[1])            
            else:
                print("Threshold is missing and the configuration file {}.gpar is not found.".format(sys.argv[1]))
        else:
            print(sys.argv[1],"is not found")
            
    elif len(sys.argv) > 2:
        dic_key = {}
        if os.path.exists(sys.argv[1]):
            if sys.argv[2]=="sum" or sys.argv[2]=="test":
                nh,nmag,delim=precheck(sys.argv[1])
                #~ print("{}:{}:{}:".format(nh,nmag,delim))
                filterdata2(sys.argv[1],nhead=nh,sep=delim)
                
            elif "=" in sys.argv[2]:
                ## build paras dictionary 
                for par in sys.argv[2:]:
                    item = par.split("=")
                    if len(item) ==2 and item[0].lower() in parlist:
                        dic_key[item[0].lower()]=float(item[1])
                    else:
                        print("Key Error:",par)
                        print("Available paramemters are:",parlist)
                        quit()
                        
                ### setup initial parameters        
                if "p" not in dic_key:
                    dic_key["p"]=10
                    
                if "mem" not in dic_key:
                    dic_key["mem"]=9
                else:
                    dic_key["mem"]=int(dic_key["mem"])                    
                    
                if "mplot" not in dic_key:
                    dic_key["mplot"]=dic_key["mem"]
                else:
                    dic_key["mplot"]=int(dic_key["mplot"])             
                    
                sig = 0
                if "vdis" in dic_key:
                    dic_key["trim"] = "vdis"
                    thres = dic_key["vdis"]
                    sig = 1
                    
                if ("bene" in dic_key) or ("binde" in dic_key):
                    dic_key["trim"] = "bEne"
                    thres = dic_key["bene"]      
                    sig = 1        
                    
                if ("deta" in dic_key):
                    dic_key["trim"] = "dEta"
                    thres = dic_key["deta"]             
                    sig = 1
                    
                if ("eta" in dic_key):
                    dic_key["trim"] = "eta"
                    thres = dic_key["eta"]       
                    sig = 1
                    
                if "z0" not in dic_key and "z1" not in dic_key :
                    zr = ""
                elif "z0" in dic_key and "z1" not in dic_key :
                    zr = [0,dic_key["z1"]]
                elif "z0" not in dic_key and "z1" in dic_key :
                    zr = [dic_key["z0"],1]
                else:
                    zr = [dic_key["z0"],dic_key["z1"]]
                    
                if "vdislim" not in dic_key :
                    dic_key["vdislim"]  = 100
                    
                if "vdisplot" not in dic_key :
                    dic_key["vdisplot"]  = 200
                    
                if sig == 0:
                    dic_key["trim"] = "sigpl"
                    thres = 0
                print("Parameter Dict:",dic_key)
                main(sys.argv[1], trim=dic_key["trim"], thres = thres, p=dic_key["p"],
                    memLim=dic_key["mem"], memPlot=dic_key["mplot"],
                    zr=zr, vdisLim=dic_key["vdislim"], vdisPlot=dic_key["vdisplot"])             
            else:
                print('You entered {}'.format(sys.argv[1]))
    
        else:
            print('Can not find ', sys.argv[1])
                            
    elif len(sys.argv) < 2:
        print('Please input a file with the first three columns as below: \nRA (in degree), DEC (in degree), redshift (or radial velocity in km/s).')
    else:
        print('Too many parameters.')
