Blooming Tree Hierarchical Structure Analysis Script
2025.03  Version 1.85g

This python stript is developed to detect structures with galaxy redshift data.
It takes the projected binding energy as the linking length, 
and provides three main analysis approaches to trim the hierarchical tree: 
1. the direct trimming (binding energy, velocity disperion, or eta), 
2. the sigma plateau (2011MNRAS.412..800S), when no trimming threshold is appointed
3. and the blooming tree (2018ApJ...860..118Y).
It is developed by Dr. Heng Yu of Beijing Normal University, 
with Prof. Antonaldo Diaferio of Turin University, 
and his Caustic Group: http://www.dfg.unito.it/ricerca/caustic/

NOTE: This tool only works in the terminal.

##  Usage 

- blmtree.py LIST test    # check and summarize the input catalog, which is named as "LIST" for example
- blmtree.py LIST bEne=-1 # cut the tree with a given binding energy threshold.
- blmtree.py LIST vdis=50 # cut the tree with a given velocity disperion.
- blmtree.py LIST mem=10  # cut the tree with the sigma plateau approach.
- blmtree.py LIST dEta=20 # cut the tree with the blooming tree algorithm, with a given threshold 20.
- blmtree.py LIST dEta=20 z0=0.1 z1=0.2 # filter input catalog with a given redshift range and analyze it
- blmtree.py LIST dEta=50 mem=20 # 20 is the lowest limit of the structure size, 10 by default. 

Once the script runs, it will creat a new catalog LIST.g?? with an self-increasing suffix.
If you want to make use of previous outputs, please run the script as below.
The script will adopt parameters from the corresponding .gpar file.

- blmtree.py LIST.g00
- blmtree.py LIST.g00 dEta=50
- blmtree.py LIST.g00 dEta=50 mem=15
        
##  Options : case-insensitive 

- [p]    : the coefficient between kinetic energy and potential energy, related to the mass of the particle, 10 by default
- [vdis] : threshold of velocity dispersion for tree trimming
- [bEne] : threshold of binding energy for tree trimming
- [bindE]: threshold of binding energy for tree trimming
- [dEta] : threshold of delta eta for tree trimming, 20 by defalut 
- [Eta]  : threshold of eta for tree trimming
- [z0,z1]: redshift range of analysis
- [mem]  : minimal number of members in searching groups, 10 by default
- [mplot]: minimal  number of members in ploting groups,  10 by default
- [vdisLim] : minimal velocity dispersion in searching groups, 100 km/s by default
- [vdisPlot]: minimal velocity dispersion in ploting groups, 200 km/s by default

##  FILES  

- LIST : input file with 3 columns(ra,dec,redshift) at least, additional columns and lines started with # will be ignored. 
The first line can be a summary line containing: number of records, center of RA, center of DEC, target redshift (or cz).
- LIST.g00 : filtered and uniformed input catalog. The number suffix will grow automatically when the program runs multiple times to avoid overwriting files.

#### files LIST.g00.g???? are temporary files that are only generated when the program is run for the first time and are no longer needed for subsequent analyses.

- LIST.g00.gpwbe : pairwise projected binding energy. There are 3 columns: ID1, ID2, engergy in binary format to save space. It is not needed once the tree file (*.gtre) is generated.
- LIST.g00\_dEta\_\*.getap : eta profiles to search for key nodes. Each row contains eta values of all nodes in the rountine, starting with the minID.

#### files LIST.g00.g??? are output data.

- LIST.g00.gpar : parameter file, key words and values are seperated with a colon.
- LIST.g00.gtre : hierarchy tree, 14 columns : nodeid(start from 0) lkid rkid bind_energy(ascending) number_of_nodes redshift(cz) Vdis r_avg r_cluster distance ra dec eta delta_eta
- LIST.g00.gepr : binding energy profile, 3 columns: leave_id, x(of the tree plot), binding energy

- LIST.g00.*.glis : membership of structures, 4 columns: RA, DEC, z, ID_str
- LIST.g00.*.gsum : summary of structures, 7 columns: Cl_ID, sub_ID, num, RA, DEC, v, vdis.  Note: Cl_ID is only valid for the sigma plateau approach.
- LIST.g00.*.gsub : details of structures for debug, 16 columns.

- LIST.g00.*.gthr : threshold output of the sigma plateau approach, containing 11 numbers: v_dis,  error of v_dis,  ID_cl, ID_sub, Energy of cluster threshold,  Energy of substructure threshold, v_dis of cluster threshold, v_dis of subtructure threshold, plateau weight in the GMM fitting, weight/sigma of Gaussian component, p-value
- LIST.g00.*.gbrc : branch tracing result of the BLT approach, 4 columns: ID of minimum, number of members under sub ID, sub ID, radius of sub

#### files LIST.g00.??? are output figures.

- \*\_dtreep.pdf : tree profile, based on the file LIST.g00.gepr.
- \*\_k\*\_vhst.png : velocity dispersion profile of the main brach, valid for the sigma plateau approach.
- \*\_dtree\_\*.pdf" : binary tree with identified structures
- \*\_stree\_\*.pdf' : simplified binary tree with identified structures
- \*\_zDt\_\*.pdf' : the redshift diagram of the tree analysis result in Cartesian coordinate system
- \*\_1D2\_\*.pdf : redshift distribution of groups
- \*\_2D\_\*.pdf : spatial distribution in Eclidean space
- \*.reg : ds9 region files with structure members
