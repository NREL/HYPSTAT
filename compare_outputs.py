import pandas as pd
import numpy as np
from glob import glob
import os

def common_sums_match(df1, df2, count_no_shared_indices=False):
    #checks if the sums of rows with shared indices between two dataframes match
    #also returns False if df1 and df2 have no shared indices, unless count_no_shared_indices is set to True

    shared_indices = False
    for i in df1.index:
        if i in df2.index:
            shared_indices = True
            if not np.isclose(df1.loc[i].sum(),df2.loc[i].sum()):
                return False
    
    if shared_indices or count_no_shared_indices:
        return True
    else:
        return False
            

test_dir = 'Test Cases/test_case_grid/' #should have the slash on the end
comp_dir = 'Test Cases/test_case_inputs/' #should have the slash on the end

test_files = [os.path.normpath(f) for f in glob(test_dir+'*csv')]
comp_files = [os.path.normpath(f) for f in glob(comp_dir+'*csv')]

dif_found = False

for cf in comp_files:
    #print(cf)
    f = cf.split('\\')[-1]
    tf = test_dir+f
    try:
        if f=='Renewable_Capacity.csv' or tf.split('_')[-1]=='raw.csv' or f.startswith('Storage'):
            idx = [0,1]
        else:
            idx = 0
        tdata = pd.read_csv(tf,index_col=idx).astype(np.float64)
    except FileNotFoundError:
        print('{} file not in test directory!'.format(f))
        print()
        dif_found = True


for tf in test_files:
    dif_flagged = False
    f = tf.split('\\')[-1]
    cf = comp_dir+f
    if f=='Renewable_Capacity.csv' or tf.split('_')[-1]=='raw.csv' or f.startswith('Storage'):
            idx = [0,1]
    else:
        idx = 0
    tdata = pd.read_csv(tf,index_col=idx).astype(np.float64)
    try:
        if f=='Renewable_Capacity.csv' or cf.split('_')[-1]=='raw.csv' or f.startswith('Storage'):
            idx = [0,1]
        else:
            idx = 0
        cdata = pd.read_csv(cf,index_col=idx).astype(np.float64)
    except FileNotFoundError:
        print('{} file not in comparison directory!'.format(f))
        dif_found = True
        print()
        continue

    #compare indices and columns
    if len(tdata.index) != len(cdata.index) or (tdata.index != cdata.index).any():
        if not dif_flagged:
            print(f)
            dif_flagged = True
        print('\tIndices do not match!')
        dif_found = True

    if len(tdata.columns) != len(cdata.columns) or (tdata.columns != cdata.columns).any():
        if not dif_flagged:
            print(f)
            dif_flagged = True
        print('\tColumns do not match!')
        dif_found = True

    #if not tdata.equals(cdata):
    if tdata.values.shape != cdata.values.shape or not np.allclose(tdata.values,cdata.values):
        if not dif_flagged:
            print(f)
            dif_flagged = True
        print('\tData do not match!')
        dif_flagged = True
        dif_found = True

    
    #check sums for Zone_Capacities
    if f=='Zone_Capacities.csv' and dif_flagged:
        tsum = tdata.sum(axis=1)
        csum = cdata.sum(axis=1)
        if common_sums_match(tsum,csum):
            print('\tBut technology sums match')
    
    if dif_flagged:
        print()

if not dif_found:
    print('No differences found!')
print()
print('Done!')

