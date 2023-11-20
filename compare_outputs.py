import pandas as pd
import numpy as np
from glob import glob
import os

test_dir = 'test_case_pipelines_no_imports/' #should have the slash on the end
comp_dir = 'test_case_no_imports/' #should have the slash on the end

test_files = [os.path.normpath(f) for f in glob(test_dir+'*csv')]
comp_files = [os.path.normpath(f) for f in glob(comp_dir+'*csv')]

dif_found = False

for cf in comp_files:
    #print(cf)
    f = cf.split('\\')[-1]
    tf = test_dir+f
    try:
        if f=='Renewable_Capacity.csv' or tf.split('_')[-1]=='raw.csv':
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
    if f=='Renewable_Capacity.csv' or tf.split('_')[-1]=='raw.csv':
            idx = [0,1]
    else:
        idx = 0
    tdata = pd.read_csv(tf,index_col=idx).astype(np.float64)
    try:
        if f=='Renewable_Capacity.csv' or cf.split('_')[-1]=='raw.csv':
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
    
    if dif_flagged:
        print()

if not dif_found:
    print('No differences found!')
print()
print('Done!')

