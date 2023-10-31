import pandas as pd
from glob import glob
import os

test_dir = 'test_case_outputs/' #should have the slash on the end
comp_dir = 'test_case_correct_results/' #should have the slash on the end

test_files = [os.path.normpath(f) for f in glob(test_dir+'*csv')]
comp_files = [os.path.normpath(f) for f in glob(comp_dir+'*csv')]

dif_found = False

for cf in comp_files:
    f = cf.split('\\')[-1]
    tf = test_dir+f
    try:
        tdata = pd.read_csv(tf)
    except FileNotFoundError:
        print('{} file not in test directory!'.format(f))
        dif_found = True


for tf in test_files:
    f = tf.split('\\')[-1]
    cf = comp_dir+f
    tdata = pd.read_csv(tf)
    try:
        cdata = pd.read_csv(cf)
    except FileNotFoundError:
        print('{} file not in comparison directory!'.format(f))
        dif_found = True
        continue

    if not tdata.equals(cdata):
        print('{} files do not match!'.format(f))
        dif_found = True

if not dif_found:
    print('No differences found!')
print()
print('Done!')

