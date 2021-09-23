import numpy as np
import os
import shutil
import argparse
import pickle
import time
import sys
import itertools

my_str = '''
#!/bin/bash
#BSUB -J cdiff
#BSUB -o output/${6}${1}${4}/cdiff-${0}-%J.out
#BSUB -e output/${6}${1}${4}/cdiff-${0}-%J.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q normal

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here

cd /PHShome/jjd65/cdiff_analyses/scripts
'''

my_str_cox = '''python3 ./main_cox_fast.py -ix {0} -i {1} -o {2} -type {3} -week {4} -folds {5}'''

my_str_cox2 = '''python3 ./main_cox_2.py -ix {0} -i {1} -o {2} -type {3} -week {4} -folds {5}'''
# my_str_lr = '''python ./main_parallel.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''
my_str_lr = '''python3 ./main_lr_fast.py -ix {0} -i {1} -o {2} -type {3} -week {4} -folds {5}'''

my_str_rf = '''python3 ./rf_nested.py -ix {0} -i {1} -o {2} -type {3} -week {4} -folds {5}'''

parser = argparse.ArgumentParser()
parser.add_argument("-o","--o",help = 'out file', type = str)
parser.add_argument("-models","--models",help = 'model', type = str, nargs = '+')
parser.add_argument("-folds","--folds",help = 'use_folds', type = int, nargs = '+')
parser.add_argument("-weeks","--weeks",help = 'week', type = float, nargs='+')
parser.add_argument("-i","--i",help = 'input data', type = str, nargs = '+')
args = parser.parse_args()

if 99 in args.weeks:
    args.weeks.remove(99)
    args.weeks.append([1,1.5,2])


for model in args.models:
    if model == 'cox':
        my_str = my_str + my_str_cox
    elif model == 'LR':
        my_str = my_str + my_str_lr
    elif model == 'RF':
        my_str = my_str + my_str_rf
    elif model == 'cox2':
        my_str = my_str + my_str_cox2

    for week in args.weeks:
        if isinstance(week, list):
            week = '_'.join([str(w) for w in week])
            wname = week.replace('.', 'a')
        else:
            week = week
            wname = week

        for use_folds in args.folds:
            if use_folds == 1:
                f_folder = 'PredictiveAnalysisResults_Folds'
            else:
                f_folder = 'PredictiveAnalysisResults_2'
            if not os.path.isdir(f_folder):
                os.mkdir(f_folder)

            out_path = f_folder + '/' + model + '_week' + str(wname)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)

            for in_dat in args.i:
                if not os.path.isdir('output'):
                    os.mkdir('output')
                if not os.path.isdir('output/' +model + in_dat + str(week)):
                    os.mkdir('output/' +model + in_dat + str(week))
                path_out = out_path + '/' + in_dat + '/'
                fname = 'cdiff.lsf'
                if not os.path.exists(path_out + 'coef' + '_ix_' + str(0) + '.pkl'):
                    f = open(fname, 'w')
                    f.write(my_str.format(0, in_dat, out_path, 'coef', week, use_folds, model))
                    f.close()
                    os.system('bsub < {}'.format(fname))
                    time.sleep(0.01)
                for ii in np.arange(48):
                    if os.path.exists(path_out + 'auc' + '_ix_' + str(ii) + '.pkl'):
                        continue
                    else:
                        f = open(fname, 'w')
                        f.write(my_str.format(ii, in_dat, out_path, 'auc', week, use_folds, model))
                        f.close()
                        os.system('bsub < {}'.format(fname))
                        time.sleep(0.001)





