import numpy as np
import os
import shutil
import argparse
import pickle
import time
import sys
import itertools
import subprocess


my_str_cox = '''python3 ./cox_regression.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

# my_str_lr = '''python ./main_parallel.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''
my_str_lr = '''python3 ./logistic_regression.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

my_str_rf = '''python3 ./random_forest.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

parser = argparse.ArgumentParser()
parser.add_argument("-o","--o",help = 'out file', type = str)
parser.add_argument("-models","--models",help = 'model', type = str, nargs = '+')
parser.add_argument("-weeks","--weeks",help = 'week', type = float, nargs='+')
parser.add_argument("-i","--i",help = 'input data', type = str, nargs = '+')
args = parser.parse_args()
max_load = 10

if args.o:
    f_folder = args.o
else:
    f_folder = 'PredictiveAnalysisResults'
if not os.path.isdir(f_folder):
    os.mkdir(f_folder)

if 99 in args.weeks:
    args.weeks.remove(99)
    args.weeks.append([1,1.5,2])
if not args.weeks:
    args.weeks = [1.,2.]
print(args.weeks)
pid_list = []
for model in args.models:
    if model == 'cox':
        my_str =  my_str_cox
    elif model == 'LR':
        my_str = my_str_lr
    elif model == 'RF':
        my_str = my_str_rf

    for week in args.weeks:
        if isinstance(week, list):
            week = '_'.join([str(w) for w in week])
            wname = week.replace('.', 'a')
        else:
            week = week
            wname = week

        out_path = f_folder + '/' + model + '_week' + str(wname)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        for in_dat in args.i:

            path_out = out_path + '/' + in_dat + '/'
            fname = 'cdiff_lr.lsf'
            if not os.path.exists(path_out + 'coef' + '_ix_' + str(0) + '.pkl'):
                cmnd = my_str.format(0, in_dat, out_path, 'coef', week)
                args2 = cmnd.split(' ')
                print(args2)
                pid = subprocess.Popen(args2)
                pid_list.append(pid)
            else:
                print('Done: ' + path_out + 'coef' + '_ix_' + str(0) + '.pkl')
            while sum([x.poll() is None for x in pid_list]) >= max_load:
                time.sleep(30)
            for ii in np.arange(48):
                if os.path.exists(path_out + 'auc' + '_ix_' + str(ii) + '.pkl'):
                    print('Done: ' + path_out + 'auc' + '_ix_' + str(ii) + '.pkl')
                    continue
                else:
                    cmnd = my_str.format(ii, in_dat, out_path, 'auc', week)
                    args2 = cmnd.split(' ')
                    print(args2)
                    pid = subprocess.Popen(args2)
                    pid_list.append(pid)
                    while sum([x.poll() is None for x in pid_list]) >= max_load:
                        time.sleep(30)





