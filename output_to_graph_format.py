"""
Quick and dirty program to take the intermediate result files and reorganize them
into the format used by the tikZ code that generates the graphs in the paper

Before running this code, make sure that you have run automated_full_aux.py to completion
and all result files are in the directory that EXP_RESULTS_DIR points to below

This file certainly doesn't follow best coding practices, and is meant to be a quick way to automate
the experiment workflow. If you encounter bugs, don't spend your time trying to fix them.
Please report bugs via email to the first author of the paper and we will do our best to fix them promptly.
"""
import os

SETTINGS_FILE = 'exp_settings.csv'
EXP_RESULTS_DIR = 'experiment_results'
EXP_RESULTS_PREFIX = EXP_RESULTS_DIR + "/"
LATEX_RESULTS_DIR = 'results_for_latex'
LATEX_RESULTS_PREFIX = LATEX_RESULTS_DIR + "/"

EXP_PREFIXES = ['eta_mu_100', 'eta_mu_250', 'eta_mu_500', 'lucene', 'zipf_nkw_100', 'zipf_nkw_250' \
                       'zipf_nkw_500', 'num_files_small', 'num_files_large', 'num_files_runtime', 'dist', 'aux']

ETA_MU_LIST = ["eta_mu_100_10.txt", "eta_mu_100_100.txt", "eta_mu_100_250.txt", "eta_mu_100_500.txt", "eta_mu_100_750.txt", "eta_mu_100_1000.txt"]
ETA_MU_250_LIST = ["eta_mu_250_10.txt", "eta_mu_250_100.txt", "eta_mu_250_250.txt", "eta_mu_250_500.txt", "eta_mu_250_750.txt", "eta_mu_250_1000.txt"]
ETA_MU_500_LIST = ["eta_mu_500_10.txt", "eta_mu_500_100.txt", "eta_mu_500_250.txt", "eta_mu_500_500.txt", "eta_mu_500_750.txt", "eta_mu_500_1000.txt"]
# can reuse eta_mu_100.txt because the attack is deterministic and the settings would be the same as aux_100, saves us some time
AUX_LIST = ["aux_10.txt", "eta_mu_100_100.txt", "aux_200.txt", "aux_300.txt", "aux_400.txt", "aux_500.txt"]
NUM_FILES_RUNTIME_LIST = ["num_files_small_1.txt", "num_files_small_10.txt", "num_files_small_100.txt", "num_files_runtime_1000.txt", \
                          "num_files_runtime_10000.txt", "num_files_large_100k.txt"]
NUM_FILES_SMALL_LIST = ["num_files_small_1.txt", "num_files_small_10.txt", "num_files_small_50.txt", "num_files_small_100.txt"]
NUM_FILES_LARGE_LIST = ["num_files_large_30k.txt", "num_files_large_40k.txt", "num_files_large_50k.txt", "num_files_large_60k.txt", \
                        "num_files_large_70k.txt", "num_files_large_80k.txt", "num_files_large_90k.txt", "num_files_large_100k.txt"]
LUCENE_LIST = ["lucene_10.txt", "lucene_100.txt", "lucene_250.txt", "lucene_500.txt", "lucene_750.txt", "lucene_1000.txt"]
DIST_LIST = ["dist_0.txt", "dist_60.txt", "dist_120.txt", "dist_180.txt", "dist_240.txt", "dist_300.txt", "dist_360.txt", "dist_420.txt", \
            "dist_480.txt", "dist_540.txt", "dist_600.txt"]

ETA_MU_LINES = [4, 12]
AUX_LINES = [4, 12]
DIST_LINES = []
NUM_FILES_LINES = []
LUCENE_LINES = [4, 12]

def _get_lines_to_skip(filename):
    if 'eta_mu' in filename: return ETA_MU_LINES
    if 'aux' in filename: return AUX_LINES
    if 'dist' in filename: return DIST_LINES
    if 'num_files' in filename: return NUM_FILES_LINES
    if 'lucene' in filename: return LUCENE_LINES
    print("Error when getting the right lines to skip")
    exit(1)

def most_experiments(file, lines_to_skip=[]):
    '''
    returns a line to write to the combined results file
    '''
    with open(file, 'r') as f:
        count = 0
        res = ""
        for line in f:
            count += 1
            # skip lines that don't have to do with accuracy
            if not line.startswith("0"):
                continue
            # skip lines that deal with runtime
            if count in lines_to_skip:
                continue
            res += line.strip() + " "
        return res

def zipf(files):
    results = ""
    for file in files:
        with open(file, 'r') as f:
            count = 0
            res = ""
            for line in f:
                # skip lines that don't have to do with accuracy
                if not line.startswith("0"):
                    continue
                count += 1
                res += line.strip() + " "
            results += res
    return results  

def run_zipf():
    '''
    runs the zipf() function enough to get a full line,
    returns all of them in one go in the all_lines list
    '''
    params = ["10", "100", "250", "500", "750", "1000"]
    all_lines = []
    prefix100 = EXP_RESULTS_PREFIX + "zipf_nkw_100_"
    prefix250 = EXP_RESULTS_PREFIX + "zipf_nkw_250_"
    prefix500 = EXP_RESULTS_PREFIX + "zipf_nkw_500_"
    suffix = ".txt"
    for num in params:
        file_list = [prefix100 + num + suffix, prefix250 + num + suffix, prefix500 + num + suffix]
        all_lines.append(zipf(file_list) + "\n")
    return all_lines

def num_files_runtime(files):
    '''
    give this function num_files_1, 10, 100, ..., 100000
    '''
    res_line = []
    num_files = []
    for file in files:
        last_line = ""
        with open(EXP_RESULTS_PREFIX + file, 'r') as f:
            for curr_line in f:
                curr_line = curr_line.strip()
                if curr_line == "":
                    res_line.append(last_line)
                    for _ in range(3):
                        f.readline()
                    num_files.append(f.readline().strip().split(" ")[-1])
                    continue
                last_line = curr_line
    return res_line, num_files

def num_files(files):
    '''
    can do both small and large
    '''
    res_line = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                if not line.startswith("0"):
                    continue
                res_line.append(line.strip())

if not os.path.exists(LATEX_RESULTS_DIR):
    os.mkdir(LATEX_RESULTS_DIR)

# all eta_mu experiments
res_filenames = ["eta_mu.txt", "eta_mu_250.txt", "eta_mu_500.txt", "lucene.txt"]
EXPS_TO_PROCESS = [ETA_MU_LIST, ETA_MU_250_LIST, ETA_MU_500_LIST, LUCENE_LIST]
for curr_filename, file_list in zip(res_filenames, EXPS_TO_PROCESS):
    with open(LATEX_RESULTS_PREFIX + curr_filename, 'w') as f:
        f.write("eta_mu avg_s_acc avg_uf_acc avg_spuf_acc avg_ufid_acc avg_spufid_acc\n")
        params = ["10", "100", "250", "500", "750", "1000"]
        for file, param in zip(file_list, params):
            file = EXP_RESULTS_PREFIX + file
            f.write(param + " " + most_experiments(file, _get_lines_to_skip(file)) + "\n")

# aux.txt
for curr_filename in res_filenames:
    with open(LATEX_RESULTS_PREFIX + "aux.txt", 'w') as f:
        f.write("eta_mu avg_s_acc avg_uf_acc avg_spuf_acc avg_ufid_acc avg_spufid_acc\n")
        params = ["10", "100", "200", "300", "400", "500"]
        files = AUX_LIST
        lines = []
        for file in files:
            file = EXP_RESULTS_PREFIX + file
            lines.append(most_experiments(file, AUX_LINES))
        for line, param in zip(lines, params):
            f.write(param + " " + line + "\n")

# num_files_runtime.txt
with open(LATEX_RESULTS_PREFIX + 'num_files_runtime.txt', 'w') as f:
    f.write("num_files avg_runtime\n")
    lines, num_files = num_files_runtime(NUM_FILES_RUNTIME_LIST)
    for line, filenum in zip(lines, num_files):
        f.write(filenum + " " + line + "\n")

# num_files_small.txt
with open(LATEX_RESULTS_PREFIX + 'num_files_small.txt', 'w') as f:
    f.write("num_files avg_ufid_acc avg_spufid_acc avg_runtime\n")
    num_files = ["1", "10", "50", "100"]
    files = NUM_FILES_SMALL_LIST
    lines = []
    for file in files:
        file = EXP_RESULTS_PREFIX + file
        lines.append(most_experiments(file))
    for line, filenum in zip(lines, num_files):
        f.write(filenum + " " + line + "\n")

# num_files_large.txt
with open(LATEX_RESULTS_PREFIX + 'num_files_large.txt', 'w') as f:
    f.write("num_files avg_ufid_acc avg_spufid_acc\n")
    num_files = ["30000", "40000", "50000", "60000", "70000", "80000", "90000", "100000"]
    files = NUM_FILES_LARGE_LIST
    lines = []
    for file in files:
        file = EXP_RESULTS_PREFIX + file
        lines.append(most_experiments(file))
    for line, filenum in zip(lines, num_files):
        f.write(filenum + " " + line + "\n")

#zipf.txt
with open(LATEX_RESULTS_PREFIX + 'zipf.txt', 'w') as f:
    f.write("eta_mu avg_ufid_acc_100 avg_spufid_acc_100 avg_ufid_acc_250 avg_spufid_acc_250 avg_ufid_acc_500 avg_spufid_acc_500\n")
    lines = run_zipf()
    params = ["10", "100", "250", "500", "750", "1000"]
    for line, param in zip(lines, params):
        f.write(param + " " + line)

# dist.txt
with open(LATEX_RESULTS_PREFIX + 'dist.txt', 'w') as f:
    f.write("std_dev avg_ufid_acc avg_spufid_acc\n")
    dist_nums = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    files = DIST_LIST
    lines = []
    for file in files:
        file = EXP_RESULTS_PREFIX + file
        lines.append(most_experiments(file))
    for line, dist in zip(lines, dist_nums):
        f.write(str(dist) + " " + line + "\n")
