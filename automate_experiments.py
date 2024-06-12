import numpy as np
import pickle
from collections import Counter
from scipy.optimize import linear_sum_assignment as hungarian
import sys
import os
import time

def extract_probs_enron():
    '''
    gets probabilities from enron_db.pkl
    :return kw_dict: keys are the keywords and values are the probs
    '''
    data = "enron_db.pkl"
    with (open(data, "rb")) as p:
        while True:
            try:
                dataset, kw_dict = pickle.load(p)
                return dataset, kw_dict
            except EOFError:
                break

def extract_probs_lucene():
    '''
    gets probabilities from ;ucene_db.pkl
    :return kw_dict: keys are the keywords and values are the probs
    '''
    data = "lucene_db.pkl"
    with (open(data, "rb")) as p:
        while True:
            try:
                dataset, kw_dict = pickle.load(p)
                return dataset, kw_dict
            except EOFError:
                break

dataset_enron, kw_dict_enron = extract_probs_enron()
dataset_lucene, kw_dict_lucene = extract_probs_lucene()

# parameters used for all experiments
NUM_WEEKS = 100 # must be a multiple of 2
SPLIT_TIME = int(NUM_WEEKS / 2)
NUM_EXPERIMENTS = 30
PRINT_EACH_EXPERIMENT = False
ZIPF_PARAM = 1.1

EXP_DIRECTORY = "experiment_results"

def generate_week_update(probs, update_budget):
    '''
    generates one week of update patterns based on a multinomial distribution
    :param probs: list of probabilities for each kw to be updated. length must be equal to kw list
    :param update_budget: total number of updates this week (mu_k)
        NOTE: if mu varies with time, caller must account for this when calling this function
    :return: list of size update_budget, [[kw, file], [kw, file], ..., [kw, file]]
    '''
    # check parameters
    if len(probs) != len(chosen_ids):
        raise AssertionError
    
    probs /= sum(probs) # normalize so probs sum to 1

    n_up_this_week = np.random.poisson(update_budget)

    # generate update choices
    updates_this_week = np.random.choice(chosen_ids, n_up_this_week, p=probs)

    # attach file identifiers to the choices
    updates_with_files_this_week = []
    for kw in updates_this_week:
        file_id = get_file_zipf(kw) if ZIPF_EXP else get_file_norm(kw)
        updates_with_files_this_week.append([kw, file_id])

    return np.array(updates_with_files_this_week)

def generate_week_search(probs, search_budget):
    '''
    generates one week of search patterns based on a multinomial distribution
    :param probs: list of probabilities for each kw to be searched. length must be equal to kw list
    :param search_budget: total number of searches this week (eta_k)
        NOTE: if eta varies with time, caller must account for this when calling this function
    :return: list of size search_budget [kw, kw, ..., kw]
    '''
    # check parameters
    if len(probs) != len(chosen_ids):
        raise AssertionError
    
    probs /= sum(probs) # normalize so probs sum to 1

    n_qr_this_week = np.random.poisson(search_budget)
    
    # generate update choices
    searches_this_week = np.random.choice(chosen_ids, n_qr_this_week, p=probs)

    return searches_this_week.tolist()

def print_settings():
    print("Experiment settings:")
    print("Eta:", NUM_SEARCHES)
    print("Mu : ", NUM_UPDATES)
    print("Number of files:", NUM_FILES)
    print("Distribution scale:", DIST_SCALE)
    print("Number of weeks:", NUM_WEEKS)
    print("Number of unique keywords client is querying/updating:", NUM_KEYWORDS)
    print("Auxiliary traces and observed traces are split at week", SPLIT_TIME)
    print()

def get_file_norm(kw):
    '''
    helper function that returns the appropriate file for a particular keyword based on a normal
    distribution with its unique midpoint 
    '''
    file_range = np.arange(NUM_FILES)
    midpoint = int(NUM_FILES / 2)
    # apply a shift based on which index the kw is at in kw_list
    shift = 0
    try:
        shift = kw_ids.index(kw, shift)
    except ValueError:
        print("keyword wasn't in kw_list")
        exit(0)

    file_range = np.roll(file_range, shift)

    index = int(np.random.normal(loc=midpoint, scale=DIST_SCALE))
    file = file_range[index] if 0 <= index < NUM_FILES else file_range[midpoint]

    return file

def get_file_zipf(kw):
    '''
    helper function that returns the appropriate file for a particular keyword based 
    on sampling from a zipf distribution 
    '''
    file_range = np.arange(NUM_FILES)
    midpoint = int(NUM_FILES / 2)
    # apply a shift based on which index the kw is at in kw_list
    shift = 0
    try:
        shift = kw_ids.index(kw, shift)
    except ValueError:
        print("keyword wasn't in kw_list")
        exit(0)

    file_range = np.roll(file_range, shift)
    index = -1
    while True:
        index = np.random.zipf(ZIPF_PARAM)
        if 0 <= index < NUM_FILES: break # if sample was out of range, just resample
    file = file_range[index]

    return file

def get_tag(kw_id, inverted_index):
    '''
    for a given kw_id, gets its tag that the server calculated based on its unique access pattern
    more sophisticated tagging strategies are possible if attacking schemes which obfuscate this info
    '''
    access_pattern = inverted_index[kw_id]
    try:
        return seen_access_patterns.index(access_pattern)
    except ValueError:
        seen_access_patterns.append(access_pattern)
        return len(seen_access_patterns) - 1

def construct_inverted_index():
    '''
    :return: an inverted index dict of form {kw_id: access pattern}
    '''
    inverted_index = {}
    for kw_id in range(NUM_KEYWORDS):
        inverted_index[chosen_ids[kw_id]] = [doc_id for doc_id, doc_kws in enumerate(DATASET) if chosen_kws[kw_id] in doc_kws] # kw: AP
    return inverted_index

def calculate_aux_probs_search(trace):
    ''' Calculates the number of times each keyword was searched each week
    and reports it as a probability
    [week1, week2, ...] where week1 = [kw1_prob, kw2_prob, ...]

    :return: list of lists (num_ids x num_weeks)
    '''
    n_tags = NUM_KEYWORDS
    n_weeks = len(trace)
    kw_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_kws in enumerate(trace):
        if len(weekly_kws) > 0:
            counter = Counter([chosen_ids.index(kw) for kw in weekly_kws])
            for key in counter:
                prob = counter[key] / len(weekly_kws)
                kw_trend_matrix[key, i_week] = prob
    return kw_trend_matrix

def calculate_observations_search(trace):
    '''
    Calculates the number of times each tag was observed each week
    [week1, week2, ...] where week1 = [tag1_obs, tag2_obs, ...]

    For example, if tag 41 was searched twice in week 0, then
    tag_trend_matrix[41][0] = 2/len(week_0)

    :return: list of lists (num_weeks x num_tags)
    '''
    n_tags = len(seen_access_patterns)
    n_weeks = len(trace)
    tag_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_tags in enumerate(trace):
        if len(weekly_tags) > 0:
            counter = Counter(weekly_tags)
            for key in counter:
                tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
    return tag_trend_matrix

def calculate_aux_probs_update_only(trace):
    '''
    Calculates the number of times each keyword was searched each week and reports it
    as a probability
    [week1, week2, ...] where week1 = [kw1_prob, kw2_prob, ...]

    :return: list of lists (num_ids x num_weeks)
    '''
    n_tags = NUM_KEYWORDS
    n_weeks = len(trace)
    kw_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_kws in enumerate(trace):
        if len(weekly_kws) > 0:
            counter = Counter([chosen_ids.index(kw) for kw in weekly_kws])
            for key in counter:
                kw_trend_matrix[key, i_week] = counter[key] / len(weekly_kws)
    return kw_trend_matrix

def calculate_aux_probs_update(trace):
    '''
    Calculates the number of times each keyword + file-id pair was updated each week and reports it
    as a probability

    :return: matrix of size num_kws x num_files X len(trace)
        rows and columns index which kw + file-id pair we are on.
        probability for a given week is stored along the depth.
        i.e. arr(1, 2, 4) = probability of kw 1 being updated into file 2 in week 4
    '''
    n_tags = NUM_KEYWORDS
    n_weeks = len(trace)
    kw_trend_matrix = np.zeros((n_tags, NUM_FILES, n_weeks))
    for i_week, weekly_kws in enumerate(trace):
        counts = {}
        if len(weekly_kws) > 0:
            for update in weekly_kws:
                kw_id = update[0]
                file_id = update[1]
                key = (chosen_ids.index(kw_id), file_id)
                if key not in counts:
                    counts[key] = 1
                else: 
                    counts[key] += 1
            for key in counts:
                prob = counts[key] / len(weekly_kws)
                kw_trend_matrix[key[0], key[1], i_week] = prob
    return kw_trend_matrix

def calculate_observations_update_only(trace):
    '''
    Calculates the number of times each tag was observed each week
    [week1, week2, ...] where week1 = [tag1_obs, tag2_obs, ...]

    :return: list of lists (num_weeks x num_tags)
    '''
    n_tags = len(seen_access_patterns)
    n_weeks = len(trace)
    tag_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_tags in enumerate(trace):
        if len(weekly_tags) > 0:
            counter = Counter(weekly_tags)
            for key in counter:
                tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
    return tag_trend_matrix

def calculate_observations_update(trace):
    '''
    Calculates the number of times each tag + file-id pair was observed each week

    :return: matrix of size num_kws x num_files x num_weeks/2
        rows and columns index which kw + file-id pair we are on.
        number of observations for a given week are stored along the depth.
        i.e. arr(1, 2, 4) = number of times tag 1 was updated into file 2 in week 4
    '''
    mat = np.zeros((len(seen_access_patterns), NUM_FILES, len(trace)))
    for i in range(len(trace)): # iterate over weeks in the trace
        week = trace[i]
        for update in week: # iterate over updates in the week
            mat[update[0], update[1], i] += 1 / len(week)
    return mat

def calculate_file_id_observations(trace):
    '''
    Calculates the number of times each file was updated in each week

    :return: matrix of size num_files x num_weeks
        mat[129][12] = number of times file 129 was updated in week 12
    '''
    mat = np.zeros((NUM_FILES, NUM_WEEKS))
    week_ctr = 0
    for week in trace:
        for update in week:
            file_id = update[1]
            mat[file_id][week_ctr] += 1
        week_ctr += 1
    return mat

def extract_unseen_updates_ind_full(trace, up_trace):
    '''
    When a client issues a search, all previous updates for that token are
    leaked to the server.

    This function takes an update+file_id trace and removes all updates which do not yet
    appear in the provided search patterns

    Searching for tag i in week 10 reveals all updates for i in weeks 0-10
    Searching for tag i in week 10 must NOT reveal updates for i in weeks 11-100

    :param trace: the search patterns observed by the server
    :param up_trace: the full update patterns observed by the server
    '''
    rm_list = {}
    for i_week in range(len(trace)):
        for pair in up_trace[i_week]:
            tag = pair[0]
            if not any(tag in week for week in trace[i_week:]):
                rm_list[tag] = i_week
    wknum = 0
    ind_list = []
    for week in up_trace:
        for pair in week:
            tag = pair[0]
            if tag in rm_list and wknum >= rm_list[tag]:
                ind_list.append((wknum, week.tolist().index(pair.tolist())))
        wknum += 1
    for ind_pair in reversed(ind_list):
        up_trace[ind_pair[0]] = np.delete(up_trace[ind_pair[0]], ind_pair[1], 0)
    return up_trace

def run_hungarian_attack_given_matrix(c_matrix):
    """
    From Oya SAP code
    Runs the Hungarian algorithm with the given cost matrix
    :param c_matrix: cost matrix, (n_keywords x n_tags)
    :return: query_predictions_for_each_tag: dict of tag -> kw_id
    """
    row_ind, col_ind = hungarian(c_matrix)
    predictions_for_each_tag = {}
    for tag, keyword in zip(col_ind, row_ind):
        predictions_for_each_tag[tag] = keyword
    return predictions_for_each_tag

def cost_matrix_update_only(aux_update_frequencies, obs_update_patterns):
    # build trend matrix for observed frequencies. trends_tags[i,j] = # of times kw i was observed in week j
    tt_update = np.array(calculate_observations_update_only(obs_update_patterns))

    n_up_per_week = [len(trace) for trace in obs_update_patterns]
    log_c_matrix = np.zeros((NUM_KEYWORDS, len(tt_update)))
    # fill the cost matrix
    for i_week, n_up in enumerate(n_up_per_week): # gives a counter that goes from 0-SPLIT_TIME along with mu for that week
        probabilities = aux_update_frequencies[:, i_week].copy() # len = num_weeks/2

        # replace all zero probabilities with a tiny value
        probabilities[probabilities == 0] = 10**-9
        log_c_matrix += (n_up * tt_update[:, i_week]) * np.log(np.array([probabilities]).T) # sum of eta_k * freq(tag_k) * log(aux_freq(tag_k))
    return -log_c_matrix

def cost_matrix_search(aux_search_frequencies, obs_search_patterns):
    # build trend matrix for observed frequencies. trends_tags[i,j] = # of times kw i was observed in week j
    trends_tags = np.array(calculate_observations_search(obs_search_patterns)) # (#_chosen_kws x num_weeks/2)

    nq_per_week = [len(trace) for trace in obs_search_patterns] # list of all eta_k
    log_c_matrix = np.zeros((NUM_KEYWORDS, len(trends_tags)))
    # fill the cost matrix
    for i_week, nq in enumerate(nq_per_week): # (0, 20), (1, 20), ... (n_weeks/2 - 1, 20)
        # picks a column from the normalized trend matrix (each row is a kw, each col is a week)
        # so it is taking the frequencies from all kws for one week
        probabilities = aux_search_frequencies[:, i_week].copy()

        # replace all zero probabilities with a tiny value
        probabilities[probabilities == 0] = 10**-9

        log_c_matrix += (nq * trends_tags[:, i_week]) * np.log(np.array([probabilities]).T) # sum of eta_k * freq(tag_j) * log(aux_freq(kw_i))
    return -log_c_matrix

def cost_matrix_update_file_id(aux_update_frequencies, obs_update_patterns):
    '''
    w = tag_id
    x = kw_id
    y = file_id
    z = week_id
    sum of mu_(y,z) * freq_(w,y,z) * log(aux_freq_(x,y,z))
    '''
    # build trend matrix for observations
    trends_tags = np.array(calculate_observations_update(obs_update_patterns))

    # build matrix for file update observations. needed to calculate mu
    file_freqs = calculate_file_id_observations(obs_update_patterns)

    n_up_per_week = [len(trace) for trace in obs_update_patterns]
    log_c_matrix = np.zeros((NUM_KEYWORDS, len(trends_tags)))
    for z_week, n_up in enumerate(n_up_per_week):
        # display progress (this loop can take a while)
        interval = 10# if NUM_FILES < 5000 else 1
        if PRINT_EACH_EXPERIMENT and (NUM_KEYWORDS > 100 or NUM_FILES > 10000) \
            and (z_week % interval == 0 or z_week == SPLIT_TIME - 1):
            print("z = " + str(z_week))

        for y in range(NUM_FILES):
            # we take the slice because that gets all kws/tags in one go. If we didn't slice we'd have to iterate over all w
            mu = file_freqs[y][z_week] # number of times file y was updated in week z
            freq = trends_tags[:, y, z_week] # observed updates for all tags, file y, week z

            probabilities = aux_update_frequencies[:, y, z_week].copy() # aux probs, all kws, file y, week z
            probabilities[probabilities == 0] = 10**-9 # replace all zero probabilities with a tiny value

            aux = np.log(np.array(probabilities)).T # size (nkw,)
            aux = np.reshape(aux, (NUM_KEYWORDS, 1)) # change size to (nkw, 1) so matrix multiplication works correctly
            log_c_matrix += (mu * freq * aux)
    return -log_c_matrix

def calculate_accuracy(obs_patterns, full_patterns, predictions_for_each_tag):
    '''
    Prints the accuracy of the attacks
    full_search_patterns[SPLIT_TIME:] is the ground truth

    :param: obs_patterns either obs_search_patterns or obs_update_patterns
    :param: full_patterns either full_search_patterns or full_update_patterns
    '''
    # goes through each tag in obs_search_patterns and appends its predicted kw_id to the list
    predictions_for_each_obs = []
    for weekly_tags in obs_patterns:
        try:
            predictions_for_each_obs.append([chosen_ids[predictions_for_each_tag[tag_id]] for tag_id in weekly_tags])
        except IndexError:
            type, value, traceback = sys.exc_info()
            print(value)
            print([tag_id for tag_id in weekly_tags])
            print([predictions_for_each_tag[tag_id] for tag_id in weekly_tags])
            exit(1)

    for week in full_patterns[SPLIT_TIME:]: # debug check
        for kw in week:
            if kw not in chosen_ids:
                print("kw " + str(kw) + " not in chosen_ids")

    if (NUM_SEARCHES <= 50 and PRINT_EACH_EXPERIMENT):
        print([kw for kw in full_patterns[SPLIT_TIME:][0]])
        print()
        print(predictions_for_each_obs[0])

    flat_real = [kw for week_kws in full_patterns[SPLIT_TIME:] for kw in week_kws]
    flat_pred = [kw for week_kws in predictions_for_each_obs for kw in week_kws]
    accuracy = np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(flat_real, flat_pred)]))
    if PRINT_EACH_EXPERIMENT: print(accuracy)
    return accuracy


EXP_SETTINGS_FILE = "exp_settings.csv"

with open(EXP_SETTINGS_FILE, 'r') as settings_file:
    for row in settings_file:
        if row.startswith("#"):
            continue
        # apply settings
        settings_list = [setting.strip() for setting in row.split(",")]
        print("Running experiment " + str(settings_list))
        EXP_NAME = settings_list[0]
        LUCENE_EXP = settings_list[1] == 'True'
        NUM_SEARCHES = int(settings_list[2])
        NUM_UPDATES = int(settings_list[3])
        NUM_KEYWORDS = int(settings_list[4])
        S_DIFF = int(settings_list[5])
        UP_DIFF = int(settings_list[6])
        NUM_FILES = 30109 if 'x' in settings_list[7] else int(settings_list[7])
        DIST_SCALE = 0.0033 * NUM_FILES if 'x' in settings_list[8] else int(settings_list[8])
        UPDATE_FILE_ID_ONLY = settings_list[9] == 'True'
        AUX_EXP = settings_list[10] == 'True'
        ZIPF_EXP = settings_list[11] == 'True'
        RESULTS_FILENAME = EXP_NAME + ".txt"
        DATASET = dataset_enron if not LUCENE_EXP else dataset_lucene
        KW_DICT = kw_dict_enron if not LUCENE_EXP else kw_dict_lucene


        ''' Set up keywords and probabilities '''

        np.random.seed(123456789)
        seed_list = [int(np.random.random() * 10**8) for _ in range(NUM_EXPERIMENTS)]
        print("Seeds:", seed_list)
        kw_ids = [i for i in range(len(KW_DICT.keys()))]
        kw_list = [kw for kw in KW_DICT.keys()]
        assert NUM_KEYWORDS <= len(kw_list)

        probs = [entry['trend'][:NUM_WEEKS] for entry in KW_DICT.values()]

        search_accuracies = []
        search_times = []
        update_only_accuracies = []
        update_only_accuracies_mult = []
        update_only_times = []
        update_accuracies = []
        update_accuracies_mult = []
        update_times = []
        combined_accuracies = []

        for exp_number in range(len(seed_list)):
            print("Experiment ", exp_number + 1)
            np.random.seed(seed_list[exp_number])
            # permute keywords and choose n_kw
            permutation = np.random.permutation((len(kw_list)))
            chosen_kws = [kw_list[idx] for idx in permutation[:NUM_KEYWORDS]]
            chosen_ids = [idx for idx in permutation[:NUM_KEYWORDS]]
            chosen_probs = [probs[kw_id] for kw_id in chosen_ids]
            seen_access_patterns = []

            ''' Generate all search patterns '''

            inv_index = construct_inverted_index()

            full_search_patterns = []
            for i in range(NUM_WEEKS):
                if AUX_EXP and i <= NUM_WEEKS/2:
                    full_search_patterns.append(generate_week_search([row[i] for row in chosen_probs], NUM_SEARCHES - S_DIFF))
                else:
                    full_search_patterns.append(generate_week_search([row[i] for row in chosen_probs], NUM_SEARCHES))

            aux_search_patterns = full_search_patterns[:SPLIT_TIME]
            obs_search_patterns = []

            for i in range(SPLIT_TIME):
                obs_search_patterns.append([get_tag(kw_id, inv_index) for kw_id in full_search_patterns[SPLIT_TIME:][i]])

            ''' Generate all update patterns '''

            full_update_patterns = [] # list of np arrays
            full_update_only_patterns = []
            for i in range(NUM_WEEKS):
                if AUX_EXP and i <= NUM_WEEKS/2:
                    full_update_patterns.append(generate_week_update([row[i] for row in chosen_probs], NUM_UPDATES - UP_DIFF))
                else:
                    full_update_patterns.append(generate_week_update([row[i] for row in chosen_probs], NUM_UPDATES))

            # server can only observe updates for kws which have been searched
            # if a kw hasn't been searched by week i, exclude it from the observed trace for weeks >= i
            full_update_patterns = extract_unseen_updates_ind_full(full_search_patterns, full_update_patterns)

            # for the UF attack, we will use the same traces but remove the file ID info
            for week in full_update_patterns:
                full_update_only_patterns.append(np.array([tag for tag, file_id in week]))

            aux_update_patterns = full_update_patterns[:SPLIT_TIME]
            aux_update_only_patterns = full_update_only_patterns[:SPLIT_TIME]
            obs_update_patterns = []
            obs_update_only_patterns = []

            # it is valid to call the get_tag function for these updates because the corresponding keywords have already been searched.
            # this is within the leakage profiles of forward and level 2 backward-private schemes
            for i in range(SPLIT_TIME):
                obs_update_patterns.append([(get_tag(kw_id, inv_index), file_id) for kw_id, file_id in full_update_patterns[SPLIT_TIME:][i]])
                
            for week in obs_update_patterns:
                obs_update_only_patterns.append(np.array([tag for tag, file_id in week]))

            ''' Calculate auxiliary frequencies '''
            aux_search_frequencies = np.array(calculate_aux_probs_search(aux_search_patterns)) # (num_keywords x num_weeks/2)

            aux_update_frequencies = np.array(calculate_aux_probs_update(aux_update_patterns))

            aux_update_only_frequencies = np.array(calculate_aux_probs_update_only(aux_update_only_patterns))

            ''' Use the generated traces to create a cost matrix,
                run the hungarian algorithm to solve the matrix, analyze the results'''
            # format must be obs_patterns = [week1, week2, weekn], where week1 = [tag1, tag2, tagn]
            
            if PRINT_EACH_EXPERIMENT: print("Query Frequency Attack:")
            search_time_start = time.process_time()
            log_c_matrix_search = cost_matrix_search(aux_search_frequencies, obs_search_patterns)

            search_predictions_for_each_tag = run_hungarian_attack_given_matrix(log_c_matrix_search)
            search_time_end = time.process_time()
            search_acc = calculate_accuracy(obs_search_patterns, full_search_patterns, search_predictions_for_each_tag)
            search_accuracies.append(search_acc)
            search_times.append(search_time_end - search_time_start)
            if not UPDATE_FILE_ID_ONLY:
                if PRINT_EACH_EXPERIMENT: print("\nUpdate Frequency (UF) Attack:")
                update_only_time_start = time.process_time()
                log_c_matrix_update_only = cost_matrix_update_only(aux_update_only_frequencies, obs_update_only_patterns)
                update_only_predictions_for_each_tag = run_hungarian_attack_given_matrix(log_c_matrix_update_only)
                log_c_matrix_update_only_mult = log_c_matrix_search * log_c_matrix_update_only
                update_only_mult_predictions_for_each_tag = run_hungarian_attack_given_matrix(log_c_matrix_update_only_mult)
                update_only_time_end = time.process_time()
                update_only_acc = calculate_accuracy(obs_search_patterns, full_search_patterns, update_only_predictions_for_each_tag)
                # SP+UF attack
                update_only_mult_acc = calculate_accuracy(obs_search_patterns, full_search_patterns, update_only_mult_predictions_for_each_tag)
                update_only_accuracies.append(update_only_acc)
                update_only_accuracies_mult.append(update_only_mult_acc)
                update_only_times.append(update_only_time_end - update_only_time_start)
            if PRINT_EACH_EXPERIMENT: print("\nUpdate + File ID (UFID) Attack:")
            update_time_start = time.process_time()
            log_c_matrix_update_file_id = cost_matrix_update_file_id(aux_update_frequencies, obs_update_patterns)
            update_predictions_for_each_tag = run_hungarian_attack_given_matrix(log_c_matrix_update_file_id)
            log_c_matrix_update_mult = log_c_matrix_search * log_c_matrix_update_file_id
            update_mult_predictions_for_each_tag = run_hungarian_attack_given_matrix(log_c_matrix_update_mult)
            update_time_end = time.process_time()
            update_acc = calculate_accuracy(obs_search_patterns, full_search_patterns, update_predictions_for_each_tag)
            # SP+UFID
            update_mult_acc = calculate_accuracy(obs_search_patterns, full_search_patterns, update_mult_predictions_for_each_tag)
            update_accuracies.append(update_acc)
            update_accuracies_mult.append(update_mult_acc)
            update_times.append(update_time_end - update_time_start)

        if not os.path.exists(EXP_DIRECTORY):
            os.makedirs(EXP_DIRECTORY)

        with open(os.path.join(EXP_DIRECTORY, RESULTS_FILENAME), 'w') as f:
            if not UPDATE_FILE_ID_ONLY:
                f.write("Query Frequency Attack [Oya and Kerschbaum 2021]:\n")
                f.write("Accuracies: " + str(search_accuracies))
                f.write("\n")
                f.write(str(np.mean(search_accuracies)))
                f.write("\n")

                f.write("Runtimes: " + str(search_times))
                f.write("\n")
                f.write(str(np.mean(search_times)))
                f.write("\n")

                f.write("UF Attack:\n")
                f.write("Accuracies: " + str(update_only_accuracies))
                f.write("\n")
                f.write(str(np.mean(update_only_accuracies)))
                f.write("\n")

                f.write("SP+UF Attack:\n")
                f.write("Accuracies: " + str(update_only_accuracies_mult))
                f.write("\n")
                f.write(str(np.mean(update_only_accuracies_mult)))
                f.write("\n")

                f.write("Runtimes: " + str(update_only_times))
                f.write("\n")
                f.write(str(np.mean(update_only_times)))
                f.write("\n")

            f.write("UFID Attack:\n")
            f.write("Accuracies: " + str(update_accuracies))
            f.write("\n")
            f.write(str(np.mean(update_accuracies)))
            f.write("\n")

            f.write("SP+UFID Attack:\n")
            f.write("Accuracies: " + str(update_accuracies_mult))
            f.write("\n")
            f.write(str(np.mean(update_accuracies_mult)))
            f.write("\n")

            f.write("Runtimes: " + str(update_times))
            f.write("\n")
            f.write(str(np.mean(update_times)))
            f.write("\n")
            f.write("\n")
            f.write("Number of Experiments: " +  str(NUM_EXPERIMENTS) + "\n")
            f.write("Eta: " + str(NUM_SEARCHES) + "\n")
            f.write("Mu : " + str(NUM_UPDATES) + "\n")
            if AUX_EXP:
                f.write("During auxiliary period, eta = " + str(NUM_SEARCHES - S_DIFF) + "\n")
                f.write("During auxiliary period, mu = " + str(NUM_UPDATES - UP_DIFF) + "\n")
            f.write("Number of files: " + str(NUM_FILES) + "\n")
            if LUCENE_EXP:
                f.write("Using Lucene dataset\n")
            else:
                f.write("Using Enron dataset\n")
            if ZIPF_EXP:
                f.write("Using Zipfian distribution with parameter = " + str(ZIPF_PARAM) + "\n")
            else:
                f.write("Using Normal distribution with deviation of " + str(DIST_SCALE) + " files\n")
            f.write("Number of weeks: " + str(NUM_WEEKS) + "\n")
            f.write("Number of unique keywords client is querying/updating: " + str(NUM_KEYWORDS) + "\n")
            f.write("Auxiliary traces and observed traces are split at week " + str(SPLIT_TIME) + "\n")