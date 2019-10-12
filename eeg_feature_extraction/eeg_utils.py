import numpy as np
import pandas as pd
import scipy.io as io
import os
import scipy
import torch

from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from torch import nn

### general helper functions ###

def get_matfiles(task:str, subdir = '\\results_zuco\\'):
    """
        Args: 
            Task number ("task1", "task2", "task3") plus sub-directory
        Return: 
            12 matlab files (one per subject) for given task
    """
    path = os.getcwd() + subdir + task
    files = [os.path.join(path,file) for file in os.listdir(path)[1:]]
    assert len(files) == 12, 'each task must contain 12 .mat files'
    return files

def get_sent_lens_per_task(task:str, held_out_indices:list):
    files = get_matfiles(task)
    data = io.loadmat(files[0], squeeze_me=True, struct_as_record=False)['sentenceData']
    sent_lens = [len(sent.word) for i, sent in enumerate(data) if i not in held_out_indices]
    return sent_lens

def get_eeg_locs(features:str):
    path = os.getcwd() + '\\eeg_feature_extraction\\' + features
    files = [os.path.join(path, file) for file in os.listdir(path)]
    eeg_locs_all_freqs = [np.loadtxt(file, dtype=int) for file in files if not file.endswith('.ipynb_checkpoints')]
    return eeg_locs_all_freqs

def get_held_out_sents(task:str):
    path = os.getcwd() + '\\eeg_feature_extraction\\' + '\\held_out_test_set\\'
    files = [os.path.join(path, file) for file in os.listdir(path) if not file.endswith('.ipynb_checkpoints')]
    held_out_sents = [np.loadtxt(file, dtype=int).tolist() for file in files]
    return list(set(held_out_sents[0])) if task == 'task2' else list(set(held_out_sents[1]))

def load_embeddings(classification:str, k = None):
    subdir = '\\embeddings_binary\\' if classification == 'binary' else '\\embeddings_multi\\'
    path = os.getcwd() + '\\embeddings\\' + subdir
    path = path + '\\' + str(k) + '\\' if classification == 'binary' else path
    files = [os.path.join(path, file) for file in os.listdir(path)]
    all_embeddings = [np.loadtxt(file) for file in files if not file.endswith('.ipynb_checkpoints')]
    return all_embeddings

def get_rel_labels():
    # load labeled sentences
    relation_labels_task2 = pd.read_csv('./task_materials/relations_labels_task2.csv', sep = ',')
    relation_labels_task3 = pd.read_csv('./task_materials/relations_labels_task3.csv', sep = ';')
    # get indices of sentences with relation labels
    indices_relations_task2 = [idx for idx, relation in enumerate(relation_labels_task2.relation_types.values)
                               if relation != 'NO-RELATION'] 
    indices_no_relations_task2 = [idx for idx, relation in enumerate(relation_labels_task2.relation_types.values) 
                                  if relation == 'NO-RELATION'] 
    assert len(indices_relations_task2) + len(indices_no_relations_task2) == 300
    indices_relations_task3 = [idx for idx, relation in enumerate(relation_labels_task3['relation-type'].values)
                               if relation != 'CONTROL'] 
    indices_no_relations_task3 = [idx for idx, relation in enumerate(relation_labels_task3['relation-type'].values)
                                  if relation == 'CONTROL'] 
    assert len(indices_relations_task3) + len(indices_no_relations_task3) == 407
    return indices_relations_task2, indices_no_relations_task2, indices_relations_task3, indices_no_relations_task3

### Main EEG feature extraction helper function ###

def stack_eeg_features(word, all_fields:list, eeg_locs_all_freqs:list, merge:str):
    """
        Args:
              all EEG attributes per Eye-Tracking feature (list),
              most important EEG features per freq domain per Eye-Tracking feature (list)
        Return: 
              binned, and stacked EEG features per word (single vector of dimensionality 4 x 4 x k) (np.ndarray)
    """
    eeg_freqs = []
    for fields, eeg_locs in zip(all_fields, eeg_locs_all_freqs):
        #first, bin frequency domains
        if merge == 'avg':
            eeg_freq = np.mean(np.vstack([getattr(word, field) if hasattr(word, field) and
                                          len(getattr(word, field)) > 0 else 0 
                                          for field in fields]), axis = 0)
        elif merge == 'max':
            eeg_freq = np.amax(np.vstack([getattr(word, field) if hasattr(word, field) and
                                          len(getattr(word, field)) > 0 else 0 
                                          for field in fields]), axis = 0)
        #second, extract the most important eeg features / locations per freq domain
        eeg_freqs.append(eeg_freq[eeg_locs])
    #third, stack all eeg feats horizontally
    return np.hstack([eeg_freq for eeg_freq in eeg_freqs])

def get_eeg_features(task:str, sbj:int, n_features:str, merge:str, duplicate_sents=None, held_out_indices=None, split_words=False,
                     ner_indices=None, split_sents=False, relation_indices=None, freq_domain=None, et_feat=None, dim_reduction=False):
    """
        Args: 
            Task (NR vs. TSR) (str), 
            test subject number (int), 
            all features per frequency domain per Eye-Tracking feature or most important EEG features extracted through 
            Random Forest (str),
            merging / binning strategy for EEG freq domains(str),
            duplicate sentences (list),
            indices for held out test sentences (list),
            whether we want to split words into words with NER labels (bool),
            word indices for all words with NER labels (list),
            whether we want to split sentences into sentences with relations (bool),
            sentence indices for all sentences with relation labels (list),
            EEG frequency domain (theta, alpha, beta, or gamma) (str),
            Eye-Tracking feature for which we want to extract EEG feats (str),
            whether dimensionality reduction is performed (bool)
        Return: 
             NumPy matrix of respective EEG features on word level (np.ndarray)
    """
    files = get_matfiles(task)
    data = io.loadmat(files[sbj], squeeze_me=True, struct_as_record=False)['sentenceData']
    
    if duplicate_sents != None:
        n_words = sum([len(sent.word) for sent in data if not isinstance(sent.word, float) and sent.content in duplicate_sents])
    elif dim_reduction:
        if held_out_indices != None and isinstance(held_out_indices, list):
            n_words = sum([len(sent.content.split()) for i, sent in enumerate(data) if i not in held_out_indices])
        else:
            n_words = sum([len(sent.content.split()) for sent in data])
    else:
        n_words = sum([len(sent.word) for sent in data if not isinstance(sent.word, float)])
    
    if split_sents:
        assert isinstance(relation_indices, list), 'If you want to split the data by relations, you must pass a list of sentence indices'
        
    if split_words: 
        assert isinstance(ner_indices, list), 'If you want to split the data by named entities, you must pass a list of word indices'
    
    if n_features == 'most_important':
        fields = [['TRT_a1', 'TRT_a2'],
                  ['TRT_b1', 'TRT_b2'],
                  ['TRT_g1', 'TRT_g2'],
                  ['TRT_t1', 'TRT_t2']]
        
        path = os.getcwd() + '\\eeg_feature_extraction\\' + '\\important_eeg_features\\'
        files = [os.path.join(path, file) for file in os.listdir(path) if not file.endswith('.ipynb_checkpoints')]
        eeg_locs_all_freqs = [np.loadtxt(file, dtype=int) for file in files]
        n_eeg_freqs = 8 
        n_eeg_freqs_binned = n_eeg_freqs // 2
        k = 10
        word2eeg = np.zeros((n_words, int(n_eeg_freqs_binned * k)))
        
    elif n_features == 'all':
        if freq_domain == 'theta':
            eeg_feat_1, eeg_feat_2 = et_feat + '_t1', et_feat + '_t2'
        elif freq_domain == 'alpha':
            eeg_feat_1, eeg_feat_2 = et_feat + '_a1', et_feat + '_a2'   
        elif freq_domain == 'beta':
            eeg_feat_1, eeg_feat_2 = et_feat + '_b1', et_feat + '_b2' 
        elif freq_domain == 'gamma':
            eeg_feat_1, eeg_feat_2 = et_feat + '_g1', et_feat + '_g2'
        fields = [eeg_feat_1, eeg_feat_2]
        n_electrodes = 105
        word2eeg = np.zeros((n_words, n_electrodes))

    else:
        raise ValueError('Number of features must be one of [all, most_important]')
        
    fixated = 0
    if split_words or dim_reduction:
        j = 0
    for i, sent in enumerate(data):
        # don't use hold out test set for any feature extraction or data analysis step prior to final modelling
        if held_out_indices != None and isinstance(held_out_indices, list):
            if i in held_out_indices:
                continue
        # for data visualisation, compare EEG activity distribution for sentences that occurred in both tasks
        if duplicate_sents != None:
            if sent.content not in duplicate_sents:
                continue
        # if there is no data, skip sentence (most probably due to technical issues)
        if isinstance(sent.word, float):
            #print("No data to analyse for sentence {}".format(i))
            #print("Inspect whether you are using all subjects for current data transformation!")
            if split_words or dim_reduction:
                sent_len = len(sent.content.split())
                j += sent_len
            continue
        else:
            if not split_sents:
                pass
            elif split_sents and i in relation_indices:
                pass
            elif split_sents and i not in relation_indices:
                continue
            for word in sent.word:
                if not split_words:
                    pass
                elif split_words and j in ner_indices:
                    pass
                elif split_words and j not in ner_indices:
                    continue
                # if there was no fixation, skip word (for statistical analyses and visualisation
                # we only care about words where a fixation landed)
                # if we reduce dimensionalities, however, we have to compute vectors for every word in the data set
                if isinstance(word.nFixations, np.ndarray):
                    if split_words or dim_reduction:
                        j += 1
                    continue
                else:
                    if n_features == 'most_important':
                        # bin and stack most important eeg features horiztonally (4 x 4 x k dimensional vector)
                        eeg_freq = stack_eeg_features(word, fields, eeg_locs_all_freqs, merge)
                    elif n_features == 'all':
                        if merge == 'avg':   
                            eeg_freq = np.mean(np.vstack([getattr(word, field) if hasattr(word, field) and
                                                          len(getattr(word, field)) > 0 else 0 
                                                          for field in fields]), axis = 0)
                        elif merge == 'max':
                            eeg_freq = np.amax(np.vstack([getattr(word, field) if hasattr(word, field) and
                                                          len(getattr(word, field)) > 0 else 0 
                                                          for field in fields]), axis = 0)
                        else:
                            raise ValueError('Binning strategy must be one of [max-pool, average]')
                    eeg_freq[np.isnan(eeg_freq)] = 0
                    if dim_reduction:
                        word2eeg[j] += eeg_freq
                        j += 1
                    else:
                        word2eeg[fixated] += eeg_freq
                        fixated += 1
                    if split_words:
                        j += 1
    word2eeg = word2eeg if (duplicate_sents != None and n_features == 'all') or (dim_reduction == True) else word2eeg[:fixated, :]
    return word2eeg


### Helper functions to compute embeddings and reshape EEG data into tensor ###

def reshape_into_tensor(eeg_data_per_task:np.ndarray, task:str):
    """
    Transforms EEG data matrix into 3D sequence tensor
        Args:
            word embeddings per task (numpy matrix of size N (number of words) x D (embedding dim)),
            task (str)
        Return:
              reshaped word embeddings (tensors of size N (number of sents) x T (sequence length per task) x D (embedding dim))
    """
    held_out_indices_task2 = get_held_out_sents('task2')
    held_out_indices_task3 = get_held_out_sents('task3')
    
    sent_lens_task2 = get_sent_lens_per_task('task2', held_out_indices_task2)
    sent_lens_task3 = get_sent_lens_per_task('task3', held_out_indices_task3)
    sent_lens_current_task = sent_lens_task2 if task == 'task2' else sent_lens_task3
    
    assert eeg_data_per_task.shape[0] == sum(sent_lens_current_task), 'n_rows in matrix must be equal to the n_words_total in task'
    
    sent_mean_len_task2 = round(np.mean(sent_lens_task2))
    sent_mean_len_task3 = round(np.mean(sent_lens_task3))
    mean_len = np.min((sent_mean_len_task2, sent_mean_len_task3))
    mean_len = torch.tensor(mean_len).long().item()
    
    embed_dim = eeg_data_per_task.shape[1]
    max_len, n_sents = np.max(sent_lens_current_task), len(sent_lens_current_task)
    eeg_seq_tensor = torch.zeros(n_sents, max_len, embed_dim, dtype = torch.double)
    # cumulative sentence length
    cum_sent_len = 0
    for i, sent_len in enumerate(sent_lens_current_task):
        eeg_seq_tensor[i, :sent_len, :] += torch.from_numpy(eeg_data_per_task[cum_sent_len:cum_sent_len+sent_len, :])
        cum_sent_len += sent_len
    # truncate sequences at mean sentence length per task
    eeg_seq_tensor = eeg_seq_tensor[:, :mean_len, :]
    return eeg_seq_tensor

def create_multiclass_word_labels(indices_rel_task:list, indices_no_rel_task:list, task:str):
    held_out_sents = get_held_out_sents(task)
    files_task = get_matfiles(task)
    # use sbj_1 to create labels on word level since we have data for all sentences for this participant
    data_task = io.loadmat(files_task[0], squeeze_me=True, struct_as_record=False)['sentenceData']
    n_words = sum([len(sent.word) for i, sent in enumerate(data_task) if i not in held_out_sents])
    labels = np.zeros((n_words, 1)) if task == 'task2' else np.ones((n_words, 1)) * 2
    j = 0
    for i, sent in enumerate(data_task):
        if i not in held_out_sents:
            for word in sent.word:
                if i in indices_rel_task:
                    labels[j] += 1
                    j += 1
    return labels

def compute_embeddings(feat_extraction_methods:list, freq_domains:list, merge = 'avg', n_features = 'all', k = 15,
                       classification = 'binary'):
    """
        Args:
            feature extraction methods (i.e., Random Forest or NCA) (list),
            EEG frequency domains (alpha, beta, theta) (list),
            merging / binning strategy of EEG frequencies (str),
            all features per frequency domain per Eye-Tracking feature or most important EEG features extracted through 
            Random Forest (not relevant for dim reduction through NCA) (str),
            k most important features to extract (int),
            embeddings will be computed for a binary or multi-class classification objective (str)
       Return:
             cognitive word embeddings in EEG space for both feature extraction methods and both tasks respectively (dict)
    """
    eeg_locs_all_freqs = get_eeg_locs('\\eeg_features_for_embeddings\\' + '\\' + str(k) + '\\')
    held_out_sents_task2, held_out_sents_task3 = get_held_out_sents('task2'), get_held_out_sents('task3')
    
    if classification == 'multiclass':
        indices_relations_task2, indices_no_relations_task2, indices_relations_task3, indices_no_relations_task3 = get_rel_labels()
        # get indices for (dev) sentences with and without relations respectively
        indices_rel_task2 = [idx for idx in indices_relations_task2 if idx not in held_out_sents_task2]
        indices_no_rel_task2 = [idx for idx in indices_no_relations_task2 if idx not in held_out_sents_task2]
        indices_rel_task3 = [idx for idx in indices_relations_task3 if idx not in held_out_sents_task3]
        indices_no_rel_task3 = [idx for idx in indices_no_relations_task3 if idx not in held_out_sents_task3]
        
    et_feature = 'TRT'
    rnd_state = 42
    embeddings = defaultdict(dict)
    for feat_extraction_method in feat_extraction_methods:
        if classification == 'multiclass' and feat_extraction_method == 'RandomForest':
            continue
        embeddings_task2, embeddings_task3 = [], []
        for idx, freq_domain in enumerate(freq_domains):
            
            X_NR = eeg_freqs_across_sbj('task2', freq_domain, merge, et_feature, n_features,
                                      held_out_indices=held_out_sents_task2, all_sbjs=False,
                                      dim_reduction=True)
            X_AR = eeg_freqs_across_sbj('task3', freq_domain, merge, et_feature, n_features,
                                      held_out_indices=held_out_sents_task3, all_sbjs=False,
                                      dim_reduction=True)
            
            if classification == 'multiclass':
                Y_NR = create_multiclass_word_labels(indices_rel_task2, indices_no_rel_task2, 'task2')
                Y_AR = create_multiclass_word_labels(indices_rel_task3, indices_no_rel_task3, 'task3')
                
            elif classification == 'binary':
                Y_NR, Y_AR = np.zeros((X_NR.shape[0], 1)), np.ones((X_AR.shape[0], 1))
            
            if feat_extraction_method == 'RandomForest':
                embeddings_task2.append(X_NR[:, eeg_locs_all_freqs[idx]])
                embeddings_task3.append(X_AR[:, eeg_locs_all_freqs[idx]])
                
            elif feat_extraction_method == 'NCA':
                n_words_task2 = X_NR.shape[0]
                X, y = np.vstack((X_NR, X_AR)), np.vstack((Y_NR, Y_AR))
                X_transformed = dimensionality_reduction(X, y.ravel(), feat_extraction_method, k = k)
                
                # NOTE: to increase gap between classes, NCA transforms data into vector space of large continuous numbers
                # hence, we have to center and normalize transformed data prior to embeddings computation to reduce range
                for i in range(X_transformed.shape[1]):
                    mean, std = X_transformed[:, i].mean(), X_transformed[:, i].std()
                    X_transformed[:, i] -= mean
                    X_transformed[:, i] /= std
                    
                embeddings_task2.append(X_transformed[:n_words_task2, :])
                embeddings_task3.append(X_transformed[n_words_task2:, :])
                
            print("{} embeddings computed through {}".format(freq_domain.capitalize(), feat_extraction_method))
                
        embeddings[feat_extraction_method]['NR'] = np.hstack(embeddings_task2)
        embeddings[feat_extraction_method]['TSR'] = np.hstack(embeddings_task3)
                
    return embeddings

### Helper functions to visualise EEG data ###

def extract_electrodes_and_indices(eeg_electrodes:np.ndarray, eeg_locs:np.ndarray, electrodes_freq:list, k=0):
    all_electrodes_per_freq = eeg_electrodes[eeg_locs]
    cortex_indices_per_freq = np.array([eeg_electrodes[eeg_locs].tolist().index(electrod) for electrod in electrodes_freq])
    all_cortex_electrodes_per_freq = np.array([all_electrodes_per_freq[cortex_indices_per_freq]]).ravel()
    all_cortex_indices_per_freq = np.array([idx + k for idx in cortex_indices_per_freq]).ravel()
    return all_cortex_electrodes_per_freq, all_cortex_indices_per_freq

def truncating(eeg_mat):
    mean_len = np.mean([len(sent) for sent in eeg_mat], dtype=int)
    eeg_mat_trunc = np.zeros((eeg_mat.shape[0], mean_len), dtype=float)
    for i, sent in enumerate(eeg_mat):
        if len(sent) <= mean_len:
            eeg_mat_trunc[i, :len(sent)] += sent
        else:
            eeg_mat_trunc[i, :len(sent)] += sent[:mean_len]
    return eeg_mat_trunc

def zero_padding(eeg_mat):
    max_len = max([len(sent) for sent in eeg_mat])
    eeg_mat_padded = np.zeros((eeg_mat.shape[0], max_len), dtype=float)
    for i, sent in enumerate(eeg_mat):
        eeg_mat_padded[i, :len(sent)] += sent
    return eeg_mat_padded

def map_electrode_onto_tensor(eeg_tensor, electrode_idx):
    return np.array(list(map(lambda sent:sent[:, electrode_idx].ravel(), eeg_tensor)))

def reshape_into_tensor_vis(eeg_data_all_sbjs, sent_lens_sbj):
    # get eeg data per sbj (all sentences)
    eeg_data_sbj = eeg_data_all_sbjs[:sum(sent_lens_sbj), :]
    # split eeg data into 3D tensor (N (sentences) x D (words) x K (EEG activity in electrode X for X frequency domain))
    eeg_data_sbj_tensor = []
    # cumulative sent len
    cum_sent_len = 0
    for sent_len in sent_lens_sbj:
        eeg_data_sbj_tensor.append(eeg_data_sbj[cum_sent_len:cum_sent_len+sent_len])
        cum_sent_len += sent_len
    return np.array(eeg_data_sbj_tensor)


### Helper functions for feature extraction on word level (with Random Forest or NCA) per EEG frequency domain and per Eye-Tracking feature across all subjects ###

def eeg_freqs_across_sbj(task:str, freq_domain:str, merge:str, et_feature:str, n_features:str, held_out_indices=None, all_sbjs=True,
                      duplicate_sents=None, dim_reduction=False, split_sents=False, rel_indices=None, strategy='avg'):
    all_sbjs = False if dim_reduction else True
    #NOTE, to compute embeddings: n_features has to be 'most_important' for Random Forest 
    #                             but 'all' for NCA as Random Forest is not a dimensionality reduction method (i.e., feat extraction)
    #                             (hence, we have to index through most important channels to get most informative dims)
    if all_sbjs:
        eeg_feats_all_sbjs = [get_eeg_features(task=task, sbj=i, n_features=n_features, merge=merge, 
                                               held_out_indices=held_out_indices, duplicate_sents=duplicate_sents,
                                               split_sents=split_sents, relation_indices=rel_indices,
                                               freq_domain=freq_domain, et_feat=et_feature, dim_reduction=dim_reduction) 
                                               for i in range(12)]
    else:
        #NOTE: for dim reduction, we don't want all sbjs (would highly bias the data due to many missing values)
        sbjs_to_skip = [6, 11] if task == 'task2' else [3, 7, 11]
        eeg_feats_all_sbjs = [get_eeg_features(task=task, sbj=i, n_features=n_features, merge=merge, 
                                               held_out_indices=held_out_indices, duplicate_sents=duplicate_sents,
                                               split_sents=split_sents, relation_indices=rel_indices,
                                               freq_domain=freq_domain, et_feat=et_feature, dim_reduction=dim_reduction) 
                                               for i in range(12) if i not in sbjs_to_skip]
    if dim_reduction:
        return np.mean(eeg_feats_all_sbjs, axis = 0) if strategy == 'avg' else np.amax(eeg_feats_all_sbjs, axis=0)
    else:
        return np.vstack(eeg_feats_all_sbjs)
    
def remove_non_fixated_words(mean_eeg_feats_per_word):
    eeg_feats_fixated_words = []
    for eeg_feats_per_word in mean_eeg_feats_per_word:
        try:
            if int(np.unique(eeg_feats_per_word).item()) == 0:
                continue
        except ValueError:
            eeg_feats_fixated_words.append(eeg_feats_per_word)
    return np.array(eeg_feats_fixated_words)

def feature_extraction(feat_extraction_methods:list, freq_domains:list, 
                       merge = 'avg', n_features = 'all', k = 15, evaluate=False):
    et_feature = 'TRT'
    rnd_state = 42
    accs = defaultdict(dict)
    held_out_sents_task2, held_out_sents_task3 = get_held_out_sents('task2'), get_held_out_sents('task3')
    transformed_eeg_features = dict()
    for feat_extraction_method in feat_extraction_methods:
        if not evaluate: transformed_eeg_features_per_method = []
        for freq_domain in freq_domains:
            mean_eeg_feats_per_word_task2 = eeg_freqs_across_sbj('task2', freq_domain, merge, et_feature, n_features,
                                                              held_out_indices=held_out_sents_task2, all_sbjs=False,
                                                              dim_reduction=True)
            mean_eeg_feats_per_word_task3 = eeg_freqs_across_sbj('task3', freq_domain, merge, et_feature, n_features,
                                                                  held_out_indices=held_out_sents_task3, all_sbjs=False,
                                                                  dim_reduction=True)
            X_NR = remove_non_fixated_words(mean_eeg_feats_per_word_task2)
            X_AR = remove_non_fixated_words(mean_eeg_feats_per_word_task3)
            Y_NR = np.zeros((X_NR.shape[0], 1))
            Y_AR = np.ones((X_AR.shape[0], 1))
            X, y = np.vstack((X_NR, X_AR)), np.vstack((Y_NR, Y_AR))
            
            if evaluate:
                X, y = shuffle(X, y, random_state=rnd_state)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=rnd_state)
                y_train, y_test = y_train.ravel(), y_test.ravel()
                accs[feat_extraction_method][freq_domain] = dimensionality_reduction(X_train, y_train, 
                                                                                     feat_extraction_method, 
                                                                                     X_val=X_test, y_val=y_test,
                                                                                     evaluate=evaluate)
            else:
                transformed_eeg_features_per_method.append(dimensionality_reduction(X, y.ravel(), feat_extraction_method))
        if not evaluate: transformed_eeg_features[feat_extraction_method] = (np.hstack(transformed_eeg_features_per_method), y)
    if evaluate: return accs
    else:
        assert transformed_eeg_features['RandomForest'][0].shape[1] == transformed_eeg_features['NCA'][0].shape[1] == int(k*len(freq_domains))
        return transformed_eeg_features

def dimensionality_reduction(X_train, y_train, feat_extration_method:str, X_val=None, y_val=None,
                             k=15, rnd_state=42, evaluate=False):
    # instantiate kNN to evaluate feature extraction techniques
    knn = KNeighborsClassifier(n_neighbors=3)
    if feat_extration_method == 'RandomForest':
        # instantiate Random Forest with 100 estimators
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=False, random_state=rnd_state)
        clf.fit(X_train, y_train)
        most_important_feats = clf.feature_importances_
        # extract k most important EEG electrodes according to Random Forest splits
        k_most_important_feats = np.argsort(most_important_feats)[::-1][:k]
        if evaluate:
            #clf.fit(X_train[:, k_most_important_feats], y_train)
            #acc = clf.score(X_val[:, k_most_important_feats], y_val)
            knn.fit(X_train[:, k_most_important_feats], y_train)
            acc = knn.score(X_val[:, k_most_important_feats], y_val)
            return acc
        else:
            return X_train[:, k_most_important_feats]        
    elif feat_extration_method == 'NCA':
        # instantiate NCA with k components
        nca = NeighborhoodComponentsAnalysis(n_components=k, random_state=rnd_state)
        nca.fit(X_train, y_train)
        if evaluate:
            X_train_transformed, X_val_transformed = nca.transform(X_train), nca.transform(X_val)
            knn.fit(X_train_transformed, y_train)
            acc = knn.score(X_val_transformed, y_val)
            return acc
        else:
            return nca.transform(X_train)

def grid_search(X_train, X_test, y_train, y_test, rnd_state=42):
    param_grid = {
    'bootstrap': [False, True],
    'max_features': ['auto', 'log2'],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [4, 5, 6],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [50, 100]
    }
    # create Random Forest instance
    clf = RandomForestClassifier(random_state=rnd_state)
    # instantiate the grid search model
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                               cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    grid_search.best_params_
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, test_features, test_labels)
    print(grid_accuracy)
    return grid_search.best_params_