import numpy as np
import scipy.io as io
import os
import scipy

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

def get_matfiles(task:str, subdir = '\\results_zuco\\'):
    """
        Args: Task number ("task1", "task2", "task3") plus sub-directory
        Return: 12 matlab files (one per subject) for given task
    """
    path = os.getcwd() + subdir + task
    files = [os.path.join(path,file) for file in os.listdir(path)[1:]]
    assert len(files) == 12, 'each task must contain 12 .mat files'
    return files

def stack_eeg_features(word, all_fields:list, eeg_locs_all_freqs:list, merge:str):
    """
        Args: list with all EEG attributes per Eye-Tracking feature,
              list with most important EEG features per freq domain per Eye-Tracking feature
        Return: binned, and stacked EEG features per word (single vector of dimensionality 4 x 4 x k)
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
                     ner_indices=None, split_sents=False, relation_indices=None, freq_domain=None, et_feat=None):
    """
        Args: Task (NR vs. TSR); test subject number; EEG frequency domain (theta, alpha, beta, or gamma); 
              Eye-Tracking feature for which we want to extract EEG features; binning strategy; 
              all features per frequency domain per Eye-Tracking features or most important features across all 
              frequency domains and Eye-Tracking features
        Return: NumPy matrix of respective EEG features on word level
    """
    files = get_matfiles(task)
    data = io.loadmat(files[sbj], squeeze_me=True, struct_as_record=False)['sentenceData']
    n_words = sum([len(sent.word) for sent in data if not isinstance(sent.word, float)])
    
    if split_sents:
        assert isinstance(relation_indices, list), 'If you want to split the data by relations, you must pass a list of sentence indices'
        
    if split_words: 
        assert isinstance(ner_indices, list), 'If you want to split the data by named entities, you must pass a list of word indices'
    
    if n_features == 'most_important':
        fields = [['FFD_a1', 'FFD_a2'], ['GD_a1', 'GD_a2'], ['GPT_a1', 'GPT_a2'], ['TRT_a1', 'TRT_a2'],
                  ['FFD_b1', 'FFD_b2'], ['GD_b1', 'GD_b2'], ['GPT_b1', 'GPT_b2'], ['TRT_b1', 'TRT_b2'],
                  ['FFD_g1', 'FFD_g2'], ['GD_g1', 'GD_g2'], ['GPT_g1', 'GPT_g2'], ['TRT_g1', 'TRT_g2'],
                  ['FFD_t1', 'FFD_t2'], ['GD_t1', 'GD_t2'], ['GPT_t1', 'GPT_t2'], ['TRT_t1', 'TRT_t2']]
        
        path = os.getcwd() + '\\eeg_feature_extraction\\' + '\\important_eeg_features\\'
        files = [os.path.join(path, file) for file in os.listdir(path)]
        eeg_locs_all_freqs = [np.loadtxt(file, dtype=int) for file in files]
        n_et_feats = 4
        n_eeg_freqs = 8 
        n_eeg_freqs_binned = n_eeg_freqs // 2
        k = 10
        word2eeg = np.zeros((n_words, int(n_et_feats * n_eeg_freqs_binned * k)))
        
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
        if duplicate_sents != None:
            n_words = sum([len(sent.word) for sent in data if not isinstance(sent.word, float) and sent.content in duplicate_sents])
        word2eeg = np.zeros((n_words, n_electrodes))

    else:
        raise ValueError('Number of features must be one of [all, most_important]')
        
    fixated = 0
    if split_words:
        j = 0
    for i, sent in enumerate(data):
        # don't use hold out test set for EEG feature extraction (with Random Forest)
        if held_out_indices != None and isinstance(held_out_indices, list):
            if i in held_out_indices:
                continue
        # if you want to extract most important EEG features, you have to compare sentences that occurred in both tasks
        if duplicate_sents != None:
            if sent.content not in duplicate_sents:
                continue
        # if there is no data, skip sentence (most probably due to technical issues)
        if isinstance(sent.word, float):
            if split_words:
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
                # if there was no fixation, skip word (we only care about words where a fixation landed)
                if isinstance(word.nFixations, np.ndarray):
                    if split_words:
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
                    word2eeg[fixated] += eeg_freq
                    fixated += 1
                    if split_words:
                        j += 1
    word2eeg = word2eeg[:fixated, :] if duplicate_sents == None else word2eeg
    return word2eeg

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

def reshape_into_tensor(eeg_data_all_sbjs, sent_lens_sbj):
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


### EEG feature extraction on word level (with Random Forest) per frequency domain per eye-tracking feature across all subjects ###

# less data (and thus computationally more efficient) but maybe not as informative as stacking all sbjs (vertically)
def mean_freq_per_sbj(task:str, freq_domain:str, merge:str, et_feature:str, held_out_indices=None, duplicate_sents=None):
    sbjs_to_skip = [6, 11] if task == 'task2' else [3, 7, 11]    
    eeg_feats_all_sbjs = [get_eeg_features(task=task, sbj=i, n_features='all', merge=merge, 
                                           held_out_indices=held_out_indices, duplicate_sents=duplicate_sents, 
                                           freq_domain=freq_domain, et_feat=et_feature) 
                                           for i in range(12) if i not in sbjs_to_skip]
    min_n_words_fixated = np.min([eeg_feats_sbj.shape[0] for eeg_feats_sbj in eeg_feats_all_sbjs])
    X = list(map(lambda eeg_feats: eeg_feats[:min_n_words_fixated, :], eeg_feats_all_sbjs))
    return np.mean(X, axis=0)

# huge data set (and thus computationally not so efficient) but most probably more informative than averaging overs subjects
def stacked_freq_per_sbj(task:str, freq_domain:str, merge:str, et_feature:str, held_out_indices=None, all_sbjs=True,
                         duplicate_sents=None):
    if all_sbjs:
        stacked_eeg_feats_all_sbjs = np.vstack([get_eeg_features(task=task, sbj=i, n_features='all', merge=merge, 
                                                held_out_indices=held_out_indices, duplicate_sents=duplicate_sents,
                                                freq_domain=freq_domain, et_feat=et_feature) 
                                                for i in range(12)])
    else:
        sbjs_to_skip = [6, 11] if task == 'task2' else [3, 7, 11]
        stacked_eeg_feats_all_sbjs = np.vstack([get_eeg_features(task=task, sbj=i, n_features='all', merge=merge, 
                                                held_out_indices=held_out_indices, duplicate_sents=duplicate_sents,
                                                freq_domain=freq_domain, et_feat=et_feature) 
                                                for i in range(12) if i not in sbjs_to_skip])
    return stacked_eeg_feats_all_sbjs

def clf_fit(X_train, X_test, y_train, y_test, clf, rnd_state=42):
    if clf == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=False, random_state=rnd_state)
    elif clf == 'LogReg':
        model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=rnd_state)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    #print(model.score(X_test, y_test))
    print(accuracy_score(y_test, y_hat))
    if clf == 'LogReg': return model.coef_
    else: return model.feature_importances_