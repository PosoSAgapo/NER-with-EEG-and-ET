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

#NOTE: We are now using only EEG frequencies for TRT to extract features (maybe, we should also use GD, GPT, or FFD)
def get_eeg_freqs(task:str, sbj:int, freq_domain:str, merge:str):
    """
        Args: Task (NR vs. TSR), Test subject number, EEG frequency domain (theta, alpha, beta, gamma), Binning strategy
        Return: NumPy matrix of respective EEG features on word level
    """
    files = get_matfiles(task)
    data = io.loadmat(files[sbj], squeeze_me=True, struct_as_record=False)['sentenceData']
    n_words = sum([len(sent.word) for sent in data if not isinstance(sent.word, float)])
    n_electrodes = 105
    word2eeg = np.zeros((n_words, n_electrodes))
    
    if freq_domain == 'theta':
        fields = ['TRT_t1', 'TRT_t2']
    elif freq_domain == 'alpha':
        fields = ['TRT_a1', 'TRT_a2']
    elif freq_domain == 'beta':
        fields = ['TRT_b1', 'TRT_b2']    
    elif freq_domain == 'gamma':
        fields = ['TRT_g1', 'TRT_g2']
        
    fixated = 0
    for sent in data:
        # if there is no data, skip sentence (most probably due to technical issues)
        if isinstance(sent.word, float):
            continue
        else:
            for word in sent.word:
                # if there was no fixation, skip word
                if isinstance(word.nFixations, np.ndarray):
                    continue
                else:
                    if merge == 'avg':
                        eeg_freq = np.mean(np.vstack([getattr(word, field) if hasattr(word, field) and len(getattr(word, field)) 
                                                      > 0 else 0 for field in fields]), axis = 0)
                    elif merge == 'max':
                        eeg_freq = np.amax(np.vstack([getattr(word, field) if hasattr(word, field) and len(getattr(word, field))
                                                      > 0 else 0 for field in fields]), axis = 0)
                    else:
                        raise ValueError('Binning strategy must be one of {max-pool, average}')
                    eeg_freq[np.isnan(eeg_freq)] = 0
                    word2eeg[fixated] += eeg_freq
                    fixated += 1
    word2eeg = word2eeg[:fixated, :]
    return word2eeg


def mean_freq_per_sbj(task:str, freq_domain:str, merge:str):
    sbjs_to_skip = [6, 11] if task == 'task2' else [3, 7, 11]
    X = []
    for i in range(12):
        if i not in sbjs_to_skip:
            X.append(get_eeg_freqs(task, i, freq_domain, merge))
    X_mean = np.zeros((X[0].shape[0], 105))
    if task == 'task2':
        D_0, D_1, D_2, D_3, D_4, D_5, D_7, D_8, D_9, D_10 = X 
        for i, (sbj_0, sbj_1, sbj_2, sbj_3, sbj_4, sbj_5, sbj_7, sbj_8, sbj_9, sbj_10) in enumerate(zip(D_0, D_1, D_2, D_3, D_4, D_5, D_7, D_8, D_9, D_10)):
            X_mean[i] += np.mean((sbj_0, sbj_1, sbj_2, sbj_3, sbj_4, sbj_5, sbj_7, sbj_8, sbj_9, sbj_10), axis=0)
    elif task == 'task3':
        D_0, D_1, D_2, D_4, D_5, D_6, D_8, D_9, D_10 = X
        for i, (sbj_0, sbj_1, sbj_2, sbj_4, sbj_5, sbj_6, sbj_8, sbj_9, sbj_10) in enumerate(zip(D_0, D_1, D_2, D_4, D_5, D_6, D_8, D_9, D_10)):
             X_mean[i] += np.mean((sbj_0, sbj_1, sbj_2, sbj_4, sbj_5, sbj_6, sbj_8, sbj_9, sbj_10), axis=0)
    return X_mean

def clf_fit(X_train, X_test, y_train, y_test, clf, rnd_state=42):
    if clf == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=False, random_state=rnd_state)
    elif clf == 'LogReg':
        model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=rnd_state,)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    #print(model.score(X_test, y_test))
    print(accuracy_score(y_test, y_hat))
    if clf == 'LogReg':
        return model.coef_
    else:
        return model.feature_importances_