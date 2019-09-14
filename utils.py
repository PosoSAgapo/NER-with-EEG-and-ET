from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.io as io
import gzip
import math
import os
import re
import scipy

def get_bncfreq(subdir = '\\BNC\\', file = 'all.al.gz', n_fields = 4):
    """
        Args: British National Corpus word frequency list;
              four fields per line (0: freq, 1: word, 2: pos, 4 :n_files the word occurs in)
        Return: FreqDict for BNC
    """
    bnc_freq = defaultdict(float)
    path = os.getcwd() + subdir
    # unzip automatically and make sure you don't read in binary mode (-> 'rt' instead of 'rb')
    with gzip.open(os.path.join(path, file), 'rt') as file:
        bnc_freqlist = list(map(lambda el: el.split(), file.readlines()[1:]))
        for line in bnc_freqlist:
            if len(line) == n_fields:
                bnc_freq[line[1]] += float(line[0])
        return bnc_freq

def get_matfiles(task:str, subdir = '\\results_zuco\\'):
    """
        Args: Task number ("task1", "task2", "task3") plus sub-directory
        Return: 12 matlab files (one per subject) for given task
    """
    path = os.getcwd() + subdir + task
    files = [os.path.join(path,file) for file in os.listdir(path)[1:]]
    assert len(files) == 12, 'each task must contain 12 .mat files'
    return files

class DataFrameLoader:
    """
        DataFrame loader object for ZuCo
    """
    
    def __init__(self, task:str, subject:int, level:str):
        tasks = ['task1', 'task2', 'task3']
        if task not in tasks:
            raise Exception('Task can only be one of "task1", "task2", or "task3"')
        else:
            self.task = tasks
        subjects = list(range(12))
        if subject not in subjects:
            raise Exception('Access subject data with an integer value between 0 - 11')
        else:
            self.subject = subject
        levels = ['sentence', 'word']
        if level not in levels:
            raise Exception('Data can only be processed on sentence or word level')
        
    def __call__(self):
        return mk_dataframe(self)
    
    def mk_dataframe(self):
        """
            Args: Task number ("task1", "task2", "task3") , test subject (0-11)
            Return: DataFrame with features (i.e., attributes) on word level
        """
        bnc_freq = get_bncfreq()
        files = get_matfiles(self.task)
        data = io.loadmat(files[self.subject], squeeze_me=True, struct_as_record=False)['sentenceData']
        
        if self.level == 'sentence':
            fields = ['SentLen', 'nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 
                      'GPT', 'BNCFreq']
            features = np.zeros((len(data), len(fields)))

        elif self.level == 'word':
            n_words = sum([len(sent.word) for sent in data])    
            fields = list(set(field for sent in data for word in sent.word for field in word._fieldnames\
                         if not field.startswith('raw')))
            fields = sorted(fields, reverse=True)
            fields.insert(0, 'word_id')
            fields.insert(0, 'sent_id')
            df = pd.DataFrame(index=range(n_words), columns=[fields])
            k = 0

        for i, sent in enumerate(data):
            for j, word in enumerate(sent.word):
                if level == 'sentence':
                    features[i,1:-1] += [getattr(word, field) if hasattr(word, field)\
                                    and not isinstance(getattr(word, field), np.ndarray) else\
                                    0 for field in fields[1:-1]]
                    token = re.sub('[^\w\s]', '', word.content)
                    #TODO: figure out whether divsion by 100 leads to log = -inf 
                    features[i,-1] += np.log(bnc_freq[token]/100) if bnc_freq[token]/100 != 0 else 0 
                elif level == 'word:
                    df.iloc[k, 0] = str(i) + '_NR' if task=='task1' or task=='task2' else str(i) + '_TSR'
                    df.iloc[k, 1] = j
                    df.iloc[k, 2:] = [getattr(word, field) if hasattr(word, field) else np.nan\
                                      for field in fields[2:]]
                    k += 1

            if level == 'sentence':
                features[i, 0] = len(sent.word)
                features[i, 1:] /= len(sent.word)
                
        if level == 'sentence':
            # remove all rows with -inf or inf values (if data matrix contains any)
            features = self.check_inf(features)
            # normalize data featurewise
            features = np.array([feat / max(feat) for i, feat in enumerate(features.T)])
            df = pd.DataFrame(data=features.T, index=range(features.shape[1]), columns=[fields])
            
        return df
    
    @staticmethod
    def check_inf(features):
        pop_idx = 0
        for i, feat in enumerate(features):
            if True in np.isneginf(feat) or True in np.isinf(feat):
                features = np.delete(features, i-pop_idx, axis=0)
                pop_idx += 1
        return features