import numpy as np
import pandas as pd
import seaborn as sns 
import scipy.io as io
import gzip
import math
import os
import random
import re
import scipy

from collections import defaultdict
from scipy.stats import pearsonr, ttest_rel
from plot_funcs import hinton


def get_bncfreq(subdir = '\\BNC\\', file = 'all.al.gz', n_fields = 4):
    """
        Args: British National Corpus word frequency list;
              four fields per line (0: freq, 1: word, 2: POS, 4: n_files the word occurs in)
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

class DataTransformer:
    """
        Transforms ET (and EEG data) to use for further analysis (per test subject)
    """
    
    def __init__(self, task:str, level:str, data='ET', scaling='min-max', fillna='zeros'):
        """
            Args: task ("task1", "task2", or "task3"), data level, scaling technique, how to treat NaNs
        """
        tasks = ['task1', 'task2', 'task3']
        if task in tasks:
            self.task = task
        else:
            raise Exception('Task can only be one of "task1", "task2", or "task3"') 
        levels = ['sentence', 'word']
        if level in levels:
            self.level = level
        else:
            raise Exception('Data can only be processed on sentence or word level')
        data_sources = ['ET', 'EEG']
        if data in data_sources:
            self.data = data
        else:
            raise Exception('Features can be extracted either for Eye-Tracking (ET) or EEG data')
        #display raw (absolut) values or normalize data according to specified feature scaling technique
        feature_scalings = ['min-max', 'mean-norm', 'standard', 'raw']
        if scaling in feature_scalings:
            self.scaling = scaling
        else:
            raise Exception('Features must either be min-max scaled, mean-normalized or standardized')
        fillnans = ['zeros', 'mean', 'min']
        if fillna in fillnans:
            self.fillna = fillna
        else:
            raise Exception('Missing values must be replaced with zeros, the mean or min per feature')
    
    def __call__(self, subject:int):
        """
            Args: test subject (0-11)
            Return: DataFrame with normalized features (i.e., attributes) on sentence or word level
        """
        # subject should not be a property of data transform object (thus, it's not in the init method), 
        # since we want to apply the same data transformation to each subject
        subjects = list(range(12))
        if subject not in subjects:
            raise Exception('Access subject data with an integer value between 0 - 11')  
        bnc_freq = get_bncfreq()
        files = get_matfiles(self.task)
        data = io.loadmat(files[subject], squeeze_me=True, struct_as_record=False)['sentenceData']
        
        if self.level == 'sentence':
            if self.data == 'ET':
                fields = ['SentLen', 'MeanWordLen',  'omissionRate', 'nFixations', 'meanPupilSize', 
                          'GD', 'TRT', 'FFD', 'GPT', 'BNCFreq']
            elif self.data == 'EEG':
                fields = ['SentLen', 'MeanWordLen',  'omissionRate', 'mean_theta', 'mean_alpha', 'mean_beta', 'mean_gamma',
                          'nFixations', 'meanPupilSize', 'BNCFreq',]
                start_eeg = fields.index('mean_theta')
            start_et = fields.index('nFixations')
            if self.task == 'task1' and subject == 2:
                features = np.zeros((len(data)-101, len(fields)))
            elif self.task == 'task2' and (subject == 6 or subject == 11):
                features = np.zeros((len(data)-50, len(fields)))
            elif self.task == 'task3' and subject == 3:
                features = np.zeros((len(data)-47, len(fields)))
            elif self.task == 'task3' and subject == 7:
                features = np.zeros((len(data)-48, len(fields)))
            elif self.task == 'task3' and subject == 11:
                features = np.zeros((len(data)-89, len(fields)))
            else:
                features = np.zeros((len(data), len(fields)))

        elif self.level == 'word':
            if self.task == 'task1' and subject == 2:
                n_words = sum([len(sent.word) for i, sent in enumerate(data[:-1]) if i < 150 or i > 249])
            elif self.task == 'task2' and subject == 6:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i > 49])  
            elif self.task == 'task2' and subject == 11:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 50 or i > 99])
            elif self.task == 'task3' and subject == 3:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 178 or i > 224])
            elif self.task == 'task3' and subject == 7:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 359])
            elif self.task == 'task3' and subject == 11:
                n_words = sum([len(sent.word) for i, sent in enumerate(data) if i < 270 or (i > 313 and i < 362)])
            else:
                n_words = sum([len(sent.word) for sent in data])
            fields = ['Sent_ID', 'Word_ID', 'Word', 'nFixations', 'meanPupilSize', 
                      'GD', 'TRT', 'FFD', 'GPT', 'WordLen', 'BNCFreq']
            df = pd.DataFrame(index=range(n_words), columns=[fields])
            k = 0
        
        idx = 0
        for i, sent in enumerate(data):
            if (self.task == 'task1' and subject == 2) and ((i >= 150 and i <= 249) or i == 399):
                continue
            elif (self.task == 'task2' and subject == 6) and (i <= 49):
                continue
            elif (self.task == 'task2' and subject == 11) and (i >= 50 and i <= 99):
                continue
            elif (self.task == 'task3' and subject == 3) and (i >= 178 and i <= 224):
                continue
            elif (self.task == 'task3' and subject == 7) and (i >= 359):
                continue
            elif (self.task == 'task3' and subject == 11) and ((i >= 270 and i <= 313) or (i >= 362 and i <= 406)):
                continue
            else:
                n_words_fixated = 0
                total_word_len = 0
                for j, word in enumerate(sent.word):
                    token = re.sub('[^\w\s]', '', word.content)
                    #lowercase words at the beginning of the sentence only
                    token = token.lower() if j == 0 else token
                    total_word_len += len(token)
                    if self.level == 'sentence':
                        et_features = [getattr(word, field) if hasattr(word, field)\
                                         and not isinstance(getattr(word, field), np.ndarray) else\
                                         0 for field in fields[start_et:-1]]
                        features[idx, start_et:-1] += et_features
                        n_words_fixated += 0 if (len(set(et_features)) == 1 and next(iter(set(et_features))) == 0) else 1
                        
                        #NOTE: we have to divide bnc freq by 100 to get freq by million (bnc freq is computed for 100 million words)
                        features[idx, -1] += np.log(bnc_freq[token]/100) if bnc_freq[token]/100 != 0 else 0 
                        
                    elif self.level == 'word':
                        df.iloc[k, 0] = str(idx)+'_NR' if self.task=='task1' or self.task=='task2'\
                                        else str(idx)+'_TSR'
                        df.iloc[k, 1] = j
                        df.iloc[k, 2] = token
                        df.iloc[k, 3:-2] = np.array([getattr(word, field) if hasattr(word, field)\
                                            and not isinstance(getattr(word, field), np.ndarray) else\
                                            0 for field in fields[3:-2]])
                        df.iloc[k,-2] = len(token)
                        #NOTE: we have to divide bnc freq by 100 to get freq by million (bnc freq is computed for 100 million words)
                        df.iloc[k,-1] = np.log(bnc_freq[token]/100) if bnc_freq[token]/100 != 0 else 0
                        k += 1

                if self.level == 'sentence':
                    sent_len = len(sent.word)
                    features[idx, 0] = sent_len
                    # divide total number of characters in sent by number of words in sent to get mean word len per sent
                    features[idx, 1] = total_word_len / sent_len
                    features[idx, 2] = sent.omissionRate
                    if self.data == 'EEG':
                        #NOTE: for each frequency domain, eeg values are stored in np.array (105 values for each electrode)
                        #NOTE: first stack values per sub-domain, then average over single electrodes
                        binned_eeg_theta = np.mean(np.vstack([getattr(sent, theta) if hasattr(sent, theta) and len(getattr(sent, theta))\
                                                              > 0 else 0 for theta in ['mean_t1', 'mean_t2']]), axis=0)
                        binned_eeg_alpha = np.mean(np.vstack([getattr(sent, alpha) if hasattr(sent, alpha) and len(getattr(sent, alpha))\
                                                              > 0 else 0 for alpha in ['mean_a1', 'mean_a2']]), axis=0)
                        binned_eeg_beta = np.mean(np.vstack([getattr(sent, beta) if hasattr(sent, beta) and len(getattr(sent, beta))\
                                                             > 0 else 0 for beta in ['mean_b1', 'mean_b2']]), axis=0)
                        binned_eeg_gamma = np.mean(np.vstack([getattr(sent, gamma) if hasattr(sent, gamma) and len(getattr(sent, gamma))\
                                                              > 0 else 0 for gamma in ['mean_g1', 'mean_g2']]), axis=0)
                        freq_domains = [binned_eeg_theta, binned_eeg_alpha, binned_eeg_beta, binned_eeg_gamma]
                        eeg_features = np.array([np.average(freq_domain) for freq_domain in freq_domains])
                        eeg_features[np.isnan(eeg_features)] = 0
                        features[idx, start_eeg:start_et] = eeg_features
                    # normalize ET features by number of words for which fixations were reported
                    features[idx, start_et:-1] /= n_words_fixated
                    # normalize bnc freq by number of words in sentence
                    features[idx, -1] /= len(sent.word)
                
                idx += 1

        #handle -inf, inf and NaN values
        if self.level == 'sentence': 
            features = self.check_inf(features)
            
        elif self.level == 'word':
            if self.fillna == 'zeros':
                df.iloc[:,:-1].fillna(0, inplace=True)
            elif self.fillna == 'min':
                for i, field in enumerate(fields[:-1]):
                    df.iloc[:,i].fillna(getattr(df, field).values.min(), inplace=True)
            elif self.fillna == 'mean':
                for i, field in enumerate(fields[:-1]):
                    df.iloc[:,i].fillna(getattr(df, field).values.mean(), inplace=True)
            
            # replace NaNs for BNC freq with min BNC freq (not with zeros)
            df.iloc[:,-1].fillna(getattr(df,fields[-1]).values.min(), inplace=True)    
            df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

        #normalize data according to feature scaling technique
        if self.scaling == 'min-max':
            if self.level == 'sentence':
                features = np.array([(feat - min(feat))/(max(feat) - min(feat)) for feat in features.T])
            elif self.level == 'word':
                df.iloc[:, 3:] = [(getattr(df,field).values - getattr(df,field).values.min())/\
                                  (getattr(df,field).values.max() - getattr(df,field).values.min())\
                                  for field in fields[3:]]
                
        elif self.scaling == 'mean-norm':
            if self.level == 'sentence':
                features = np.array([(feat - np.mean(feat))/(max(feat) - min(feat)) for feat in features.T])
            elif self.level == 'word':
                df.iloc[:, 3:] = [(getattr(df,field).values - getattr(df,field).values.mean())/\
                                  (getattr(df,field).values.max() - getattr(df,field).values.min())\
                                  for field in fields[3:]]
                
        elif self.scaling == 'standard':
            if self.level == 'sentence':
                features = np.array([(feat - np.mean(feat))/np.std(feat) for feat in features.T])
            elif self.level == 'word':
                df.iloc[:, 3:] = [(getattr(df,field).values - getattr(df,field).values.mean())/\
                                  getattr(df,field).values.std() for field in fields[3:]]
                
        if self.level == 'sentence':
            if self.scaling == 'raw':
                df = pd.DataFrame(data=features, index=range(features.shape[0]), columns=[fields])
            else:
                df = pd.DataFrame(data=features.T, index=range(features.shape[1]), columns=[fields])
                
            if self.fillna == 'zeros':
                df.iloc[:,:-1].fillna(0, inplace=True)
            elif self.fillna == 'min':
                for i, field in enumerate(fields[:-1]):
                    df.iloc[:,i].fillna(getattr(df, field).values.min(), inplace=True)
            elif self.fillna == 'mean':
                for i, field in enumerate(fields[:-1]):
                    df.iloc[:,i].fillna(getattr(df, field).values.mean(), inplace=True)
            df.iloc[:,-1].fillna(getattr(df,fields[-1]).values.min(), inplace=True)    
           
        return df
    
    @staticmethod
    def check_inf(features):
        pop_idx = 0
        for idx, feat in enumerate(features):
            if True in np.isneginf(feat) or True in np.isinf(feat):
                features = np.delete(features, idx-pop_idx, axis=0)
                pop_idx += 1
        return features
    
def split_data(sbjs): 
    """
        Args: Data per sbj on sentence level for task 1
        Purpose: Function is necessary to control for order effects (only relevant for Task 1 (NR))
    """
    first_half, second_half = [], []
    for sbj in sbjs:
        first_half.append(sbj[:len(sbj)//2])
        second_half.append(sbj[len(sbj)//2:])
    return first_half, second_half

def corr_mat(data, features:str, heatmap=False, mask=False, hinton_mat=False):
    if features=='ET':
        fields = ['SentLen', 'MeanWordLen', 'omissionRate', 'nFixations', 'meanPupilSize', 
                  'GD', 'TRT', 'FFD', 'GPT', 'BNCFreq']
    elif features=='EEG':
        fields = ['SentLen', 'MeanWordLen',  'omissionRate', 'nFixations', 'meanPupilSize',
                  'mean_theta', 'mean_alpha', 'mean_beta', 'mean_gamma', 'BNCFreq',]
    corr_mat = np.zeros((len(data), len(data)))
    for i, feat_x in enumerate(data):
        for j, feat_y in enumerate(data):
            corr_mat[i, j] = pearsonr(feat_x, feat_y)[0]
    df = pd.DataFrame(corr_mat, index = fields, columns = fields)
    if heatmap:
        if mask:
            mask = np.zeros_like(corr_mat)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("white"):
                return sns.heatmap(df, vmax=1., mask=mask, cmap="YlGnBu")
        else:
            return sns.heatmap(df, vmax=1., cmap="YlGnBu")
    elif hinton_mat:
        return hinton(df)
    else:
        return df

def compute_means(task, features:str):
    sentlen, wordlen, omissions, fixations, pupilsize, bncfreq = [], [], [], [], [], []
    if features == 'ET':
        gd, trt, ffd, gpt = [], [], [], []
    elif features == 'EEG':
        mean_theta, mean_alpha, mean_beta, mean_gamma = [], [], [], []
    for sbj in task:
        sentlen.append(sbj.SentLen.values.mean())
        wordlen.append(sbj.MeanWordLen.values.mean())
        omissions.append(sbj.omissionRate.values.mean())
        fixations.append(sbj.nFixations.values.mean())
        pupilsize.append(sbj.meanPupilSize.values.mean())
        bncfreq.append(sbj.BNCFreq.values.mean())
        if features == 'ET':
            gd.append(sbj.GD.values.mean())
            trt.append(sbj.TRT.values.mean())
            ffd.append(sbj.FFD.values.mean())
            gpt.append(sbj.GPT.values.mean())
        elif features == 'EEG':
            mean_theta.append(sbj.mean_theta.values.mean())
            mean_alpha.append(sbj.mean_alpha.values.mean())
            mean_beta.append(sbj.mean_beta.values.mean())
            mean_gamma.append(sbj.mean_gamma.values.mean())
    if features == 'ET':
        return sentlen, wordlen, omissions, fixations, pupilsize, gd, trt, ffd, gpt, bncfreq
    elif features == 'EEG':
        return sentlen, wordlen, omissions, fixations, pupilsize, mean_theta, mean_alpha, mean_beta, mean_gamma, bncfreq


def compute_allvals(task, features:str):
    sentlens = [val[0] for sbj in task for val in sbj.SentLen.values]
    wordlens = [val[0] for sbj in task for val in sbj.MeanWordLen.values]
    omissions = [val[0] for sbj in task for val in sbj.omissionRate.values]
    fixations = [val[0] for sbj in task for val in sbj.nFixations.values]
    pupilsize = [val[0] for sbj in task for val in sbj.meanPupilSize.values]
    bnc_freqs = [val[0] for sbj in task for val in sbj.BNCFreq.values]
    if features == 'ET':
        gd = [val[0] for sbj in task for val in sbj.GD.values]
        trt = [val[0] for sbj in task for val in sbj.TRT.values]
        ffd = [val[0] for sbj in task for val in sbj.FFD.values]
        gpt = [val[0] for sbj in task for val in sbj.GPT.values]
        return sentlens, wordlens, omissions, fixations, pupilsize, gd, trt, ffd, gpt, bnc_freqs
    elif features == 'EEG':
        mean_theta = [val[0] for sbj in task for val in sbj.mean_theta.values]
        mean_alpha = [val[0] for sbj in task for val in sbj.mean_alpha.values]
        mean_beta = [val[0] for sbj in task for val in sbj.mean_beta.values]
        mean_gamma = [val[0] for sbj in task for val in sbj.mean_gamma.values]
        return sentlens, wordlens, omissions, fixations, pupilsize, mean_theta, mean_alpha, mean_beta, mean_gamma, bnc_freqs

def randomsample_paired_ttest(vals_nr:list, vals_tsr:list):
    """
        Args: feature values for NR, feature values for TSR
        Return: p-value (computed by dependent t-test)
    """
    #randomly sample N sentences for each task to have equally sized list of values
    k = min(len(vals_nr), len(vals_tsr)) // 2
    random.seed(42)
    random_samples_nr = random.sample(vals_nr, k)
    random_sample_tsr = random.sample(vals_tsr, k)
    #paired t-test (because we compare within subjects between tasks - subjects are always the same)
    p_val = ttest_rel(random_samples_nr, random_sample_tsr)[1]
    return p_val