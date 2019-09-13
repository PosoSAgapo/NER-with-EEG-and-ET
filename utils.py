import numpy as np
import pandas as pd
import scipy.io as io
import os
import scipy

def get_files(task:str, subdir = '\\results_zuco\\'):
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
    
    def __init__(self, task:str, subject:int):
        tasks = ['task1', 'task2', 'task3']
        if task not in tasks:
            raise Exception('task can only be one of "task1", "task2", or "task3"')
        else:
            self.task = tasks
        subjects = list(range(12))
        if subject not in subjects:
            raise Exception('subject must be an integer value between 0 - 11')
        else:
            self.subject = subject
        
    def __call__(self):
        return mk_dataframe(self)
    
    def mk_dataframe(self):
        """
            Args: Task number ("task1", "task2", "task3") , test subject (0-11)
            Return: DataFrame with features (i.e., attributes) on word level
        """
        files = get_files(self.task)
        data = io.loadmat(files[self.subject], squeeze_me=True, struct_as_record=False)['sentenceData']

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
                df.iloc[k, 0] = str(i) + '_NR' if task=='task1' or task=='task2' else str(i) + '_TSR'
                df.iloc[k, 1] = j
                df.iloc[k, 2:] = [getattr(word, field) if hasattr(word, field) else np.nan\
                                  for field in fields[2:]]
                k += 1

        return df