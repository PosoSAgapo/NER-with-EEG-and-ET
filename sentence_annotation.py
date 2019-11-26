__all__ = ['load_wiki_dataset', 'load_semeval2010_files', 'filter_sents', 'extract_wiki_rels', 'extract_rels_semeval2007', 'extract_rel_labels', 'extract_ner_labels', 'convert_to_sentence_level', 'annotate_sentences']

import pandas as pd
import os 
import re
import random

from collections import Counter, defaultdict
from itertools import islice

random.seed(42)

def load_wiki_dataset(subdir:str='./datasets/Wikipedia/csv/all_relation_sents/'):
    files = iter(map(lambda file:subdir+file, os.listdir(subdir)))
    all_rels = pd.concat([pd.read_csv(file, header=None, encoding='utf-8') for file in files])
    all_rels.drop(columns=[0, 1], inplace=True)
    all_rels.rename(columns={2: "sent", 3: "label"}, inplace=True)
    return all_rels

def load_semeval2010_files(subdir:str='./datasets/SemEval2010/SemEval2010_task8_all_data/', train:bool=True, labels:bool=False):
    task = 'SemEval2010_task8'
    dataset = task + '_' + 'training/' if train else task + '_' + 'testing'
    if not train: dataset = dataset + '_' + 'keys/' if labels else dataset + '/'
    if train:
        file = 'TRAIN_FILE.TXT'
    elif not train and not labels:
        file = 'TEST_FILE.TXT'
    elif not train and labels:
        file = 'TEST_FILE_KEY.TXT'
    path = subdir + dataset + file
    return path

def filter_sents(sents:list, labels:list, zuco_sents:list):
    sents, labels = zip(*[(sent, label) for sent, label in zip(sents, labels) if sent not in zuco_sents])
    return (sents, labels)

def extract_wiki_rels(relations, thresh:int=100):
    n_relations = Counter(relations.label.values)
    rels_to_keep, _ = zip(*list(filter(lambda kv: kv[1] >= thresh, n_relations.items())))
    rels = defaultdict(list)
    for idx, row in relations.iterrows():
        if isinstance(row.sent, str):
            sent, label = row.sent, row.label
            if label in rels_to_keep:
                if label not in rels[sent]:
                    rels[sent].append(label)
    sents, labels = zip(*rels.items())
    sents = list(map(lambda sent: sent.split() + [""], sents))
    return sents, labels

def extract_rels_semeval2007(dataset:str, subdir:str='./datasets/SemEval2007/SemEval2007-Task4/'):
    files = [subdir+dataset+'/'+file for file in os.listdir(subdir+dataset) if file.endswith('.txt')]
    relations = ['Cause-Effect', 'Instrument-Agency', 'Product-Producer', 'Origin-Entity', 'Theme-Tool',
                 'Part-Whole', 'Content-Container']
    assert len(files) == len(relations) + 1
    line_start = 3
    sents, labels = [], []
    for i, file in enumerate(islice(files, 1, None)):
        with open(file, 'r', encoding='latin-1') as f:
            for j, line in enumerate(f):
                try:
                    _ = int(line[0]) # check if first character in line is a digit
                    sent = line[line_start:].strip().strip('"')
                    sent = re.sub('(<[/]*e[0-9]+>)', '', sent)
                    sent = sent.split()
                    sent.append("") # append blank line
                    sents.append(sent)
                except:
                    if line.startswith('WordNet'):
                        rel = relations[i]+'(e#,e#)'+' = '
                        rel_start = line.index(relations[i])
                        rel_end = line.index(', Query')
                        if re.search("true", line[rel_start+len(rel):rel_end]):
                            labels.append(relations[i])
                        else:
                            labels.append('Negative')
    return sents, labels

def extract_rel_labels(file:str, start_idx:int=0, train:bool=True, keys:bool=False):
    sents, labels = [], []
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(islice(f, start_idx, None)):
            try:
                _ = int(line[0]) # check if first character is a digit
                line_start = len(str(i)) if train else len(str(i + 8000))
                sent = line[line_start:].strip().strip('"')
                sent = re.sub('(<[/]*e[0-9]+>)', '', sent)
                if keys:
                    labels.append(sent)
                else:
                    sent = sent.split() # convert string into list
                    sent.append('') # we need blank lines after each sentence for binary sentence classification objective
                    sents.append(sent)
            except ValueError:
                if train:
                    if i % 2 == 0 or line.strip() is "":
                        continue
                    else:
                        pass
                else:
                    continue
            if train:
                if i % 2 != 0:
                    label = re.sub('(\((e[0-9]+[,]*)+\))', '', line).strip()
                    labels.append(label)
    if train:
        return sents, labels
    elif not train and not keys:
        return sents
    elif not train and keys:
        return labels

def extract_ner_labels(file:str, dataset:str, w_idx:int, ner_idx:int):
    """
    Args: 
        file with data (train, dev or test) in CoNLL format (str),
        dataset (str),
        word index in each row (int) - (0 for CoNLL2003, 3 for ontonotes),
        NER label index in each row (int) - (3 for CoNLL2003, 10 for ontonotes)
    Return:
        list of tuples, where each tuple is (word, label) (number of rows is the same as in original file)
    """
    datasets = ['conll', 'ontonotes']
    if dataset not in datasets:
        raise Exception("Dataset must be one of ['ontonotes', 'conll']")
    word_labels = []
    start_idx = 2 if dataset == 'conll' else 1
    current_ner_label = ''
    with open(file, 'r', encoding = 'utf-8') as f:
        # skip header (islice(iterator, start, stop, [step]))
        for i, line in enumerate(islice(f, start_idx, None)):
            line = line.split()
            try:
                if dataset == 'ontonotes':
                    word = line[w_idx]
                    ner_label = re.sub('\*', '', line[ner_idx])
                    ner_label = re.sub('\(', '', ner_label)
                    current_ner_label = re.sub('\)', '', ner_label) if ner_label.isupper() else current_ner_label
                    word_labels.append((word, current_ner_label))
                    current_ner_label = '' if re.match('\)', ner_label) else current_ner_label
                elif dataset == 'conll':
                    ner_label = re.sub('.-', '', line[ner_idx])
                    word = line[w_idx].capitalize() if (ner_label != 'O' or i == 0 or word_labels[-1] == ('','')) else line[w_idx].lower() 
                    word_labels.append((word, ner_label))        
            except IndexError:
                # catch blank lines
                word_labels.append(('',''))
    return word_labels

def convert_to_sentence_level(word_level_data:list):
    """
    Args:
        Words and corresponding NER labels in CoNLL 2003 format on word level (flattened list)
    Return:
        Words and corresponding NER labels chunked into sentences (nested list, nested list)
    """
    sents, labels = [], []
    cum_sentlen = 0
    for i, (word, label) in enumerate(word_level_data):
        if word == '' and label == '':
            sent_len = i - cum_sentlen
            sents.append(list(map(lambda wl: wl[0], word_level_data[cum_sentlen: cum_sentlen + sent_len + 1])))
            labels.append(list(map(lambda wl: wl[1], word_level_data[cum_sentlen: cum_sentlen + sent_len + 1])))
            sent_len = len(sents[-1])
            cum_sentlen += sent_len
    return sents, labels

def annotate_sentences(word_level_data, task:str, subtask=None):
    """
    Args:
        Words and corresponding NER labels on word level (flattened list)
    Return:
        DataFrame with words and placeholders ('_') annotated with binary labels ('+' or '-') on sentence level (pd.DataFrame)
    """
    if task == 'NER':
        relevant_labels = ['PERSON', 'PER', 'LOC', 'ORG', 'MISC'] #CoNLL 2003 NER labels
        n_elements = len(word_level_data)
        all_sents, all_labels = convert_to_sentence_level(word_level_data)

        sent_labels = []
        for word_labels in all_labels:
            is_entity = False
            for label in relevant_labels:
                is_entity = True if label in word_labels else is_entity
            if is_entity:
                sent_labels.append('+')
            else:
                sent_labels.append('-')
                
    elif task == 'RelExtract':
        assert isinstance(subtask, str), 'subtask must be specified for RelExtract datasets'
        all_sents, all_labels = word_level_data
        n_elements = sum(list(map(lambda sent: len(sent), all_sents)))
        #unique_labels = list(set(all_labels))
        if subtask != 'Wiki':
            n_labels = Counter(all_labels)
        if subtask == 'SemEval2010':
            n_pos = n_labels['Entity-Origin'] + n_labels['Entity-Destination']
            relevant_relations = ['Entity-Origin', 'Entity-Destination']
        elif subtask == 'SemEval2007':
            n_pos = len(all_labels) - n_labels['Negative']
        elif subtask == 'Wiki':
            pos_rel = 'JOB_TITLE'
            n_pos = len(list(filter(lambda label: pos_rel in label, all_labels)))
        sent_labels = []
        for label in all_labels:
            if subtask == 'SemEval2010':
                if label in relevant_relations:
                    sent_labels.append('+')
                else:
                    sent_labels.append('-')
            elif subtask == 'SemEval2007':
                if label == 'Negative':
                    sent_labels.append('-')
                else:
                    sent_labels.append('+')
            elif subtask == 'Wiki':
                if pos_rel in label:
                    sent_labels.append('+')
                else:
                    sent_labels.append('-')
                    
        assert sent_labels.count('+') == n_pos
    
    df = pd.DataFrame(index = range(n_elements+len(sent_labels)), columns = ['word', 'placeholder'])
    cum_idx = 0
    for i, (sent_label, sent) in enumerate(zip(sent_labels, all_sents)):
        sent_len = len(sent)
        df.iloc[cum_idx, 0] = sent_label
        df.iloc[cum_idx, 1] = ''
        df.iloc[cum_idx+1:cum_idx+sent_len+1, 0] = sent
        df.iloc[cum_idx+1:cum_idx+sent_len, 1] = '_' #['_' for _ in range(sent_len)]
        df.iloc[cum_idx+sent_len, 1] = ''
        cum_idx += sent_len + 1
        if i > 0 and i % 5000 == 0:
            print('{} sentences processed...'.format(i))
    print('Sentence annotation finished!')
    return df