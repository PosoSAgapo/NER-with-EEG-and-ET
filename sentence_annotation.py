import numpy as np
import pandas as pd
import os 
import re

from itertools import islice

def extract_ner_labels(ontonotes_file:str, word_idx=3, ner_idx=10):
    """
        Args: 
            file with ontonotes data (train, dev, or test) in CoNLL format (str),
            word index in each row (int),
            NER label index in each row (int)
        Return:
            list of tuples, where each tuple is (word, label) (number of rows is the same as in orig file)
    """
    onto_word_labels = []
    current_ner_label = ''
    with open(ontonotes_file, 'r', encoding = 'utf-8') as f:
        # skip header
        for line in islice(f, 1, None):
            line = line.split()
            try:
                word = line[word_idx]
                ner_label = re.sub('\*', '', line[ner_idx])
                ner_label = re.sub('\(', '', ner_label)
                current_ner_label = re.sub('\)', '', ner_label) if ner_label.isupper() else current_ner_label
                onto_word_labels.append((word, current_ner_label))
                current_ner_label = '' if re.match('\)', ner_label) else current_ner_label
            except IndexError:
                # catch blank lines
                onto_word_labels.append(('', ''))
    return onto_word_labels

def convert_to_sentence_level(onto_word_level:list):
    """
        Args:
            Ontonotes words and corresponding NER labels on word level (flattened list)
        Return:
            Ontonotes words and corresponding NER labels chunked into sentences (nested list, nested list)
    """
    onto_sents, onto_labels = [], []
    cum_sentlen = 0
    for i, (word, label) in enumerate(onto_word_level):
        if word == '' and label == '':
            sent_len = i - cum_sentlen
            onto_sents.append([word for word, _ in onto_word_level[cum_sentlen: cum_sentlen + sent_len + 1]])
            onto_labels.append([label for _, label in onto_word_level[cum_sentlen: cum_sentlen + sent_len + 1]])
            sent_len = len(onto_sents[-1])
            cum_sentlen += sent_len
    return onto_sents, onto_labels

def annotate_sentences(onto_word_level:list):
    """
        Args:
            Ontonotes words and corresponding NER labels on word level (flattened list)
        Return:
            DataFrame with words and placeholders annotated with binary labels on sentence level (pd.DataFrame)
    """
    relevant_labels = ['PERSON', 'LOC', 'ORG', 'MISC']
    n_elements = len(onto_word_level)
    onto_all_sents, onto_all_labels = convert_to_sentence_level(onto_word_level)
    
    sent_labels = []
    for onto_word_labels in onto_all_labels:
        is_entity = False
        for label in relevant_labels:
            is_entity = True if label in onto_word_labels else is_entity
        if is_entity:
            sent_labels.append('+')
        else:
            sent_labels.append('-')
    
    df = pd.DataFrame(index = range(n_elements+len(sent_labels)), columns = ['word', 'placeholder'])
    cum_idx = 0
    for sent_label, sent in zip(sent_labels, onto_all_sents):
        sent_len = len(sent)
        df.iloc[cum_idx, 0] = sent_label
        df.iloc[cum_idx, 1] = ''
        df.iloc[cum_idx+1:cum_idx+sent_len+1, 0] = sent
        df.iloc[cum_idx+1:cum_idx+sent_len, 1] = '_' #['_' for _ in range(sent_len)]
        df.iloc[cum_idx+sent_len, 1] = ''
        cum_idx += sent_len + 1
    return df