import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(infile):
    '''
    Load & prepare fasta protein sequences.
    '''
    # Loading the protein fasta sequences; csv file containing sequences,
    # their labels, and seuence names
    data = pd.read_csv(infile, header=0, sep=',')

    # Get the set of all letters (single letter amino acid codes) in the data
    char_set = list(set("".join(data['sequence'])))
    vocab_size = len(char_set)
    # Creat a dictionary of aino acid letters and their index value in the list.
    # So that we have a numerical code for each letter.
    vocab = dict(zip(char_set, range(vocab_size)))

    # Split data into train & test dataset; using a constant random seed for consistency
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Embedding the characters using the dictioary built above. Basically,
    # replacing  each character with its numerical code in the dictionary.
    train_seqs = np.array([list(map(vocab.get, k)) for k in train_df['sequence']])
    test_seqs = np.array([list(map(vocab.get, k)) for k in test_df['sequence']])

    # Converting categorical labels to actual numbers For two categories of
    # labels, for example, we'll have classes [0, 1].
    train_lebels = np.array(train_df['label'].astype('category').cat.codes)
    test_lebels = np.array(test_df['label'].astype('category').cat.codes)

    return train_seqs, test_seqs, train_lebels, test_lebels

def next_batch(x_data, y_data, batch_size):
    """Returns batches of x & y"""
    '''
    idx = []
    for _ in range(batch_size):
        selected = random.randint(0, x_data.shape[0]-1)
        while selected in idx:
            selected = random.randint(0, x_data.shape[0]-1)
        idx.append(selected)
    '''
    # Choose a random set of row indices with the size of batch_size
    idx = np.random.choice(np.arange(len(x_data)), size=batch_size, replace=False)
    # Return the subset (batch) of data using the randomly chosen row indices
    return x_data[idx, :], y_data[idx]
