# The naïve unit test for writerRNN using Hydrogen on Atom

# %% Commonly used libraries and variables
import numpy as np
import os
import string
from collections import Counter
import re
from importlib import reload
import writerRNN
import torch.nn as nn
import torch
path = writerRNN.PATH
ALL_CHARS = writerRNN.ALL_CHARS

# %% utf8_to_ascii for ALL_CHARS including numbers of punctuations
file = '666-temp-by-foo.txt'
writerRNN.utf8_to_ascii(
    'Pendant plusieurs jours de suite des lambeaux d\'armée en déroute avaient traversé la ville.')
writerRNN.utf8_to_ascii(
    'SONNET 18\t Shall I compare thee to a summer’s day?\nThou art more lovely and more temperate.\n Rough winds do shake the darling buds of May,\n And summer’s lease hath all too short a date.')

# %% read from files: note the pass-by-object property of python!
files = writerRNN.list_by_extension(path)
file = files[1]
file
text = writerRNN.read_text_from_file(file)
text[:50]

# %% clean up text!
small_text = '\tSONNET 18\n\nShall I compare thee to a summers day?\nThou art more lovely -- and more temperate.'
small_1 = small_text.replace('--', ' ')
small_1  # remove '--'
small_2 = small_1.translate(str.maketrans({key: " {0} ".format(
    key) for key in (string.whitespace + string.punctuation)}))
small_2  # whitespace between word and non-words
small_3 = small_2
# small_3 = small_2[1:-1]
# small_3  # remove whitespace at head & tail
small_4 = re.sub(" +", " ", small_3)
small_4  # compress whitespaces
small_5 = re.split(r'( +)', small_4)
small_5  # split on whitespace
small_6 = [t.casefold() for t in small_5]
small_6  # make everything lowercase
small_cleaned = small_6
reload(writerRNN)
writerRNN.clean_text(small_text)


# %% fetch_data testing
raw_text = writerRNN.read_text_from_file(file, path)
clean_text = writerRNN.clean_text(raw_text)
vocabulary = Counter(clean_text)
assert (len(vocabulary) <= len(clean_text))
text, encoder, decoder = writerRNN.fetch_data(file, path, verbose=True)

# =========================================================
# End of Phase 1 data preparation

# %%
file_name = file
text, encoder, decoder
n_vocab = len(encoder)

seq_size = 32
embedding_size = 64
lstm_size = 64
rnn = writerRNN.writerRNN(n_vocab, seq_size, embedding_size, lstm_size)

criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

# =========================================================
# End of Phase 2 model declaration


# %%
len(text)
encoder
reload(writerRNN)
encoded_in, encoded_out = writerRNN.load_batches(
    text * 5, encoder, 4, 16, verbose=True)
batch_in = np.reshape(encoded_in, (16, -1))
batch_in.shape
batch_out = np.reshape(encoded_out, (16, -1))
batch_out.shape

xx, yy = [], []
for i in range(0, 23 * 4, 4):
    xx.append(batch_in[:, i:i + 4])
len(xx)

reload(writerRNN)
xx, yy = writerRNN.load_batches(text * 5, encoder, 4, 16, verbose=True)

len(xx)
xx[0].shape

string.whitespace
