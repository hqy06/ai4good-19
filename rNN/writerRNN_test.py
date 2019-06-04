# The naïve unit test for writerRNN using Hydrogen on Atom

# %% Commonly used libraries and variables
import os
import string
from collections import Counter
import re
from importlib import reload
import writerRNN
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


# %%
raw_text = writerRNN.read_text_from_file(file, path)
raw_text
clean_text = writerRNN.clean_text(raw_text)
clean_text


# # %% more on clean text
# filename = files[1]
# folder_path = path
# raw_text = writerRNN.read_text_from_file(filename, folder_path)
# type(raw_text)
# tt = re.sub(' +', ' ', '\n\n  scd ')
# tt.split()
# text = writerRNN.clean_text(raw_text)
# text[:500]
# len(text)
# # text.index('')
# text[-3:]
# ttt = ['char', ' ', 'is']
# len(ttt)
# # text[767]
# reload(writerRNN)
# n_word = Counter(text)
# n_word
# len(n_word)
# string.whitespace[1:]
#
#
# # %% word encoder and decoder
# vocab = Counter({'deer': 1, 'ray': 2, 'me': 3,
#                  'far': 4, 'saw': 5, 'la': 6, 'tea': 7})
# encoder, decoder = writerRNN._generate_dictionaries(vocab)
# encoder
# decoder
# len(decoder)
#
# # %%  select which txt file to use
# files
# reload(writerRNN)
# writerRNN.select_book(files)
#
#
# # %% fetch data
# text, encoder, decoder = writerRNN.fetch_data(files[1])
# type(text)
# cleaned = ''.join(text)
# type(cleaned)
# cleaned
# ascii_str = cleaned.encode('ascii')
# clean_file = open((os.path.join(path, 'cleaned_text.txt')), 'wb')
# clean_file.close()
#
#
# # %%
# text[:100]
# small_text = text[:50]
# small_text
# encoder
# text[:10]
