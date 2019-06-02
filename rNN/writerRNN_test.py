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
reload(writerRNN)
writerRNN.utf8_to_ascii(
    'Pendant plusieurs jours de suite des lambeaux d\'armée en déroute avaient traversé la ville.')

writerRNN.utf8_to_ascii(
    'SONNET 18\t Shall I compare thee to a summer’s day?\nThou art more lovely and more temperate.\n Rough winds do shake the darling buds of May,\n And summer’s lease hath all too short a date.')

# %% read from files: note the pass-by-object property of python!
files = writerRNN.list_by_extension(path)
files
file = files[1]
file
reload(writerRNN)
text = writerRNN.read_text_from_file(file)
text[:500]

# %% clean up text!
reload(writerRNN)
text = writerRNN.read_text_from_file(file)
text
cleaned_text = writerRNN.clean_text(text)
cleaned_text[:50]

# %% more on clean text
filename = files[1]
folder_path = path
raw_text = writerRNN.read_text_from_file(filename, folder_path)
type(raw_text)
tt = re.sub(' +', ' ', '\n\n  scd ')
tt.split()
text = writerRNN.clean_text(raw_text)
text[:500]
len(text)
# text.index('')
text[-3:]
ttt = ['char', ' ', 'is']
len(ttt)
# text[767]
reload(writerRNN)
n_word = Counter(text)
n_word
len(n_word)
string.whitespace[1:]


# %% word encoder and decoder
vocab = Counter({'deer': 1, 'ray': 2, 'me': 3,
                 'far': 4, 'saw': 5, 'la': 6, 'tea': 7})
encoder, decoder = writerRNN._generate_dictionaries(vocab)
encoder
decoder
len(decoder)

# %%  select which txt file to use
files
reload(writerRNN)
writerRNN.select_book(files)


# %% fetch data
text, encoder, decoder = writerRNN.fetch_data(files[1])
type(text)
cleaned = ''.join(text)
type(cleaned)
cleaned
ascii_str = cleaned.encode('ascii')
clean_file = open((os.path.join(path, 'cleaned_text.txt')), 'wb')
clean_file.close()
