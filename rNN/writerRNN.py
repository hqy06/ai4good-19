"""
Let's write like Tolstoy/Lewis Carroll!
--------------------
Coding practice on text-generation using PyTorch.
--------------------
<<References>>
* PyTorch/Numpy/Matplotlib documentations
* Man Page
* A tutorial on word-level rnn using Keras: https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
--------------------
<<TO-DO>>
* TBA
--------------------
<<Good Deeds>>
* the magic of __main__
* import modules, do NOT import functions unless you have to
* follow the naming convention
* you are a 4th-year-to-be CS student, do modular programming even you don't unit test stuffs
"""

# ==========================================================================
# Import modules
# ==========================================================================
import string                    # string maniluplation
import os                        # file search
import unicodedata               # dealing with utf8
import re                        # regex
from collections import Counter  # building vocabulary
from datetime import datetime    # for timestamp
# ==========================================================================
# Declare global variables
# ==========================================================================
PATH = '..\\datasets\\books'
ALL_CHARS = string.printable

# ==========================================================================
# Data preprocessing
# ==========================================================================


def utf8_to_ascii(utf8_string):
    """
    convert utf8 strings to ascii, stripping away accents rather than translate them.
    ------
    Code adopted from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    Copy and paste from vanillaRNN.py
    """
    return ''.join(
        char for char in unicodedata.normalize('NFD', utf8_string)
        if unicodedata.category(char) != 'Mn'
        and char in ALL_CHARS
    )


def list_by_extension(path, extension=r".*(\.txt)"):
    """retrieve files with given extension in the given dir
    -------
    Copy and paste from vanillaRNN.py
    """
    dirlist = os.listdir(path)
    pattern = re.compile(extension)
    filtered = filter(pattern.match, dirlist)
    files = list(filtered)
    return files


def read_text_from_file(filename, folder_path=PATH):
    file_path = os.path.join(folder_path, filename)
    file = open(file_path, 'r', encoding='utf-8')
    text = file.read()
    file.close()
    return utf8_to_ascii(text)


def clean_text(text):  # TODO: clean and the text
    # replace '--' with a white space
    text = text.replace('--', ' ')
    # insert whitespace between punctuations and words
    text = text.translate(str.maketrans(
        {key: " {0} ".format(key) for key in string.punctuation}))
    # split according to white space
    tokens = re.split(r'(\s+)', text)
    # remove the empty string '' in the front
    tokens = tokens[:-2]
    # lowercase all
    tokens = [t.casefold() for t in tokens]
    # compress multiplewhite space to one without touching delimater
    tokens = [re.sub(' +', ' ', t) for t in tokens]
    # TODO: remove whitespace around delimater
    tokens = [t if t == ' ' else re.sub(' ', '', t) for t in tokens]

    return tokens


def fetch_data(filename, folder_path=PATH, verbose=False):
    raw_text = read_text_from_file(filename, folder_path)
    text = clean_text(raw_text)

    vocabulary = Counter(text)

    if verbose:
        print('number of words: {}'.format(len(vocabulary)))

    encoder_dict, decoder_dict = _generate_dictionaries(vocabulary)

    return text, encoder_dict, decoder_dict


def _generate_dictionaries(vocabulary):
    sorted_vocab = sorted(vocabulary, key=vocabulary.get, reverse=True)
    # use reverse=True because we want the indices of commonly used word smaller
    word_to_int = {i: word for i, word in enumerate(sorted_vocab)}
    int_to_word = {word: i for i, word in enumerate(sorted_vocab)}
    return word_to_int, int_to_word


def select_book(file_list):
    print("Select which file to use:")
    for i in range(len(file_list)):
        print("\t {:>3} | {}".format(i, file_list[i]))
    file_no = input("Text file chosed: ")
    return file_list[int(file_no)]


def save_to_file_ascii(some_text, name, path=None):
    file_name = '{} - {}.txt'.format(
        datetime.now().strftime('%Y%m%d-%H%M'), name)
    a_file = open(file_name, 'wb')
    ascii_text = (''.join(some_text)).encode('ascii')
    a_file.write(ascii_text)
    a_file.close()


# ==========================================================================
# Network definition
# ==========================================================================

class writerRNN(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(writerRNN, self).__init__()
        # size of the input sequence
        self.seq_size = seq_size
        # size of LSTM layer
        self.lstm_size = lstm_size
        # embedding layer for LSTM
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        # long-short term memory
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        # sometimes this is called the "dense" layer
        self.fc = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embd = self.embedding(x)
        output, state = self.lstm(embd, prev_state)
        logits = self.fc(output)    #
        return logits, state

    def init_zeros(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

# ==========================================================================
# Data loading
# ==========================================================================


# ==========================================================================
# Visualization
# ==========================================================================


# ==========================================================================
# Main
# ==========================================================================


def main(phase, verbose=False):
    path = PATH
    char_set = ALL_CHARS
    files = list_by_extension(path)

    if phase == 1:
        # use the small test file
        file_name = '666-temp-by-foo.txt'
        text, encoder, decoer = fetch_data(file_name)
        # save the cleaned result into a text file
        save_to_file_ascii(text, 'cleaned_text')
        exit()

    if verbose:
        print("\n========== Preparing Data ==========")

    file_name = select_book(files)
    text, encoder, decoder = fetch_data(
        file_name, folder_path=path, verbose=verbose)
    save_to_file_ascii(text, 'cleaned_{}'.format(file_name))
    assert (len(encoder) == len(decoder))
    n_vocab = len(encoder)

    if verbose:
        print("\n========== Create Model ==========")

    criterion = nn.CrossEntropyLoss()
    lr = 0.001   # learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    seq_size = 32
    embedding_size = 64
    lstm_size = 64  # COMBAK:
    rnn = writerRNN(n_vocab, seq_size, embedding_size, lstm_size)

    if verbose:
        print("\n========== Load Data ==========")

    if phase == 2:  # TODO: one iteration
        # seq = _one_hot_word_tensor('Albert')
        # hidden = torch.zeros(1, n_hidden)
        # out, next_hidden = rnn(word[0].view(1, -1), hidden)
        # print(out)

    return 0


if __name__ == '__main__':
    phase = input('key in phase number: ')
    main(int(phase), verbose=True)
