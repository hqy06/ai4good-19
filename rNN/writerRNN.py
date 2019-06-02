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


def fetch_data(filename, folder_path=PATH):  # TODO
    raw_text = read_text_from_file(filename, folder_path)
    text = clean_text(raw_text)

    vocabulary = Counter(text)

    # n_word = len(vocabulary)
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


# ==========================================================================
# Data loading
# ==========================================================================


# ==========================================================================
# Visualization
# ==========================================================================


# ==========================================================================
# Main
# ==========================================================================


def main(phase=0, verbose=False):
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
    text, encoder, decoder = fetch_data(file_name, folder_path=path)

    return 0


if __name__ == '__main__':
    phase = input('key in phase number: ')
    main(int(phase))
