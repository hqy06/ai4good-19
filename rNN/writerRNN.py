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
    file = open(file_path, 'r', encoding='utf-8',)
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
    tokens = tokens[1:-2]
    # lowercase all
    tokens = [t.casefold() for t in tokens]
    # compress multiplewhite space to one without touching delimater
    tokens = [re.sub(' +', ' ', t) for t in tokens]
    # TODO: remove whitespace around delimater
    tokens = [t if t == ' ' else re.sub(' ', '', t) for t in tokens]

    return tokens


def fetch_data(filename, folder_path=PATH):  # TODO
    raw_text = read_text_from_file(filename, folder_path)
    text = clean_text(text)

    n_word = Counter(text)

    pass

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


def main():
    pass


if __name__ == '__main__':
    main()
