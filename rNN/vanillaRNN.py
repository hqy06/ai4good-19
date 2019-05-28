"""
A vanilla recurrent neural network created under the guide of official PyTorch tutorial.
------
Tasks:
1. Prepare and explore the data
2. Load the data
3. Create the network
4. Train
5. Evaluate
6. Predict
"""

############################################
# Import modules
############################################
import torch                     # for tensor class
import os                        # for file search
import re                        # regex
import unicodedata               # dealing with utf8
import string                    # working with string
import matplotlib.pyplot as plt  # for ploting
from datetime import datetime    # for timestamp
import numpy as np               # for ndarray

############################################
# Declare global variables: module non-public
############################################
ALL_LETTERS = string.ascii_letters + " .,;'"
PATH = '..\\datasets\\names'


############################################
# Prepare data
############################################
def list_by_extension(path, extension=r".*(\.txt)"):
    # retrieve files with given extension in the given dir
    dirlist = os.listdir(path)
    pattern = re.compile(extension)
    filtered = filter(pattern.match, dirlist)
    files = list(filtered)
    return files


def utf8_to_ascii(utf8_string):
    """ convert utf8 strings to ascii, stripping away accents rather than translate them.
    ------
    Code adopted from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html"""
    return ''.join(
        char for char in unicodedata.normalize('NFD', utf8_string)
        if unicodedata.category(char) != 'Mn'
        and char in ALL_LETTERS
    )


def read_lines_from_file(file_path):
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    return [utf8_to_ascii(l) for l in lines]


def create_lang_name_dict(files, path):
    name_dict = {}
    for file in files:
        file_path = os.path.join(path, file)
        lang = os.path.splitext(os.path.basename(file))[0]
        names = read_lines_from_file(file_path)
        name_dict[lang] = names
    return name_dict

############################################
# Load data
############################################


def _one_hot_char_tensor(char, n_char=len(ALL_LETTERS)):
    # one-hot encode a char
    tensor = torch.zeros(1, n_char)
    char_index = ALL_LETTERS.find(char)
    tensor[0][char_index] = 1
    return tensor


def _one_hot_word_tensor(word, n_char=len(ALL_LETTERS)):
    # one-hot encode a word (string type!)
    assert (len(word) >= 1)
    chars = list(word)      # convert string to a list ot chars

    char = chars.pop(0)
    tensor = _one_hot_char_tensor(char, n_char)

    while len(chars) > 0:
        char = chars.pop(0)
        t = _one_hot_char_tensor(char, n_char)
        tensor = torch.cat([tensor, t])

    assert (tensor.shape[0] == len(word))
    return tensor


# def one_hot_dict_tensor(dict, n_char=len(ALL_LETTERS)):
#     # one-hot the dictionary
#     pass


def map_output_to_category(out, categories):
    # translate numerical values in support to categories in dictionary
    top_values, top_indices = out.topk(1)
    category_index = top_indices[0].item()
    return category_index, categories[category_index]


############################################
# Visualize data
############################################


def _get_value_count(dict, keys):
    # helper function for show_distr_dict
    counts = []
    for key in keys:
        count = len(dict[key])
        counts.append(count)
    return counts


def show_distr_dict(dict, key_name='key', value_name='value', savefig=False):
    keys = list(dict.keys())
    counts = _get_value_count(dict, keys)

    plt.bar(keys, counts, alpha=0.75, color="slateblue")
    plt.title('Distribution of the {}-{} dictionary'.format(key_name, value_name))
    plt.xlabel(key_name)
    plt.tick_params(axis='x', rotation=70)
    plt.ylabel("{} count".format(value_name))

    if savefig:
        plt.savefig(
            '{} - data distribution.png'.format(datetime.now().strftime('%Y%m%d-%H%M')))

    plt.show(block=False)
    plt.pause(2)
    plt.close()


############################################
# Main
############################################


def main(phase):

    # 1. Prepare the data
    # - From the text files, retrieve data to create a lang-names dictionary
    # - To reduce the headache brought by UTF-8, convert the chars to ASCII
    # - Display a histogram for this dataset
    path = PATH
    files = list_by_extension(path)
    lang_name_dict = create_lang_name_dict(files, path)
    categories = list(lang_name_dict.keys())
    n_categories = len(categories)
    n_char = len(ALL_LETTERS)

    if phase == 1:
        print("\nFile search result:\n\t{}".format(files))
        print("\nLanguages:\n\t{}".format(categories))
        print("\nSome numerical facts:\n\t{} language categories, {} letters in the character set".format(
            n_categories, n_char))
        show_distr_dict(lang_name_dict, 'lang', 'name', savefig=True)

    # 2. Load the data
    # - convert dictionary entry to tensor with 1-hot encoding
    # - translating support: numercial output <-> category

    if phase == 2:
        name = lang_name_dict['Spanish'][0]
        print(_one_hot_word_tensor(name).shape)
        print(_one_hot_char_tensor(list(name).pop()).shape)
        foo = np.array([-2.8523, -2.7800, -2.9394, -2.8962, -2.9287, -2.8165, -2.8406, -2.7723, -
                        3.0290, -2.9533, -2.8288, -2.9262, -2.9352, -2.8949, -2.8554, -2.9956, -2.9283, -2.8957])
        tt = torch.from_numpy(foo.reshape(1, 18))
        print(map_output_to_category(tt, categories))

    # 3. Creat the network

    if phase == 3:
        pass

    # 4. Train, Evaluate, Predict

    if phase == 4:
        pass

    return 0


if __name__ == '__main__':
    phase = input("Key in phase of exploration: \n\t0 for nothing, \t1 for data exploration\n\t2 for data loading, \t3 for network creating\n \t4 for train + evaluate + predict:\n\t")
    main(int(phase))
