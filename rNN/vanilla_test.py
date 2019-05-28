<<<<<<< HEAD
# Fragmental testing script with Atom Hydrogen package

# %%
import numpy as np
import torch
=======
# %%
>>>>>>> d5d88d6016ef850023d9d9cd4dcc8cd0a8350278
import sys
import vanillaRNN
from importlib import reload
import string


# %%
ALL_LETTERS = string.ascii_letters + " .,;'"
PATH = '..\\datasets\\names'

<<<<<<< HEAD
=======

>>>>>>> d5d88d6016ef850023d9d9cd4dcc8cd0a8350278
# %%
path = PATH
files = vanillaRNN.list_by_extension(path)
lang_name_dict = vanillaRNN.create_lang_name_dict(files, path)
categories = list(lang_name_dict.keys())
n_categories = len(categories)
n_char = len(ALL_LETTERS)

<<<<<<< HEAD
# ======================================================== Done!
=======
>>>>>>> d5d88d6016ef850023d9d9cd4dcc8cd0a8350278

# %%
langs = categories
type(langs)

lang_name_dict['Spanish']
len(lang_name_dict['Spanish'])

name_counts = []
for lang in langs:
    count = len(lang_name_dict[lang])
    name_counts.append(count)

# %%
reload(vanillaRNN)
vanillaRNN.show_distribtuion_dict(
    lang_name_dict, key_name="lang", value_name='name')
<<<<<<< HEAD
# ======================================================== Done!

# %%
# Testing function: _one_hot_char_tensor
reload(vanillaRNN)

vanillaRNN._one_hot_char_tensor('s')

# Testing torch.stack: dimensional sanity check
s = vanillaRNN._one_hot_char_tensor('s')
t = vanillaRNN._one_hot_char_tensor('t')

st = torch.stack([s, t])

st.shape
s.shape

# %%
name = lang_name_dict['Spanish'][0]
name
type(name)
name_list = list(name)
type(name_list)
name_list.pop(0)
len(name_list)
name_list
len(name)


# %%
reload(vanillaRNN)
name
vanillaRNN._one_hot_word_tensor(name)

# %%
reload(vanillaRNN)
foo = np.array([-2.8523, -2.7800, -2.9394, -2.8962, -2.9287, -2.8165, -2.8406, -2.7723, -
                3.0290, -2.9533, -2.8288, -2.9262, -2.9352, -2.8949, -2.8554, -2.9956, -2.9283, -2.8957])
foo.shape
tt = torch.from_numpy(foo.reshape(1, 18))
tt

vanillaRNN.map_output_to_category(tt, categories)
# ======================================================== Done!
=======


# %%
>>>>>>> d5d88d6016ef850023d9d9cd4dcc8cd0a8350278
