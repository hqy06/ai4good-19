# %%
import sys
import vanillaRNN
from importlib import reload
import string


# %%
ALL_LETTERS = string.ascii_letters + " .,;'"
PATH = '..\\datasets\\names'


# %%
path = PATH
files = vanillaRNN.list_by_extension(path)
lang_name_dict = vanillaRNN.create_lang_name_dict(files, path)
categories = list(lang_name_dict.keys())
n_categories = len(categories)
n_char = len(ALL_LETTERS)


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


# %%
