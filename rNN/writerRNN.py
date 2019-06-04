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
1. prepare data
2. define network structure
3. load data: batch, encoding, xx + yy
4. train
5. evaluate
6. predict
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
import torch                     # PyTorch
from collections import Counter  # building vocabulary
from datetime import datetime    # for timestamp
import torch.nn as nn            # neural network
import numpy as np               # how do you do? i do math

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


def clean_text(raw_text):
    # replace "--" with a single space
    text = raw_text.replace('--', ' ')
    text = text.translate(str.maketrans({key: " {0} ".format(
        key) for key in (string.whitespace + string.punctuation)}))
    text = re.sub(" +", " ", text)
    text = re.split(r'( +)', text)
    tokens = [t.casefold() for t in text]
    while tokens[0] == ' ' or tokens[0] == '':
        tokens = tokens[1:]
    while tokens[-1] == ' ' or tokens[-1] == '':
        tokens = tokens[:-2]
    assert (len(tokens) > 0)
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
    int_word = {i: word for i, word in enumerate(sorted_vocab)}
    word_int = {word: i for i, word in enumerate(sorted_vocab)}
    return word_int, int_word


def select_book(file_list):
    print("Select which file to use:")
    for i in range(len(file_list)):
        print("\t {:>3} | {}".format(i, file_list[i]))
    file_no = input("Text file chosed: ")
    return file_list[int(file_no)]


def save_to_file_ascii(some_text, name):
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
        # hidden states `hidden` and memory state `memory` for LSTM layers
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

# ==========================================================================
# Data loading
# ==========================================================================


def load_batches(text, encoder_dict, seq_size, batch_size, verbose=False):
    encoded = [encoder_dict[word] for word in text]
    n_word = len(encoder_dict)
    n_batch = int(np.floor(len(encoded) / (seq_size * batch_size)))
    assert (n_batch > 0), "batch_size too large!"
    if verbose:
        print("{} words organized in sequences of {}.\n  * Divide into {} batches, chop out last {} words.".format(
            len(encoded), seq_size, n_batch, len(text) - int(n_batch * seq_size * batch_size)))

    encoded_in = encoded[:int((n_batch * seq_size * batch_size))]
    encoded_out = encoded_in[1:]
    encoded_out.append(encoded_in[0])
    if verbose:
        print("  * encoded    :", len(encoded),
              encoded[:5], encoded[1465:1472])
        print("  * encoded in :", len(encoded_in),
              encoded_in[:5], encoded_in[-5:])
        print("  * encoded out:", len(encoded_out),
              encoded_out[:5], encoded_out[-5:])

    batch_in = np.reshape(encoded_in, (batch_size, -1))
    batch_out = np.reshape(encoded_out, (batch_size, -1))

    xx, yy = [], []
    for i in range(0, n_batch * seq_size, seq_size):
        xx.append(batch_in[:, i:i + seq_size])
        yy.append(batch_out[:, i:i + seq_size])

    if verbose:
        print(
            "in/out batches: {} in total, each batch of shape {}".format(len(xx), xx[0].shape))

    return xx, yy

# ==========================================================================
# Train, Evaluate & Predict
# ==========================================================================


def fit_model(writer_rnn, encoder, criterion, lr, optimizer, batch_size, in_batches, out_batches, n_epoch=3, verbose=False, save_model=False, print_every=1, save_every=2):
    iter_counter = 0
    total_loss = 0
    for epoch in range(n_epoch):
        epoch += 1
        hidden, memory = writer_rnn.init_zeros(batch_size)
        hidden = hidden.to(device)
        memory = memory.to(device)
        epoch_loss, iter_loss, iter_counter = train(writer_rnn, iter_counter, x, y, criterion,
                         lr, optimizer, verbose=verbose)

        if verbose and epoch % print_every == 0:
            print('Epoch {:>3}/{} | itr={:8}\tepoch_loss={:.5f} iter_loss={:.5f}'.format(epoch, n_epoch, iter_counter, epoch_loss, iter_loss)))

        if save_model and epoch % save_every == 0:
            # save the model (checkpoints!)
            torch.save(writer_rnn.state_dict(), 'checkpoints/{}-writer-{}.pth'.format(
                datetime.now().strftime('%Y%m%d-%H%M'), iter_counter))

        total_loss += epoch_loss

    return total_loss


def train(writer_rnn, iter_counter, batch_in, batch_out, batch_out, criterion, lr, optimizer, verbose = verbose):
    hid_state, mem_state=writer_rnn.init_zeros(batch_size)
    hid_state=hid_state.to(device)
    mem_state=mem_state.to(device)
    train_loss=0

    for x, y in zip(batch_in, batch_out):
        x=torch.tensor(x).to(device)
        y=torch.tensor(y).to(device)
        writer_rnn.train()  # a convention learned from PyTorch doc
        optimizer.zero_grad()   # prepare optimizer

        logits, states=writer_rnn(x, (hid_state, mem_state))
        (hid_state, mem_state)=states

        loss=criterion(logits.transpose(1, 2), y)
        train_loss += loss.item()
        loss.backward()

        # NOTE: detach states - a convention, but why?
        hid_state=hid_state.detach()
        mem_state=mem_state.detach()

        # to prevent vanishing and/or exploding gradients
        nn.utils.clip_grad_norm_(writer_rnn.parameters(), 5)
        # optimizer!
        optimizer.step()
        iter_counter += 1

    return epoch_loss, loss.item(), iter_counter


def predict(writer_rnn, device, encoder, decoder, text_seed, text_len, top_k = 5):
    writer_rnn.eval()  # convention before prediction/evaluation?
    hid_state, mem_state=writer_rnn.init_zeros(1)
    hid_state=hid_state.to(device)
    mem_state=mem_state.to(device)
    text=text_seed
    for w in seed_text:
        in=torch.tensor([[encoder[w]]]).to(device)
        out, (hid_state, mem_state)=writer_rnn(in, (hid_state, mem_state))

    _, top_xs=torch.topk(out[0], k = top_k)
    choice=np.random.choice((topxs.tolist())[0])

    text.append(decoder[choice])

    for i in range(text_len):
        in=torch.tensor([[choice]]).to(device)
        out, (hid_state, mem_state)=writer_rnn(in, (hid_state, mem_state))
        _, top_xs=torch.topk(out[0], k = top_k)
        choice=np.random.choice((topxs.tolist())[0])
        text.append(choice)

    return text


# ==========================================================================
# Main
# ==========================================================================


def main(phase, verbose = False):
    path=PATH
    char_set=ALL_CHARS
    files=list_by_extension(path)

    if phase == 1:
        # use the small test file
        file_name='666-temp-by-foo.txt'
        text, encoder, decoer=fetch_data(file_name)
        # save the cleaned result into a text file
        save_to_file_ascii(text, 'cleaned_text')
        exit()

    if verbose:
        print("\n========== 1. Preparing Data ==========")

    file_name=select_book(files)
    text, encoder, decoder=fetch_data(
        file_name, folder_path = path, verbose = verbose)
    save_to_file_ascii(text, 'cleaned_{}'.format(file_name))
    assert (len(encoder) == len(decoder))
    n_vocab=len(encoder)

    if verbose:
        print("\n========== 2. Create Model ==========")

    seq_size=32
    embedding_size=64
    lstm_size=64
    rnn=writerRNN(n_vocab, seq_size, embedding_size, lstm_size)

    criterion=nn.CrossEntropyLoss()
    lr=0.001   # learning rate
    optimizer=torch.optim.Adam(rnn.parameters(), lr = lr)

    if verbose:
        print("Model created\n  * {} words in sequence of {}.\n  * Embedding layer of size={}\n  * LSTM layer of size={}".format(
            n_vocab, seq_size, embedding_size, lstm_size))
        print("Hyperparameters\n  * criterion=CrossEntropyLoss\n  * learning rate={}, Adam optimizer".format(lr))

    if verbose:
        print("\n========== 3. Load Data ==========")
    batch_size=16

    if phase == 3:
        # list of in/out batches
        xx, yy=load_batches(text, encoder, seq_size,
                              batch_size, verbose = True)
        exit()

    xx, yy=load_batches(text, encoder, seq_size,
                          batch_size, verbose=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnn = rnn.to(device)

    if verbose:
        print("\n========== 4. Train the model ==========")
        print("Current device: {}". format(device.type))

    fit_model(rnn, encoder, criterion, lr, optimizer, batch_size, in_batches, out_batches, n_epoch=3, verbose=True, save_model=False, print_every=1, save_every=2)

    seed = ['summer', ' ']
    text_len = 100

    result = predict(rnn, device, encoder, decoder, seed, text_len, top_k = 5)
    print(''.join(result))

    return 0


if __name__ == '__main__':
    phase = input('key in phase number: ')
    main(int(phase), verbose=True)
