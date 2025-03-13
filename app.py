import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
import streamlit as st
import random
import nltk
import re
import unicodedata
import pyarabic.araby as araby
import contractions
import os

# Load necessary NLTK data
os.environ['NLTK_DATA'] = './punkt'

# Define the model architecture first
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# Load the vocabularies
with open("src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("trg_vocab.pkl", "rb") as f:
    trg_vocab = pickle.load(f)

# Initialize model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)
EMB_DIM = 512
HID_DIM = 1024
DROPOUT = 0.3

attn = Attention(HID_DIM, HID_DIM)
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, DROPOUT).to(device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, DROPOUT, attn).to(device)

# Initialize the Seq2Seq model
model = Seq2Seq(enc, dec, device).to(device)

# Load model weights (state_dict)
model.load_state_dict(torch.load('model.pt', map_location=device))

# Set the model to evaluation mode
model.eval()

# Define helper functions for tokenization and preprocessing
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

nltk.download('punkt')

def tokenize_ar(text):
    return [tok for tok in nltk.tokenize.wordpunct_tokenize(unicodeToAscii(text))]

def tokenize_en(text):
    return [tok for tok in nltk.tokenize.wordpunct_tokenize(unicodeToAscii(text))]

def preprocess(sequence, vocab, src=True):
    if src:
        tokens = tokenize_ar(sequence.lower())
    else:
        tokens = tokenize_en(sequence.lower())

    sequence = []
    sequence.append(vocab[''])
    sequence.extend([vocab[token] for token in tokens])
    sequence.append(vocab[''])
    sequence = torch.Tensor(sequence)
    return sequence

# Streamlit Interface
st.set_page_config(page_title="Arabic to English Translation", page_icon=":guardsman:", layout="wide")

# Header
st.title("Arabic to English Translation Model")
st.subheader("Enter Arabic text below to get its English translation")

# User input for translation
input_text = st.text_area("Enter Arabic Text for Translation:", "السماء زرقاء")

# Process the input text for translation
if st.button("Translate"):
    input_tensor = preprocess(input_text, src_vocab)
    input_tensor = input_tensor[:, None].to(torch.int64).to(device)

    # Initialize the target tensor with a maximum possible length for translation
    target_length = len(input_text.split()) + 3
    target_tensor = torch.zeros(target_length, 1).to(torch.int64)

    # Make the prediction
    with torch.no_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        output = model(input_tensor, target_tensor, 0)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)

    prediction = [torch.argmax(i).item() for i in output]
    tokens = trg_vocab.lookup_tokens(prediction)
    translation = TreebankWordDetokenizer().detokenize(tokens).replace('', "").replace('"', "").strip()

    st.subheader("Translation Result:")
    st.write(translation)

# Footer
st.markdown("""
    ---
    Built with Streamlit and PyTorch.
    [GitHub Repository](https://github.com/your-repository)
""")
