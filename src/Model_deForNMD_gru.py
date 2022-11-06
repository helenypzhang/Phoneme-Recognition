import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for comments: N for batch_size, M for max_len after pad, this seq2seq model is batch_first == True in gru

class EncoderGRU(nn.Module):
    def __init__(self, embedding_size=304, hidden_size=304):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)

    def forward(self, input):
        embedded = input
        outputs, hidden_state = self.gru(embedded.float())
        return outputs, hidden_state

    # def initHidden(self, batch_size):
    #     return torch.zeros(1, batch_size, self.hidden_size, device=device)

class DecoderGRU(nn.Module):
    def __init__(self, dict_size, embedding_size=304, hidden_size=304):
        super(DecoderGRU, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, embedding_size) #(original) --> (originial,embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, dict_size) #hidden_size to dict_size; [batch_size, in_features]->[batch_size, out_features]
        # self.softmax = nn.Softmax(dim=-1) #the last dimension will be used.

    def forward(self, input, hidden_state):
        embedded = self.embedding(input)
        embedded = embedded.view(embedded.size(0), embedded.size(1), -1) # view as three dimension
        output, hidden_state = self.gru(embedded.float(), hidden_state.float())
        linear = self.linear(output)  # output.shape = [max_length, hidden_size], linear.shape = dict_size
        # linear.shape = [batch_size, max_len, len(dictionary)]
        # softmax = self.softmax(linear)  ## softmax.shape=linear.shape=[batch_size, max_len, len(dictionary)]
        softmax = linear
        return output, softmax, hidden_state

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device) # first hidden_state 'sos' INDEX = 0 in batch

class AsrModel(nn.Module):
    def __init__(self, dict_size, max_length, sos, pad):
        super(AsrModel, self).__init__()

        self.dict_size = dict_size # == dict_size
        self.max_length = max_length
        self.encoder = EncoderGRU()
        self.decoder = DecoderGRU(self.dict_size)

        self.sos = sos #np.zeros((1, 1))
        self.pad = pad
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=int(self.pad[0][0]))

    def forward(self, input, target, teacher_forcing_ratio: float = 0.5): #input.shape=(N,frames,F), target.shape=(N,M,1)
        # print('target.shape:', target.shape) #[100, 75, 1]
        batch_size = input.shape[0]
        # frame_num = input.shape[1]
        max_phn_num = target.shape[1]

        # EncoderGRU
        encoder_outputs, encoder_hidden = self.encoder.forward(input) #hidden_state is encoder last hidden of a batch of sentences
        # encoder hidden.shape = [1, N, hiddensize]
        # print('1enouts:', encoder_outputs.shape)
        # print('2enhid:', encoder_hidden.shape)

        # DecoderGRU
        decoder_hidden = encoder_hidden
        # self.dict_size: 63 phonemes
        # first input to the decoder is the <sos> token
        decoder_input = target[:,0,:]
        decoder_softmaxs = torch.zeros(batch_size, max_phn_num, self.decoder.dict_size, device=device)
        # [N,M,hidden/dict]

        # Teacher forcing: Feed the target as the next input
        for iphn in range(max_phn_num):
            decoder_output, decoder_softmax, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
            # print('3deouts:',decoder_output.shape)
            # print('4desofmax:',decoder_softmax.shape)
            # print('5dehid:',decoder_hidden.shape)
            decoder_softmaxs[:,iphn,:] = decoder_softmax[:,0,:]  # [N,1,dict]
            teacher_forcing = random.random() < teacher_forcing_ratio # True or False; [0,1)<0 == False
            if iphn < (max_phn_num-1) and teacher_forcing == True:
                decoder_input = torch.tensor(target[:,iphn+1,:].tolist(), device=device) #teacher forcing
                decoder_input = torch.unsqueeze(decoder_input, 1)
            elif iphn < (max_phn_num-1) and teacher_forcing == False:
                decoder_input = torch.argmax(decoder_softmax, 2)[:, :, None]  # teacher forcing off

        # asr_outputs.shape = [N, M, 1] ## softmax.shape = [N, M, dict_size+tokens]
        asr_outputs = np.argmax(decoder_softmaxs[:,:-1,:].cpu().detach().numpy(), 2)[:, :, np.newaxis]
        softmax_cal, target_cal = decoder_softmaxs[:,:-1,:].reshape(-1, decoder_softmaxs.size(-1)), target[:,1:,:].reshape(-1)
        return softmax_cal, target_cal, asr_outputs

    # def save(self): # save model weights
    #     torch.save(self.encoder.state_dict(), "./models/encoder.ckpt")
    #     torch.save(self.decoder.state_dict(), "./models/decoder.ckpt")
    #
    # def load(self):
    #     self.encoder.load_state_dict(torch.load("./models/encoder.ckpt"))
    #     self.decoder.load_state_dict(torch.load("./models/decoder.ckpt"))