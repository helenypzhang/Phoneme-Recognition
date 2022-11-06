import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for comments: N for batch_size, M for max_len after pad, this seq2seq model is batch_first == True in gru

class EncoderGRU(nn.Module):
    def __init__(self, dict_size, frame_len, max_len, embedding_size=39, encoder_hidden_size=100, decoder_hidden_size = 100):
        super(EncoderGRU, self).__init__()
        self.hidden_size = encoder_hidden_size
        self.gru = nn.GRU(embedding_size, encoder_hidden_size, num_layers=1, batch_first=False, dropout=0.5, bidirectional=True)
        self.fc_hidden = nn.Linear(encoder_hidden_size*2, decoder_hidden_size)
        self.fc_outputs = nn.Linear(encoder_hidden_size*2, dict_size)
        self.fc_frame2maxlen = nn.Linear(frame_len, max_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        '''
        output (seq_len, batch, hidden_size * num_directions)
        h_n (num_layers * num_directions, batch, hidden_size)
        '''
        embedded = self.dropout(input)
        outputs, hidden_state = self.gru(embedded.float())
        # print('encoder hidd1.shape:', hidden_state.shape)
        hidden_state = torch.tanh(self.fc_hidden(torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)))
        hidden_state = hidden_state.unsqueeze(0)

        ctc_outputs = self.fc_outputs(outputs)
        # torch.cat((outputs[-1, :, :hidden_size], outputs[0, :, hidden_size:]),dim = 1)
        encoder_outputs = self.fc_frame2maxlen(outputs.transpose(0,2))
        encoder_outputs = encoder_outputs.transpose(0,2)

        # print('encoder hidd2.shape:', hidden_state.shape)
        # print('encoder outputs.shape:', outputs.shape)
        return ctc_outputs, encoder_outputs, hidden_state

    # def initHidden(self, batch_size):
    #     return torch.zeros(1, batch_size, self.hidden_size, device=device)

class DecoderGRU(nn.Module):
    def __init__(self, dict_size, embedding_size=200, hidden_size=100):
        super(DecoderGRU, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=1, batch_first=False, dropout=0.5)
        self.linear = nn.Linear(hidden_size, dict_size) #hidden_size to dict_size; [batch_size, in_features]->[batch_size, out_features]
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, hidden_state):
        embedded = self.dropout(input)
        # embedded = self.embedding(input)
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
    def __init__(self, dict_size, frame_length, max_length, sos, pad):
        super(AsrModel, self).__init__()

        self.dict_size = dict_size # == dict_size
        self.frame_length = frame_length
        self.max_length = max_length
        self.encoder = EncoderGRU(self.dict_size, self.frame_length, self.max_length)
        self.decoder = DecoderGRU(self.dict_size)

        self.sos = sos #np.zeros((1, 1))
        self.pad = pad
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=int(self.pad[0][0]))

    def forward(self, input, target, teacher_forcing_ratio: float = 0.5): #input.shape=(N,frames,F), target.shape=(N,M,1)

        batch_size = input.shape[1]
        frame_num = input.shape[0]
        max_phn_num = target.shape[0]

        # EncoderGRU
        encoder_ctc_outputs, encoder_outputs, encoder_hidden = self.encoder.forward(input)
        #hidden_state is encoder last hidden of a batch of sentences
        # encoder hidden.shape = [1, N, hiddensize]

        # DecoderGRU
        decoder_hidden = encoder_hidden # torch.Size([1, 30, 100])
        # self.dict_size: 63 phonemes
        decoder_input = encoder_outputs #torch.Size([585, 30, 200])

        decoder_outputs, decoder_softmaxs, _ = self.decoder.forward(decoder_input, decoder_hidden)
        #torch.Size([75, 30, 100]);([75, 30, 65])
        # print('decoder softmaxs.shape:', decoder_outputs.shape, decoder_softmaxs.shape)
        # asr_outputs.shape = [N, M, 1] ## softmax.shape = [N, M, dict_size+tokens]
        asr_outputs = np.argmax(decoder_softmaxs.cpu().detach().numpy(), 2)[:, :, np.newaxis]
        softmax_cal, target_cal = decoder_softmaxs.reshape(-1, decoder_softmaxs.size(-1)), target.reshape(-1)
        ctc_targets = target.transpose(0,1) #[N,M,1]

        # print('cross loss input shape:', softmax_cal.shape, target_cal.shape, asr_outputs.shape, encoder_ctc_outputs.shape, ctc_targets.shape)

        #torch.Size([2250, 65]) torch.Size([2250]) (75, 30, 1) torch.Size([585, 30, 65]) torch.Size([30, 75, 1])
        return softmax_cal, target_cal, asr_outputs, encoder_ctc_outputs, ctc_targets
