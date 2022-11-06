import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, embedding_size=304, hidden_size=304):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, 1, batch_first=False)

    def forward(self, input):
        embedded = input
        output, hidden_state = self.gru(embedded.float())
        # print('encoder outputs, hidden:',output.shape, hidden_state.shape)
        # batch_first = False
        # input = (L,N,H) output = (L,N,D); ht = (D,N,H); N:batch_size;
        # batch_first = True
        # input = (N,L,H) output = (N,L,D); ht = (D,N,H); N:batch_size;
        #outputs=torch.Size([371, 2, 304]) hidden_state = torch.Size([1, 2, 304]) batch_first = False
        return output, hidden_state

class Decoder(nn.Module):
    # input_size equals to dict_size
    def __init__(self, dict_size, embedding_size=304, hidden_size=304):
        super(Decoder, self).__init__()
        self.dict_size = dict_size
        self.embedding = nn.Embedding(self.dict_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1, batch_first=False)
        self.linear = nn.Linear(hidden_size, self.dict_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = embedded.view(embedded.size(0), embedded.size(1), -1) # many to three dim
        output, hidden_state = self.gru(embedded.float(), hidden.float())
        linear = self.linear(output)  # output.shape = [18, 500], linear.shape = decoder.dict_size
        softmax = self.softmax(linear) #softmax.shape=linear.shape=[maxlen,batch_size,len(dictionary)]; (1,batch_size, dict)
        return output, softmax, hidden_state #hidden_state.shape=(M,N,hidden)


class AsrModel(nn.Module):
    def __init__(self, dict_size, max_length, sos, pad):
        super(AsrModel, self).__init__()

        self.dict_size = dict_size # == dict_size(contains specific token)
        self.max_length = max_length
        self.encoder = Encoder() #however, input_size not been used in the encoder
        self.decoder = Decoder(self.dict_size)

        self.sos = sos #np.zeros((1, 1))
        self.pad = pad
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=int(self.pad[0][0]))  # 计算loss的时候不计算'pad' token

    def forward(self, input, target, teacher_forcing: int = 1): #input.shape=(371, N, 304); target.shape=(M, N, 1)

        batch_size = input.shape[1]
        max_len = target.shape[0]
        # Encoder
        encoder_outputs, encoder_hidden = self.encoder.forward(input) #hidden_state is the encoder result of a batch of sentences

        # Decoder 1
        # _, softmax, hidden_state = self.decoder.forward(target, encoder_hidden)
        # Decoder 2
        decoder_hidden = encoder_hidden
        decoder_input = torch.zeros(1, batch_size, target.shape[2], device=device).long() #sos token
        # decoder_input = torch.unsqueeze(target[0,:,:], 0)
        decoder_softmaxs = torch.zeros(max_len, batch_size, self.decoder.dict_size, device=device)

        # [max_length,batch_size,hidden/dict]

        # Teacher forcing: Feed the target as the next input
        for iphn in range(1, max_len):
            # decoder_input=1,N,1, decoder_hidden=1,N,D
            decoder_output, decoder_softmax, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
            # decoder_output=(1,N,D), decoder_softmax=(1,N,dict), decoder_hidden=(1,N,D)
            decoder_softmaxs[iphn] = decoder_softmax[0]  # [M,N,dict] :[37, 2, 63]
            if teacher_forcing == 1:
                # calculate loss
                # loss += self.criterion(torch.argmax(decoder_softmax, 2)[:, :, None], [target[iphn]])
                decoder_input = torch.tensor([target[iphn].tolist()], device=device) #teacher forcing on
            elif teacher_forcing == 0:
                # calculate loss
                decoder_input = torch.argmax(decoder_softmax, 2)[:, :, None] # teacher forcing off

        # loss.backward()
        outputs = np.argmax(decoder_softmaxs.cpu().detach().numpy(), 2)[:, :, np.newaxis] #outputs=(M,N,1)
        outputs = outputs.reshape(outputs.shape[1],outputs.shape[0],outputs.shape[2]) #outputs=(N,M,1)

        softmax_cal, target_cal = decoder_softmaxs.reshape(-1, decoder_softmaxs.size(-1)), target.reshape(-1)
        return softmax_cal, target_cal, outputs # softmax_cal=(M*N,dict);target_cal=(M*N);


    def save(self): # save model weights
        torch.save(self.encoder.state_dict(), "./models/encoder.ckpt")
        torch.save(self.decoder.state_dict(), "./models/decoder.ckpt")

    def load(self):
        self.encoder.load_state_dict(torch.load("./models/encoder.ckpt"))
        self.encoder.eval()
        self.decoder.load_state_dict(torch.load("./models/decoder.ckpt"))
        self.decoder.eval()