import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, embedding_size=304, hidden_size=304):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)

    def forward(self, input, hidden):
        embedded = input
        outputs, hidden_state = self.gru(embedded.float(), hidden.float()) # [batch_size,frames,features];[1,batch_size,hidden_size]
        # print('output/hidden shape:', output.shape, hidden_state.shape)
        return outputs, hidden_state

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device) # the first hidden_state

class Decoder(nn.Module):
    def __init__(self, dict_size, embedding_size=304, hidden_size=304):
        super(Decoder, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, embedding_size) #(original) --> (originial,embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, dict_size) #hidden_size to dict_size; [batch_size, in_features]-->[batch_size, out_features]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden):
        '''
        decoder input.shape: torch.Size([1, 1, 1])
        decoder embedded.shape: torch.Size([1, 1, 1, 304])
        decoder embedded.shape: torch.Size([1, 1, 304])

        if in batch:
        encoder output/hidden shape: torch.Size([50, 371, 304]) torch.Size([1, 50, 304])
        decoder input.shape: torch.Size([50, 68, 1])
        decoder embedded.shape: torch.Size([50, 68, 1, 304])
        decoder embedded.shape: torch.Size([50, 68, 304])
        '''
        embedded = self.embedding(input)
        embedded = embedded.view(embedded.size(0), embedded.size(1), -1) # view as three dimension
        output, hidden_state = self.gru(embedded.float(), hidden.float())
        linear = self.linear(output)  # output.shape = [max_length, hidden_size], linear.shape = dict_size
        # print('linear.shape:', linear.shape) #linear.shape: torch.Size([batch_size, maxlen(sen)+seos+pad, len(dictionary)])
        softmax = self.softmax(linear)  ## softmax.shape=linear.shape=[batch_size, maxlen(sen)+seos+pad, len(dictionary)]
        return output, softmax, hidden_state

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device) # the first hidden_state #token 'sos' INDEX = 0

class AsrModel(nn.Module):
    def __init__(self, dict_size, max_length, sos, pad):
        super(AsrModel, self).__init__()

        self.dict_size = dict_size # == dict_size
        self.max_length = max_length
        self.encoder = Encoder() #however, input_size not been used in the encoder
        self.decoder = Decoder(self.dict_size)

        self.sos = sos #np.zeros((1, 1))
        self.pad = pad
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=int(self.pad[0][0]))
        # if resume: # if stoped and restart, then continue with the following model
        #     self.encoder.load_state_dict(torch.load("./models/encoder.ckpt"))
        #     self.decoder.load_state_dict(torch.load("./models/decoder.ckpt"))


    def forward(self, input, target, teacher_forcing: int = 1): #input.shape=(371, 10, 304)-->(10, 371, 304), target.shape=(18, 10, 1)-->(10, 18, 1)

        batch_size = input.shape[0]
        frame_num = input.shape[1]
        max_phn_num = target.shape[1]

        # Encoder
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(batch_size, frame_num, self.encoder.hidden_size, device=device)
        # [batch_size,frames,hidden_size]
        for isen in range(batch_size):
            for ifra in range(frame_num):
                # input.shape: torch.Size([50, 371, 304])
                # torch.tensor([[input[isen][ifra].tolist()]], device=device).shape==torch.Size([1, 1, 304])
                encoder_input = torch.tensor([[input[isen][ifra].tolist()]], device=device)
                encoder_output, encoder_hidden = self.encoder.forward(encoder_input, encoder_hidden) #hidden_state is the encoder result of a batch of sentences
                encoder_outputs[isen][ifra] = encoder_output[0, 0] # == [0],[0] [batch_size,frames,hidden_size]

        # Decoder
        decoder_hidden = encoder_hidden
        # self.dict_size: 63 phonemes
        # target.shape: torch.Size([50, 68, 1])
        # initial input == torch.Size([1, 1, 1]); 'sos' token
        decoder_input = torch.zeros(1, 1, target.shape[2], device=device).long()
        ## decoder_outputs = torch.zeros(batch_size, max_length, self.decoder.hidden_size, device=device)
        decoder_softmaxs = torch.zeros(batch_size, max_phn_num, self.decoder.dict_size, device=device)
        # [batch_size,max_length,hidden/dict]

        # Teacher forcing: Feed the target as the next input
        for isen in range(batch_size):
            for iphn in range(max_phn_num):
                decoder_output, decoder_softmax, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
                decoder_softmaxs[isen][iphn] = decoder_softmax[0, 0]  # [batch_size,max_length,dict]
                if iphn < (max_phn_num-1) and teacher_forcing == 1:
                    decoder_input = torch.tensor([[target[isen][iphn+1].tolist()]], device=device) #teacher forcing
                elif iphn < (max_phn_num-1) and teacher_forcing == 0:
                    decoder_input = torch.argmax(decoder_softmax, 2)[:, :, None]  # teacher forcing off


        # # Teacher forcing: Feed the target as the next input
        # for iphn in range(max_phn_num):
        #     decoder_output, decoder_softmax, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
        #     decoder_softmaxs[:,iphn,:] = decoder_softmax[0, 0]  # [batch_size,max_length,dict]
        #     if iphn < (max_phn_num-1):
        #         decoder_input = torch.tensor([[target[:,iphn+1,:].tolist()]], device=device) #teacher forcing


        # asr_outputs.shape = [batch_size, max_length, 1] ## softmax.shape = [batch_size, max_length, dict_size+tokens]
        asr_outputs = np.argmax(decoder_softmaxs.cpu().detach().numpy(), 2)[:, :, np.newaxis]
        softmax_cal, target_cal = decoder_softmaxs.view(-1, decoder_softmaxs.size(-1)), target.view(-1)

        return softmax_cal, target_cal, asr_outputs



    def test(self, input, target):
        batch_loss, outputs = 0, []
        # Encoder
        _, hidden_state = self.encoder.forward(input) #hidden_state is the encoder result of a batch of sentences

        # Decoder
        _, softmax, hidden_state = self.decoder.forward(target, hidden_state)

        # outputs = np.argmax(softmax.data.detach().cpu().numpy(), 2)[:, :, np.newaxis]
        outputs = np.argmax(softmax.cpu().detach().numpy(), 2)[:, :, np.newaxis]

        softmax, target = softmax.view(-1, softmax.size(-1)), target.view(-1)

        # outputs.shape = [batch_size, vectors, 1]
        # target.shape = [batch_size, vectors, 1]
        # softmax.shape = [batch_size, vectors, len(dictionary)]

        # loss softmax, target.shape: torch.Size([batch_size*vectors, len(dictionary)]); torch.Size([batch_size*vectors])
        batch_loss = self.criterion(softmax, target)

        batch_loss /= len(outputs)

        # return total_loss.data[0], outputs
        return batch_loss.data, outputs

    def eval(self, input):
        sentences = []
        input = input.to('cuda:0')
        hidden_state = self.encoder.first_hidden().to('cuda:0')
        # 1. Encoder
        for isentence in input: # cycle time: number of sentences in all the testing set
            _, hidden_state = self.encoder.forward(isentence, hidden_state)

        # 2. Decoder
        sentence = []
        All_sentences = input
        for isentence in All_sentences:
            input = Variable(torch.from_numpy(self.sos)).long().to('cuda:0') #np.zeros((1, 1))
            output, _, hidden_state = self.decoder.forward(isentence, hidden_state)

            # while input.data[0, 0] != 1:
            for i in range(self.max_length): # not to translate the fixed max_length:18,
                                               # but to translate the same length with the target translation not padded

                # print('rnn eval Decoder input.shape', ivec.shape)
                output, _, hidden_state = self.decoder.forward(input, hidden_state)
                # print('rnn eval Decoder output.shape', output.shape)
                # word = np.argmax(output.data.detach().cpu().numpy()).reshape((1, 1))
                word = np.argmax(output.data.numpy()).reshape((1, 1))
                # print('rnn eval Decoder word.shape', np.array(word).shape)

                input = Variable(torch.LongTensor(word).to('cuda:0'))
                # print('rnn eval Decoder next input.shape', input.shape)

                sentence.append(word)
                # print('rnn eval Decoder sentence.shape', np.array(sentence).shape)
            sentences.append(sentence)

        return sentences

    def save(self): # save model weights
        torch.save(self.encoder.state_dict(), "./models/encoder.ckpt")
        torch.save(self.decoder.state_dict(), "./models/decoder.ckpt")

    def load(self):
        self.encoder.load_state_dict(torch.load("./models/encoder.ckpt"))
        self.encoder.eval()
        self.decoder.load_state_dict(torch.load("./models/decoder.ckpt"))
        self.decoder.eval()