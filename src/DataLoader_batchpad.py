from collections import Counter
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
from utils import *

class AsrDataLoader(object):
    def __init__(self, input_path, output_path, max_length, batch_size, mode, train_targets_path, test_targets_path):
        super(AsrDataLoader, self).__init__()
        self.train_targets_path = train_targets_path
        self.test_targets_path = test_targets_path

        self.max_length = max_length
        self.input_vecs = np.array(read_json(input_path)).astype(np.long)

        self.output_dict, self.output_vecs, self.dict_size = self.init_language(mode)

        # output_size equals to len(dictionary)
        print("Languages found and loaded.")
        # self.output_vecs = self.filter(self.output_vecs) #if not used, the result could be higher

        self.batch_size = batch_size

        self.pad, self.sos, self.eos = np.zeros((1, 1)) + 3, np.zeros((1, 1)), np.zeros((1, 1)) + 1

    def init_language(self, mode):
        train_dictionary = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]

        train_corpus = read_sentences_txt(self.train_targets_path) #['s1', 's2'] == ['str(sentence1)', 'str(sentence2)'], s1 and s2 ... are not padded, so all the string sentences with not same length.
        test_corpus = read_sentences_txt(self.test_targets_path) #['s1', 's2'] == ['str(sentence1)', 'str(sentence2)'], s1 and s2 ... are not padded, so all the string sentences with not same length.
        train_words = " ".join(train_corpus).split()  #'s1 s2 s3 s4 s5'.split() == ['s1w1','s1w2', ... , 's5w1', 's5w2'...]
        # train_mc = Counter(train_words).most_common(self.vocab_size-3) #Counter type [top vocab_size words]
        train_mc = Counter(train_words).most_common()
        train_dictionary.extend([word for word, _ in train_mc]) # covert Counter to list type; dictionary is a list type
        if mode=='train':
            vectors = [[self.vectorize(word, train_dictionary) for word in sentence.split()] for sentence in train_corpus] #vectors means ID in the first version
        elif mode=='test':
            vectors = [[self.vectorize(word, train_dictionary) for word in sentence.split()] for sentence in test_corpus] #vectors means ID in the first version

        return train_dictionary, vectors, len(train_dictionary) #dictionary, vectors, len(dictionary) == self.output_dict, self.output_vecs, self.output_size

    def data_process(self):
        in_tensor_ = self.input_vecs
        out_tensor_ = self.output_vecs
        Alldata = []
        for (in_, out_) in zip(in_tensor_, out_tensor_):
            Alldata.append((in_, out_))
        return Alldata #return a list[(in, out)]

    def generate_batch(self, data_batch):
        in_batch, out_batch = [], []
        for (in_item, out_item) in data_batch:
            in_batch.append(in_item)
            out_item = torch.tensor(out_item, dtype=torch.long)
            out_batch.append(torch.cat([torch.tensor([self.sos]) , out_item , torch.tensor([self.eos])], dim = 0))
        # print('out_batch.shape testing after sos:', np.array(out_batch[0]).shape) #[batch_size, vetorsNoPad,1,1]
        max_out_len = max([sentence.shape[0] for sentence in out_batch])
        # print('max words in batch:', max_out_len) #include <sos> <eos>
        # for sentence in out_batch:
        #     print('len(sentence) in batch:', sentence.shape[0]) ##not padded yet
        out_batch = torch.tensor([sentence.tolist() + [self.pad for _ in range(max_out_len - sentence.shape[0])] for sentence in out_batch])
        in_batch = torch.tensor(in_batch, dtype=float)
        out_batch = torch.from_numpy(np.squeeze(out_batch.numpy(), 3)) #out_batch.shape #[batch_size, vectors,1]
        print('out_batch.shape testing after pad and squeeze:', out_batch.shape) #[batch_size,maxlen(sen)+seos+pad,1]

        return in_batch, out_batch

    def get_iter(self):
        train_data = self.data_process()
        data_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn= self.generate_batch)
        return data_iter

    def vec_to_sentence(self, vectors): #(18, 1, 1)
        dict = self.output_dict
        # print('vectors.shape:', vectors.shape) #(maxlen(sen)+seos+pad,1,1)
        sentence = " ".join([dict[int(vec[0])] for vec in vectors]) #translation.shape: (20, 1, 1)
        return sentence 

    def vectorize(self, word, list): # convert one word --> vector/ID using given list/dictionary
        #vec = torch.LongTensor(1, 1).zero_()
        vec = np.zeros((1, 1), dtype=np.int32)
        index = list.index("<UNK>") if word not in list else list.index(word)
        vec[0][0] = index
        return vec


    def filter(self, output_vecs):
        i = 0
        for _ in output_vecs:
            if len(output_vecs[i]) > self.max_length:
                output_vecs.pop(i)
            else:
                i += 1

        return output_vecs