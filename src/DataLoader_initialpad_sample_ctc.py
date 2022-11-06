from collections import Counter
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
from utils import *

class AsrDataLoader(object):
    def __init__(self, input_path_train, input_path_test, batch_size, targets_path_train, targets_path_test):
        super(AsrDataLoader, self).__init__()
        self.targets_path_train = targets_path_train
        self.targets_path_test = targets_path_test

        self.input_vecs_train = np.array(read_json(input_path_train)).astype(np.float)
        self.input_vecs_test = np.array(read_json(input_path_test)).astype(np.float)

        self.batch_size = batch_size
        self.pad, self.sos, self.eos = np.zeros((1, 1)) + 3, np.zeros((1, 1)), np.zeros((1, 1)) + 1

        self.output_vecs_train, self.output_vecs_test, self.output_dict, self.dict_size, self.max_length = self.init_language()
        # delete longer than max sentences
        self.input_vecs_train, self.output_vecs_train = self.filter(self.input_vecs_train, self.output_vecs_train) #pop longer
        self.input_vecs_test, self.output_vecs_test = self.filter(self.input_vecs_test, self.output_vecs_test)  # pop longer

        self.frame_length = np.array(self.input_vecs_train).shape[1] #[N,Frame,Feature]
        print('Frames lenth of wav:', self.frame_length)
        # print('check numbers for padded speech and target', len(self.input_vecs_train), np.array(self.output_vecs_train,dtype = np.long).shape,
        #       len(self.input_vecs_test), np.array(self.output_vecs_test, dtype= np.long).shape)

        print("Languages found and loaded.")

    def init_language(self):
        dictionary_train = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]
        corpus_train = read_sentences_txt(self.targets_path_train) #['s1', 's2'] == ['str(sentence1)', 'str(sentence2)']
        corpus_test = read_sentences_txt(self.targets_path_test) #['s1', 's2'] == ['str(sentence1)', 'str(sentence2)']
        words_train = " ".join(corpus_train).split()  #'s1 s2 s3 s4 s5'.split() == ['s1w1','s1w2', ... , 's5w1', 's5w2'...]
        # mc_train = Counter(words_train).most_common(self.vocab_size-4) #CounterType[set numbers of top vocab_size words]
        mc_train = Counter(words_train).most_common() # all of the words will be counted from many to one.
        # print(mc_train)
        dictionary_train.extend([word for word, _ in mc_train]) # covert Counter to list type; dictionary is a list type
        dictionary_train.extend(" ")

        vectors_train = [[self.vectorize(word, dictionary_train) for word in sentence.split()] for sentence in corpus_train]
        vectors_test = [[self.vectorize(word, dictionary_train) for word in sentence.split()] for sentence in corpus_test]

        seos_vectors_train = [self.addseos(sentence) for sentence in vectors_train]
        seos_vectors_test = [self.addseos(sentence) for sentence in vectors_test]

        max_out_len = int(max([len(sentence) for sentence in seos_vectors_train]) * 1)
        print('max_length of the whole dataset:', max_out_len)

        vectors_train = [sentence + [self.pad for _ in range(max_out_len - len(sentence))] for sentence in seos_vectors_train]
        vectors_test = [sentence + [self.pad for _ in range(max_out_len - len(sentence))] for sentence in seos_vectors_test]

        return vectors_train, vectors_test, dictionary_train, len(dictionary_train), max_out_len

    def vectorize(self, word, list): # convert one word --> vector/ID using given list/dictionary
        #vec = torch.LongTensor(1, 1).zero_()
        vec = np.zeros((1, 1), dtype=np.int32)
        index = list.index("<UNK>") if word not in list else list.index(word)
        vec[0][0] = index
        return vec

    def addseos(self, sentence):
        sentence.insert(0,self.sos)
        sentence.append(self.eos)
        return sentence

    def filter(self, input_vecs, output_vecs): #[N,frame,D], [N, phn, 1]
        i = 0
        input_vecs = list(input_vecs)
        for _ in output_vecs:
            if len(output_vecs[i]) > self.max_length:
                output_vecs.pop(i) #throw this sentence target.
                input_vecs.pop(i) #throw this sentence speech.
            else:
                i += 1
        return input_vecs, output_vecs

    def data_process(self, in_tensor_, out_tensor_):
        Alldata = []
        for (in_, out_) in zip(in_tensor_, out_tensor_):
            Alldata.append((in_, out_))
        return Alldata  # [(in, out)]

    def generate_batch(self, data_batch):
        in_batch, out_batch = [], []
        for (in_item, out_item) in data_batch:
            in_batch.append(in_item)
            out_batch.append(out_item)
        # out_batch.shape after append tokens：[batch_size, vetorsPad,1,1]
        in_batch = torch.tensor(in_batch, dtype=torch.float)
        out_batch = torch.tensor(out_batch, dtype=torch.long)
        out_batch = torch.squeeze(out_batch, 3)  # [batch_size, vetorsPad,1]
        return in_batch, out_batch

    def get_iter(self):
        data_train = self.data_process(self.input_vecs_train, self.output_vecs_train)
        data_test = self.data_process(self.input_vecs_test, self.output_vecs_test)
        data_iter_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=False, collate_fn= self.generate_batch)
        data_iter_test = DataLoader(data_test, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        return data_iter_train, data_iter_test

    def vec_to_sentence(self, vectors): #(L, 1, 1)
        dict = self.output_dict
        # print('vectors.shape:', vectors.shape) #(maxlen(sen)+seos+pad,1,1)
        sentence = " ".join([dict[int(vec[0])] for vec in vectors]) #translation.shape: (L, 1, 1)
        return sentence

