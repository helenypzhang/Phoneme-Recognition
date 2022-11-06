import librosa
import os
import time
import json
import math
import numpy as np
import string
import re

feature_dim = 13
window_length = 0.025
window_hop = 0.01
wav_rate = 16000

import warnings
warnings.filterwarnings('ignore')

##########################################################################

## 保存json格式的文件
## save data to json file
def store(data, jsonfilePath):
    # with open('data.json','w') as fw:
    with open( jsonfilePath, 'w') as fw:
        #将numpy数组array转化成json格式
        # json_str = json.dumps(data)
        # fw.write(json_str)
        #上面两句等同于下面这句
        json.dump(data, fw)

##读取json格式的文件
##load json data from file
def load(jsonfilePath):
    # with open('data.json', 'r') as f:
    with open(jsonfilePath, 'r') as f:
        data = json.load(f)
        return data

#读取某文件夹下的所有.wav文件，并返回文件全称
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                L.append(os.path.join(root, file))
        return L

def get_phonemes_noh(inputDataset, dataset_usage, filenames):
    phonemesTxt = open(os.path.join('../targets', inputDataset + '_whole_' + dataset_usage + '_phonemes_noh.txt'), 'w')
    for filename in filenames:
        print('now processing:', filename)
        with open(filename) as f:
            eachphn = []
            lines = f.readlines()
            lines = lines[1:-1] #delete the begin and the end token #h
            for line in lines: #each line has one phn
                ## 1.去掉数字和 前面的两个空格
                line = re.sub(r'[0-9]', '', line)
                line = line[2:-1]
                ## 2.去掉英文标点符号
                punctuation_string = string.punctuation  # 列出所有的英文标点符号
                for i in punctuation_string:
                    line = line.replace(i, '')  # 把文本中出现的英文标点符号替换成nothing
                phonemesTxt.write(line+' ')
        f.close()
        phonemesTxt.write("\n")
    # samePathTxt.close()
    phonemesTxt.close()

def get_phonemes(inputDataset, dataset_usage, filenames):
    phonemesTxt = open(os.path.join('../targets', inputDataset + '_whole_' + dataset_usage + '_phonemes.txt'), 'w')
    for filename in filenames:
        print('now processing:', filename)
        with open(filename) as f:
            eachphn = []
            lines = f.readlines()
            # lines = lines[1:-1] #delete the begin and the end token #h
            for line in lines: #each line has one phn
                ## 1.去掉数字和 前面的两个空格
                line = re.sub(r'[0-9]', '', line)
                line = line[2:-1]
                # ## 2.去掉英文标点符号
                # punctuation_string = string.punctuation  # 列出所有的英文标点符号
                # for i in punctuation_string:
                #     line = line.replace(i, '')  # 把文本中出现的英文标点符号替换成nothing
                phonemesTxt.write(line+' ')
        f.close()
        phonemesTxt.write("\n")
    # samePathTxt.close()
    phonemesTxt.close()

def get_sentences_dict(inputDataset, dataset_usage, filenames):
    sentences = []
    words = []
    sentencesTxt = open(os.path.join('../targets', inputDataset+'_whole_'+ dataset_usage +'_sentences.txt'), 'w')
    for filename in filenames:
        print('now processing:', filename)
        with open(filename) as f:
            sentence = f.read()
            ## 1.去掉数字 句前空格 句末标点
            sentence = re.sub(r'[0-9]', '', sentence)
            sentence = sentence[2:-2]
            ## 2.去掉英文标点符号
            punctuation_string = string.punctuation  # 列出所有的英文标点符号
            for i in punctuation_string:
                sentence = sentence.replace(i, '')  # 把文本中出现的英文标点符号替换成nothing
            ##sentence2wordsList
            sentences.append(sentence.split())  # sentence1,2,3

            # for word in sentence.split():
            #     words.append(word)  # word1,2,3

        sentencesTxt.write(sentence)
        sentencesTxt.write("\n")

    # print(sentences)
    # print(words)
    sentencesTxt.close()

def main():
    # ## BEGIN TO PROCESS DATASET
    # ## 1.for timit dataset 2. for qinghua dataset
    # ## you need to change two parts according to your goals
    inputDataset='timit'   # part 1
    dataset_usage = 'train'  # part 2
    inputDatasetPath=os.path.join("../datasets", inputDataset)
    print('Now begin to processing ', inputDataset,'_',dataset_usage)

    wav_filesPath = []
    txt_filesPath = []
    wrd_filesPath = []
    phn_filesPath = []
    samePathTxt = open(os.path.join('../datasets', inputDataset + '_whole_' + dataset_usage + '_samePath.txt'), 'r')
    Lines = samePathTxt.readlines()
    for line in Lines:
        # part 1

        # wavPath = line[:-1]+'.wav'
        # wav_filesPath.append(wavPath)

        # txtPath = line[:-1]+'.txt'
        # txt_filesPath.append(txtPath)

        # wrd_filesPath = line[:-1]+'.wrd'

        phnPath = line[:-1]+'.phn'
        phn_filesPath.append(phnPath)

    # number of wav files in dataset
    # part 2
    filenames = phn_filesPath
    print('all '+ dataset_usage +' Path list length: ',len(filenames))

    #开始计时
    start_time=time.time()

    # part 3
    # #得到所有句子文本
    # get_sentences_dict(inputDataset, dataset_usage, filenames)

    # part 3
    #得到所有音素文本
    get_phonemes_noh(inputDataset, dataset_usage, filenames)

    #结束计时
    end_time=time.time()
    print("程序运行时长",str(end_time-start_time))
    #######################################################################3

if __name__ == '__main__':
    main()

