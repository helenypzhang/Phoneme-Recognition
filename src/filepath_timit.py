import librosa
import os
import time
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features
import pandas as pd
from scipy.signal import wavelets
import json
import math
import re
from collections import Counter
import string

feature_dim = 13
window_length = 0.025
window_hop = 0.01
wav_rate = 16000

# dataloader for timit dataset
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# %matplotlib inline
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

# 读取timit数据集下的所有.wav文件，.txt文件，.wrd文件，.phn文件
# 并依次返回全部wav文件路径，全部txt文件路径，全部wrd文件路径，全部phn文件路径

## complete train set:3696 sentences; 8 sentences * 462 speakers; 3.14 hours data; YES
## complete test set: 1344 sentences; 8 sentences * 168 speakers; 0.81 hours data; YES
def timit_complete_train_OR_test_path(dataset_dir, usage):
    files_wav = []
    files_txt = []
    files_wrd = []
    files_phn = []
    for setsFolder in os.listdir(dataset_dir):
        # if (setsFolder == "test") or (setsFolder == "train"):
        if (setsFolder == usage):
            setsFolderPath = os.path.join(dataset_dir, setsFolder)
            print(setsFolderPath)
            for dialect in os.listdir(setsFolderPath):
                dialectPath = os.path.join(setsFolderPath, dialect)
                for speaker in os.listdir(dialectPath):
                    speakerPath = os.path.join(dialectPath, speaker)
                    for sentence in os.listdir(speakerPath):
                        if sentence.find('si') != -1 or sentence.find('sx') != -1: #reture -1 means don't find
                            sentencePath = os.path.join(speakerPath, sentence)
                            if os.path.splitext(sentencePath)[1] == '.wav':
                                files_wav.append(sentencePath)
                                # print('wav',sentencePath)
                            if os.path.splitext(sentencePath)[1] == '.txt':
                                files_txt.append(sentencePath)
                                # print('txt',sentencePath)
                            if os.path.splitext(sentencePath)[1] == '.wrd':
                                files_wrd.append(sentencePath)
                                # print('wrd',sentencePath)
                            if os.path.splitext(sentencePath)[1] == '.phn':
                                files_phn.append(sentencePath)
                                # print('phn',sentencePath)
    # 返回train和test中全部wav, txt, wrd, phn文件路径
    return files_wav, files_txt, files_wrd, files_phn

## core test set: 192 sentences; 8 sentences * 24 speakers; 0.16 hours data; YES
def timit_coretest_path(dataset_dir, usage):
    files_wav = []
    files_txt = []
    files_wrd = []
    files_phn = []
    # coretest = ['DAB0', 'WBT0', 'ELC0', 'TAS1', 'WEW0', 'PAS0', 'JMP0', 'LNT0', 'PKT0',
    #             'LLL0', 'TLS0', 'JLM0', 'BPM0', 'KLT0', 'NLP0', 'CMJ0', 'JDH0', 'MGD0',
    #             'GRT0', 'NJM0', 'DHC0', 'JLN0', 'PAM0', 'MLD0']
    coretest = ['mdab0', 'mwbt0', 'felc0', 'mtas1', 'mwew0', 'fpas0', 'mjmp0', 'mlnt0', 'fpkt0',
                'mlll0', 'mtls0', 'fjlm0', 'mbpm0', 'mklt0', 'fnlp0', 'mcmj0', 'mjdh0', 'fmgd0',
                'mgrt0', 'mnjm0', 'fdhc0', 'mjln0', 'mpam0', 'fmld0']
    for setsFolder in os.listdir(dataset_dir):
        # if (setsFolder == "test") or (setsFolder == "train"):
        if (setsFolder == usage):
            setsFolderPath = os.path.join(dataset_dir, setsFolder)
            print(setsFolderPath)
            for dialect in os.listdir(setsFolderPath):
                dialectPath = os.path.join(setsFolderPath, dialect)
                count = 0
                for speaker in os.listdir(dialectPath):
                    # print(speaker,count)
                    if speaker in coretest:
                        count = count + 1
                        print(speaker, count)
                        speakerPath = os.path.join(dialectPath, speaker)
                        for sentence in os.listdir(speakerPath):
                            if sentence.find('si') != -1 or sentence.find('sx') != -1:
                                sentencePath = os.path.join(speakerPath, sentence)
                                if os.path.splitext(sentencePath)[1] == '.wav':
                                    files_wav.append(sentencePath)
                                    # print('wav',sentencePath)
                                if os.path.splitext(sentencePath)[1] == '.txt':
                                    files_txt.append(sentencePath)
                                    # print('txt',sentencePath)
                                if os.path.splitext(sentencePath)[1] == '.wrd':
                                    files_wrd.append(sentencePath)
                                    # print('wrd',sentencePath)
                                if os.path.splitext(sentencePath)[1] == '.phn':
                                    files_phn.append(sentencePath)
                                    # print('phn',sentencePath)
    # 返回train和test中全部wav, txt, wrd, phn文件路径 
    return files_wav, files_txt, files_wrd, files_phn


## whole train: complete + 2 SA dialect sentences: 4620 sentences; YES
## whole test: complete + 2 SA dialect sentences: 1680 sentences; YES
def timit_train_OR_test_path(dataset_dir, usage):
    files_wav = []
    files_txt = []
    files_wrd = []
    files_phn = []
    for setsFolder in os.listdir(dataset_dir):
        # if (setsFolder == "test") or (setsFolder == "train"):
        if (setsFolder == usage):
            setsFolderPath = os.path.join(dataset_dir, setsFolder)
            print(setsFolderPath)
            for dialect in os.listdir(setsFolderPath):
                dialectPath = os.path.join(setsFolderPath, dialect)
                for speaker in os.listdir(dialectPath):
                    speakerPath = os.path.join(dialectPath, speaker)
                    for sentence in os.listdir(speakerPath):
                        sentencePath = os.path.join(speakerPath, sentence)
                        if os.path.splitext(sentencePath)[1] == '.wav':
                            files_wav.append(sentencePath)
                            # print('wav',sentencePath)
                        if os.path.splitext(sentencePath)[1] == '.txt':
                            files_txt.append(sentencePath)
                            # print('txt',sentencePath)
                        if os.path.splitext(sentencePath)[1] == '.wrd':
                            files_wrd.append(sentencePath)
                            # print('wrd',sentencePath)
                        if os.path.splitext(sentencePath)[1] == '.phn':
                            files_phn.append(sentencePath)
                            # print('phn',sentencePath)
    # 返回train和test中全部wav, txt, wrd, phn文件路径
    return files_wav, files_txt, files_wrd, files_phn

# 读取timit数据集下的所有.wav文件，.txt文件，.wrd文件，.phn文件
# 并依次返回全部wav文件路径，全部txt文件路径，全部wrd文件路径，全部phn文件路径        
def qinghua_files_path(dataset_dir, usage):
    files_wav = []
    files_trn = []
    for level1Folder in os.listdir(dataset_dir):
        # if (setsFolder == "test") or (setsFolder == "train"):
        if (level1Folder == "data_thchs30"):
            level1FolderPath = os.path.join(dataset_dir, level1Folder)
            print(level1FolderPath)
            for level2Folder in os.listdir(level1FolderPath):
                if (level2Folder == usage):
                    level2FolderPath = os.path.join(level1FolderPath, level2Folder)
                    print(level2FolderPath)
                    for sentence in os.listdir(level2FolderPath):
                        sentencePath = os.path.join(level2FolderPath, sentence)
                        if os.path.splitext(sentencePath)[1] == '.wav':
                            files_wav.append(sentencePath)
                            # print('wav', sentencePath)
                        if os.path.splitext(sentencePath)[1] == '.trn':
                            files_trn.append(sentencePath)
                            # print('trn',sentencePath)
    # 返回train和test中全部wav, txt, wrd, phn文件路径 
    return files_wav, files_trn

def main():
    # ## BEGIN TO PROCESS DATASET
    # ## 1.for timit dataset 2. for qinghua dataset
    # ## you need to change two parts according to your goals
    inputDataset='timit'   # part 1
    dataset_usage = 'test'  # part 2 'train' or 'test' 
    inputDatasetPath=os.path.join("../datasets", inputDataset)
    print('Now begin to processing ', inputDataset,'_',dataset_usage)

    # part 3
    if (inputDataset=='timit'): #change the = timit_path() function as you need!
        wav_filesPath, txt_filesPath, wrd_filesPath, phn_filesPath= timit_train_OR_test_path(inputDatasetPath, dataset_usage)
    elif (inputDataset=='qinghua'):
        wav_filesPath, trn_filesPath= qinghua_files_path(inputDatasetPath, dataset_usage)

    filenames = wav_filesPath
    print('all '+ dataset_usage +' wavPath list length: ',len(filenames))

    #开始计时
    start_time=time.time()

    # part 4
    # change the saved path title as you need! '_whole_'; or '_complete_'; or '_core_' + dataset_usage=='test'
    samePathTxt = open(os.path.join('../datasets', inputDataset+'_whole_'+ dataset_usage +'_samePath.txt'), 'w')  #在py文件存储文件夹 生成一个文件名为SoundPath的文件


    # # ##DELETE THE ORIGINAL CSV FILE
    # csvPath = os.path.join('../features', inputDataset+'_'+ dataset_usage +'_feature_lpcs.csv')
    # if os.path.exists(csvPath):
    #     os.remove(csvPath)

    for filename in filenames:
        # print('wav file:',filename)
        samePath = os.path.splitext(filename)[0]
        # filename = samePath + '.txt'
        # print('now processing:', filename)
        ####### 1. save path file
        samePathTxt.write(samePath)
        samePathTxt.write("\n")
    samePathTxt.close()

    #结束计时
    end_time=time.time()
    print("程序运行时长",str(end_time-start_time))
    #######################################################################3

if __name__ == '__main__':
    main()
