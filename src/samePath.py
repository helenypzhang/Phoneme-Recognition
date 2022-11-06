import librosa
import os
import time
import json
import math

import warnings
warnings.filterwarnings('ignore')

# 读取timit数据集下的所有.wav文件，.txt文件，.wrd文件，.phn文件
# 并依次返回全部wav文件路径，全部txt文件路径，全部wrd文件路径，全部phn文件路径        
def timit_files_path(dataset_dir, usage):
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
    dataset_usage = 'train'  # part 2
    inputDatasetPath=os.path.join("../datasets", inputDataset)
    print('Now begin to processing ', inputDataset,'_',dataset_usage)

    # part 3
    if (inputDataset=='timit'):
        wav_filesPath, txt_filesPath, wrd_filesPath, phn_filesPath=timit_files_path(inputDatasetPath, dataset_usage)
    elif (inputDataset=='qinghua'):
        wav_filesPath, trn_filesPath=qinghua_files_path(inputDatasetPath, dataset_usage)

    # number of wav files in dataset
    filenames = wav_filesPath
    print('all '+ dataset_usage +' wavPath list length: ',len(filenames))

    # 0.save the same path into .txt
    samePathTxt = open(os.path.join('../datasets', inputDataset + '_' + dataset_usage + '_samePath.txt'), 'w')
    filenames = wav_filesPath
    for filename in filenames:
        samePathTxt.write(os.path.splitext(filename)[0])   # 写入文件操作
        samePathTxt.write("\n")  # 换行
    samePathTxt.close()

    # # 1.save wav path into .txt
    # wavPathTxt = open(os.path.join('../features', inputDataset + '_' + dataset_usage + '_wavPath.txt'), 'w')
    # filenames = wav_filesPath
    # for filename in filenames:
    #     wavPathTxt.write("\n")   # 换行
    #     wavPathTxt.write(filename)   # 写入文件操作
    # wavPathTxt.close()

    # # 2.SAVE phn filepath into txt file
    # phnPathTxt = open(os.path.join('../features', inputDataset + '_' + dataset_usage + '_phnPath.txt'), 'w')
    # filenames = phn_filesPath
    # for filename in filenames:
    #     #save wav path into .txt
    #     phnPathTxt.write("\n")   # 换行
    #     phnPathTxt.write(filename)   # 写入文件操作
    # phnPathTxt.close()

    # # 3.SAVE txt filepath into txt file
    # txtPathTxt = open(os.path.join('../features', inputDataset + '_' + dataset_usage + '_txtPath.txt'), 'w')
    # filenames = txt_filesPath
    # for filename in filenames:
    #     #save wav path into .txt
    #     txtPathTxt.write("\n")   # 换行
    #     txtPathTxt.write(filename)   # 写入文件操作
    # txtPathTxt.close()

    # # ##DELETE THE ORIGINAL CSV FILE
    # csvPath = os.path.join('../features', inputDataset+'_'+ dataset_usage +'_feature_lpcs.csv')
    # if os.path.exists(csvPath):
    #     os.remove(csvPath)

if __name__ == '__main__':
    main()

