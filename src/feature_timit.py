import librosa
import os
import time
import json
import math

feature_dim = 13
window_length = 0.025
window_hop = 0.01
wav_rate = 16000

import warnings
warnings.filterwarnings('ignore')

def preprocessing(filename):
    # 数据准备
    wavedata, fs = librosa.load(filename, sr= wav_rate)
    sample_rate = fs
    signal = wavedata
    # 预加重
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 分帧 framing
    frame_size, frame_stride = window_length, window_hop
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
    frames = pad_signal[indices]
    # print(frames.shape) #(843, 400)

    # 加窗 window
    hamming = np.hamming(frame_length)
    # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))

    frames *= hamming
    # print(frames.shape) #(843, 400)

    return frames

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

'''
1. 调用python_speech_features包
'''
def mfcc_1(filename):
    # fs, wavedata = wav.read(filename) ## doesn't work sometimes, we use librosa instead
    wavedata, fs = librosa.load(filename, sr=wav_rate)
    # mfcc_feature = python_speech_features.mfcc(wavedata, fs, winlen=0.025, winstep=0.01, nfilt=13, nfft=1024)  # mfcc系数     # nfilt为返回的mfcc数据维数，默认为13维
    mfcc_feature = python_speech_features.mfcc(wavedata, winlen=window_length, winstep=window_hop, numcep=feature_dim, samplerate=wav_rate)
    d_mfcc_feat = python_speech_features.base.delta(mfcc_feature, 1)     # feat 为mfcc数据或fbank数据    # N - N为1代表一阶差分，N为2代表二阶差分     # 返回：一个大小为特征数量的numpy数组，包含有delta特征，每一行都有一个delta向量
    d_mfcc_feat2 = python_speech_features.base.delta(mfcc_feature, 2)
    mfccs = np.hstack((mfcc_feature, d_mfcc_feat, d_mfcc_feat2))
    # print(mfccs.shape)
    # 返回 帧数*39 的mfccs参数
    return mfccs

def fbank_1(filename):   #fbank只是缺少mfcc特征提取的dct倒谱环节，其他步骤相同。actually we use logfbank as fbank
    # fs, wavedata = wav.read(filename) ## doesn't work sometimes, we use librosa instead
    wavedata, fs = librosa.load(filename, sr=wav_rate)
    fbank_feature = python_speech_features.logfbank(wavedata, winlen=window_length, winstep=window_hop, nfilt=feature_dim, samplerate= wav_rate)
    # logfbank_feature = python_speech_features.logfbank(wavedata, fs, winlen=0.064, winstep=0.032, nfilt=13, nfft=1024)  # mfcc系数     # nfilt为返回的mfcc数据维数，默认为13维
    d_fbank_feat = python_speech_features.base.delta(fbank_feature, 1)     # feat 为mfcc数据或fbank数据    # N - N为1代表一阶差分，N为2代表二阶差分     # 返回：一个大小为特征数量的numpy数组，包含有delta特征，每一行都有一个delta向量
    d_fbank_feat2 = python_speech_features.base.delta(fbank_feature, 2)
    fbanks = np.hstack((fbank_feature, d_fbank_feat, d_fbank_feat2))
    # print(fbanks.shape)
    # 返回 帧数*39 的fbanks参数
    return fbanks

def energy_1(filename):   #fbank只是缺少mfcc特征提取的dct倒谱环节，其他步骤相同。actually we use logfbank as fbank
    # fs, wavedata = wav.read(filename) ## doesn't work sometimes, we use librosa instead
    wavedata, fs = librosa.load(filename, sr=wav_rate)
    fbank_feature, energy_feature = python_speech_features.fbank(wavedata, winlen=window_length, winstep=window_hop, nfilt=feature_dim, samplerate= wav_rate)
    # logfbank_feature = python_speech_features.logfbank(wavedata, fs, winlen=0.064, winstep=0.032, nfilt=13, nfft=1024)  # mfcc系数     # nfilt为返回的mfcc数据维数，默认为13维
    log_energy = np.log(energy_feature)
    # print('log_energy shape:', log_energy.shape)
    log_energy = log_energy.reshape([log_energy.shape[0], 1])
    # print('reshape log_energy shape:', log_energy.shape)
    # 返回 帧数*1 的log_energy参数
    return log_energy

def mfcc_energy_delta(filename, grad = 1):
    # fs, wavedata = wav.read(filename) ## doesn't work sometimes, we use librosa instead
    wavedata, fs = librosa.load(filename, sr=wav_rate)
    # mfcc_feature = python_speech_features.mfcc(wavedata, fs, winlen=0.025, winstep=0.01, nfilt=13, nfft=1024)  # mfcc系数     # nfilt为返回的mfcc数据维数，默认为13维
    mfcc_feature = python_speech_features.mfcc(wavedata, winlen=window_length, winstep=window_hop, numcep=feature_dim, samplerate=wav_rate)
    fbank_feature, energy_feature = python_speech_features.fbank(wavedata, winlen=window_length, winstep=window_hop, nfilt=feature_dim, samplerate=wav_rate)

    #calc log energy
    log_energy = np.log(energy_feature) #np.log( np.sum(feat_raw**2, axis=1) )
    log_energy = log_energy.reshape([log_energy.shape[0],1])

    #z-score standardization
    mat = ( mfcc_feature - np.mean(mfcc_feature, axis=0) ) / (0.5 * np.std(mfcc_feature, axis=0))
    mat = np.concatenate((mat, log_energy), axis=1)

    mfcc_energy = mat

    #calc first order derivatives
    if grad == 1:
        gradf = python_speech_features.base.delta(mfcc_energy, 1) 
        mfcc_energy = np.concatenate((mfcc_energy, gradf), axis=1) #np.concatenate((mfcc_energy, gradf), axis=1) == np.hstack((mfcc_energy, grad))

    #calc second order derivatives
    if grad == 2:
        gradf = python_speech_features.base.delta(mfcc_energy, 1) # N - N为1代表一阶差分，N为2代表二阶差分; 一个大小为特征数量的numpy数组，包含有delta特征，每一行都有一个delta向量
        grad2f = python_speech_features.base.delta(mfcc_energy, 2)
        mfcc_energy = np.hstack((mfcc_energy, gradf, grad2f)) #np.concatenate((mfcc_energy, grad2f), axis=1) == np.hstack((mfcc_energy, gradf, grad2f))

    print(mfccs_energy.shape)
    # 返回 帧数*42 的mfccs参数
    return mfcc_energy

import numpy as np
# from scikits.audiolab import Sndfile
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@the best performance
import python_speech_features as sf

# ref part0: change the grad as you need!
def mfcc_energy_gradient(filename, numcep=12, numfilt=26, winlen=window_length, winstep=window_hop, grad=1):  
# def cal_features(filename, numcep=feature_dim, numfilt=feature_dim, winlen=window_length, winstep=window_hop, grad=1):
    wavedata, fs = librosa.load(filename, sr=wav_rate)
    #calc mfcc
    feat_raw,energy = sf.fbank(wavedata, fs, winlen,winstep, nfilt=numfilt)
    feat = np.log(feat_raw)
    feat = sf.dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = sf.lifter(feat,L=22)
    feat = np.asarray(feat)

    #calc log energy
    log_energy = np.log(energy) #np.log( np.sum(feat_raw**2, axis=1) )
    log_energy = log_energy.reshape([log_energy.shape[0],1])

    #z-score standardization
    mat = ( feat - np.mean(feat, axis=0) ) / (0.5 * np.std(feat, axis=0))
    mat = np.concatenate((mat, log_energy), axis=1)

    #calc first order derivatives
    if grad >= 1:
        gradf = np.gradient(mat)[0]
        mat = np.concatenate((mat, gradf), axis=1)

    #calc second order derivatives
    if grad == 2:
        grad2f = np.gradient(gradf)[0]
        mat = np.concatenate((mat, grad2f), axis=1)

    # print('mfcc+energy+3 shape:', mat.shape) 26,28, 26 is better

    return mat

def melspectrogram(filename):
    y, sr = librosa.load(filename, sr=wav_rate)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = 128)
    melspec = np.array(melspec)#转成numpy类型
    print('melspec.shape:', melspec.shape)
    # 返回 帧数*128 的melspectrogram参数
    return melspec.T

def log_powspec(frames):
    # fs, wavedata = wav.read(filename) ## doesn't work sometimes, we use librosa instead
    # mfcc_feature = python_speech_features.mfcc(wavedata, fs, winlen=0.025, winstep=0.01, nfilt=13, nfft=1024)  # mfcc系数     # nfilt为返回的mfcc数据维数，默认为13维
    logpowspec = python_speech_features.sigproc.logpowspec(frames, NFFT=512) #512->257; 1024->513
    logpowspec = np.array(logpowspec)
    # print('logpowspec shape is:', logpowspec.shape)
    # 返回 帧数*257 的mfccs参数
    return logpowspec
# 从一个音频信号中计算梅尔滤波器能量特征 返回：2个值。
# 第一个是一个包含着特征的大小为nfilt的numpy数组，每一行都有一个特征向量。
# 第二个返回值是每一帧的能量
# '''
# 2. 调用librosa包,只用lpc_2
# '''
# def mfcc_2(filename):
#     y, sr = librosa.load(filename, sr=None)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39, )  # n_mfcc为返回的mfcc数据维度
#     mfccs=np.array(mfccs)#转成numpy类型
#     # 返回 帧数*39 的mfccs参数
#     return mfccs.T

# def fbank_2(filename):
#     y, sr = librosa.load(filename, sr=None)
#     my_melspec = librosa.feature.melspectrogram(y=y, sr=sr)  # n_mfcc为返回的mfcc数据维度
#     my_logmelspec = librosa.power_to_db(my_melspec)
#     fbanks = my_logmelspec
#     fbanks=np.array(fbanks)#转成numpy类型
#     # 返回 帧数*39 的fbanks参数
#     return fbanks.T

def lpc_2(y):
    # print(sr) #获得线性预测系数。要指定模型顺序，请使用一般规则，即该顺序是预期共振峰数量的两倍加 
    # A = lpc(x1,8); Obtain the linear prediction coefficients. 
    # To specify the model order, use the general rule that the order is two times the expected number of formants plus 
    # 2. In the frequency range, [0,|Fs|/2], you expect three formants. Therefore, set the model order equal to 8. 
    my_lpc = librosa.lpc(y=y, order=feature_dim)  # order为返回的lpc数据维度
    # lpcs=np.array(my_lpc) #转成numpy类型
    lpcs = my_lpc
    # 返回 帧数*(13+1) 的lpcs参数
    lpcs = lpcs.T
    lpcs_final = []
    lpcs_final = lpcs[1:]
    return lpcs_final

'''
3. 手写代码 lpcc
'''
import librosa
import python_speech_features
# import soundfile as sf
import numpy as np
# lpc_3 is only used for lpcc_3
# lpc_3 can not be used to extract lpc feature for each wav audio file
def lpc_3(y, order):
    dtype = y.dtype.type
    ar_coeffs = np.zeros(order + 1, dtype=dtype)
    ar_coeffs[0] = dtype(1) # 1.0
    ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)
    # 前向和后向的预测误差
    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]
    den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(bwd_pred_error, bwd_pred_error)
    for i in range(order):
        if den <= 0:
            raise FloatingPointError("numerical error, input ill-conditioned?")
        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i+2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]
        # 前向预测误差和后向预测误差更新
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp
        q = dtype(1) - reflect_coeff ** 2
        den = q * den - bwd_pred_error[-1]**2 - fwd_pred_error[0]**2
        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]
    return ar_coeffs

def lpcc_3(y):
    # 得到lpc系数
    lpc_coeff = lpc_3(y=y, order = feature_dim+1)
    lpc_order = feature_dim+1
    # lpcc 系数个数
    lpcc_order = feature_dim+1
    lpcc_coeff = np.zeros(lpcc_order)
    lpcc_coeff[0] = lpc_coeff[0]
    for m in range(1, lpc_order):
        lpcc_coeff[m] = lpc_coeff[m]
        for k in range(0,m):
            lpcc_coeff[m] = lpc_coeff[m] + lpcc_coeff[k] * lpc_coeff[m - k] * k / m
    for m in range(lpc_order, lpcc_order):
        for k in range(m - lpc_order, m):
            lpcc_coeff[m] = lpcc_coeff[m] + lpcc_coeff[k] * lpc_coeff[m - k] * k / m
    # lpcc_coeff=np.array(lpcc_coeff)
    # lpc_coeff=np.array(lpc_coeff)
    # 返回 帧数*(13+1) 的lpcs参数
    lpcc_coeff_final=[]
    lpcc_coeff_final=lpcc_coeff[1:]
    return lpcc_coeff_final


'''
5. 调用pywt dwt wpt
'''
import pywt

def dwt_5 (y):
    '''
    Function: lpcc
    Summary: Computes the linear predictive cepstral compoents. Note: Returned values are in the frequency domain. LPCC is computed through LPC.
    '''
    # my_dwt = pywt.dwt(y, 'db1')
    my_dwt = pywt.downcoef(part = 'a', data = y, wavelet = 'db1')
    # dwts=np.array(my_dwt) #转成numpy类型
    dwts = my_dwt
    # 返回 帧数*(13) 的dwts参数
    return dwts

def get_Feature_demo(filename):

    demof_mfcc1=mfcc_1(filename)

    # print('demo mfcc_1 type:', type(demof_mfcc1)) 
    # print('demo mfcc_1 shape:', demof_mfcc1.shape)
    # print(demof_mfcc1[0,0])
    # #save data into json file demos
    # json_data = demof_mfcc1.tolist()
    # json_file_Path = os.path.join('../features', 'demomfcc1.json')
    # store(json_data,json_file_Path)
    # print('succeed save mfcc1 json file!')
    # # save mfcc1 feature
    # df = pd.DataFrame(demof_mfcc1)
    # df.to_csv(os.path.join('../features','demo_mfcc1.csv'), index=False, header=False, mode='a')
    # print('succeed save mfcc1 csv file!')
    # ## mfcc_1 shape: (843, 39)



    demof_fbank1=fbank_1(filename)

    # print('demo fbank_1 type:', type(demof_fbank1))
    # print('demo fbank_1 shape:', demof_fbank1.shape)
    # print(demof_fbank1[0,0])
    # #save data into json file demos
    # json_data = demof_fbank1.tolist()
    # json_file_Path = os.path.join('../features', 'demofbank1.json')
    # store(json_data,json_file_Path)
    # print('succeed save fbank1 json file!')
    # # save fbank1 feature
    # df = pd.DataFrame(demof_fbank1)
    # df.to_csv(os.path.join('../features','demo_fbank1.csv'), index=False, header=False, mode='a')
    # print('succeed save fbank1 csv file!')
    # ## fbank_1 shape: (843, 39)

    demof_energy1 = energy_1(filename)

    demof_mfcc_energy1 = mfcc_energy_1(filename)

    demof_mfcc_energy3 = cal_features(filename)

    my_frames = preprocessing(filename)
    demof_logpowspec = log_powspec(my_frames)

    demof_lpc2 = []
    demof_lpcc3 = []
    demof_dwt5 = []
    for my_frame in my_frames:
        frame_lpc2=lpc_2(my_frame)
        frame_lpc2 = np.array(frame_lpc2)
        # print('frame lpc_2 type:', type(frame_lpc2))
        # print('frame lpc_2 shape:', frame_lpc2.shape)
        # print(frame_lpc2)
        demof_lpc2.append(frame_lpc2)

        frame_lpcc3=lpcc_3(my_frame)
        frame_lpcc3 = np.array(frame_lpcc3)
        # print('frame lpcc_3 type:', type(frame_lpcc3))
        # print('frame lpcc_3 shape:', frame_lpcc3.shape)
        # print(frame_lpcc3)
        demof_lpcc3.append(frame_lpcc3)

        frame_dwt5=dwt_5(my_frame)
        frame_dwt5 = np.array(frame_dwt5)
        # print('frame dwt_5 type:', type(frame_dwt5))
        # print('frame dwt_5 shape:', frame_dwt5.shape)
        # print(frame_dwt5)
        demof_dwt5.append(frame_dwt5)
        
    demof_lpc2 = np.array(demof_lpc2)    
    # print(demof_lpc2.shape)
    # # save lpc2 feature
    # df = pd.DataFrame(demof_lpc2)
    # df.to_csv(os.path.join('../features','demo_lpc2.csv'), index=False, header=False, mode='a')
    # print('succeed save lpc2 csv file!')
    # ## lpc_2.shape (843, 13)

    demof_lpcc3 = np.array(demof_lpcc3) 
    # # save lpcc3 feature
    # df = pd.DataFrame(demof_lpcc3)
    # df.to_csv(os.path.join('../features','demo_lpcc3.csv'), index=False, header=False, mode='a')
    # print('succeed save lpcc3 csv file!')
    # print(demof_lpcc3.shape)
    # ## lpcc_3.shape (843, 13)


    demof_dwt5 = np.array(demof_dwt5)   
    # # save dwt5 feature
    # df = pd.DataFrame(demof_dwt5)
    # df.to_csv(os.path.join('../features','demo_dwt5.csv'), index=False, header=False, mode='a')
    # print('succeed save dwt5 csv file!')
    # print(demof_dwt5.shape)
    # ## dwt_5.shape(843, 200)



    # feature_combined = np.hstack((demof_mfcc1, demof_fbank1, demof_lpc2, demof_lpcc3, demof_dwt5))
    # feature_combined = np.hstack((demof_mfcc1, demof_lpcc3))
    # feature_combined = np.hstack((demof_fbank1, demof_lpc2))
    # feature_combined = demof_fbank1
    # feature_combined = demof_lpc2
    # feature_combined = np.hstack((demof_mfcc1, demof_logpowspec))
    # feature_combined = demof_logpowspec
    feature_combined = demof_mfcc_energy3 #ref
    # feature_combined = demof_mfcc_energy1 #yp

    # # save total features
    # df = pd.DataFrame(feature_combined)
    # df.to_csv(os.path.join('../features','feature_combined.csv'), index=False, header=False, mode='a')
    # print('succeed save feature_combined csv file!')
    # print(feature_combined.shape)
    # ## total feature_combined.shape (843, 304)
    # ## 304 = 39+39+13+13+200
    # ##不做以下这一步就可以确保column是一样的长度
    # # feature_combined_oneD = feature_combined.reshape(1,-1)

    return feature_combined

def get_Feature_Fusion(filename):

    part1_mfcc=mfcc_1(filename)
    part2_energy=energy_1(filename)
    feature_combined = np.hstack((part1_mfcc, part2_energy))
    
    # feature_combined = np.hstack((demof_mfcc1, demof_fbank1, demof_lpc2, demof_lpcc3, demof_dwt5))
    # feature_combined = np.hstack((demof_mfcc1, demof_lpcc3))
    # feature_combined = np.hstack((demof_fbank1, demof_lpc2))
    # feature_combined = demof_fbank1

    # feature_combined = mfcc_energy_gradient(filename) #ref
    # feature_combined = mfcc_energy_delta(filename) #YP

    return feature_combined

def main():
    # ## BEGIN TO PROCESS DATASET
    # ## 1.for timit dataset 2. for qinghua dataset
    # ## you need to change two parts according to your goals
    inputDataset='timit'   # part 1
    dataset_usage = 'train'  # part 2
    inputDatasetPath=os.path.join("../datasets", inputDataset)
    print('Now begin to processing ', inputDataset, dataset_usage)

    wav_filesPath = []
    txt_filesPath = []
    wrd_filesPath = []
    phn_filesPath = []

    # part3: change read samePath file name according to the data you need!
    samePathTxt = open(os.path.join('../datasets', inputDataset + '_complete_' + dataset_usage + '_samePath.txt'), 'r')
    Lines = samePathTxt.readlines()
    for line in Lines:
        wavPath = line[:-1]+'.wav'  ##as each line end with ("\n")
        # wavPath = line+'.wav'
        print(wavPath)
        wav_filesPath.append(wavPath)
        # txt_filesPath = line+'.txt'
        # wrd_filesPath = line+'.wrd'
        # phn_filesPath = line+'.phn'

    # number of wav files in dataset
    filenames = wav_filesPath
    print('all '+ dataset_usage +' wavPath list length: ',len(filenames))

    #开始计时
    start_time=time.time()

    #循环读取文件名称并提取所有的features
    # wav_index = 0
    total_features = []
    wavs_framesNum = []
    for filename in filenames:
        # print('now processing:',filename)
        #提取所有特征
        wav_features = get_Feature_Fusion(filename) #get wav_features.shape == ( frames,(39+39+13+13+200) )=(frames, 304)

        # print('wav feature shape:', wav_features.shape)
        wav_framesNum = wav_features.shape[0]
        # print('wav frames number', wav_framesNum)
        total_features.append(wav_features)
        #统计每条语音的帧数
        wavs_framesNum.append(wav_framesNum) #保存每条语音都有多少帧

    filenames = wav_filesPath
    print('successfully get all features(not sized) in ', inputDataset, '_', dataset_usage)
    total_features = np.array(total_features)
    print('total dataset feature shape:', total_features.shape)

    #wav_features 每条语音的所有features frames长度不一样
    #total_features 所有语音的所有features frames长度不一样
    #wavs_framesNum 所有语音的帧数集合

    # 按帧数排序后，找出前80%的语音数据中的最长帧数，这个帧数即为需要保留存储到json文件的帧数维度
    # 按帧数排序后，找出’全部‘的语音数据中的最长帧数，这个帧数即为需要保留存储到json文件的帧数维度
    # total_features.append 之前需要截取一下，padding一下，再保存

    # wavs_framesNum = np.array(wavs_framesNum)
    #
    # wavs_framesNum_max = np.max(wavs_framesNum)
    # print('max:', wavs_framesNum_max)
    # wavs_framesNum_min = np.min(wavs_framesNum)
    # print('min:', wavs_framesNum_min)
    # print('mean:', np.mean(wavs_framesNum))
    # wavs_framesNum_sorted = sorted(wavs_framesNum) #排序 ordered from small to big 简单升序排列，从小到大
    # threshood = math.ceil(len(filenames) * 0.995)  # 80%个数据是指第多少条语音
    # frameSize = int(wavs_framesNum_sorted[threshood]) #排序后，第90%个数据有多少帧
    # print('99.5%frameSize:', frameSize)


    frameSize = 585 #change the frameSize as same as the frameSize in the timit train set
    # frameSize = wavs_framesNum_max
    # frameSize = 778
    # print('Max frameSize:', frameSize)
    #frameSize 最后保留的每条语音数据的帧数dww sceeeeeeeeeeeee


######################################################################

#### pad for the frames in each sentence
    total_features_sized = []
    for frames_features in total_features:
        if(frames_features.shape[0] > frameSize):
            frames_features_keep = []
            frames_features = frames_features.tolist()
            frames_features_keep = frames_features[:frameSize]
            frames_features_keep = np.array(frames_features_keep)
        else:
            frames_features_keep = []
            frames_features_keep = np.array(frames_features_keep)
            paddingNum = frameSize-frames_features.shape[0]
            frames_features_keep = np.pad(frames_features, ((0, paddingNum), (0, 0)), 'constant', constant_values=0)

        total_features_sized.append(frames_features_keep)
    total_features_sized = np.array(total_features_sized)
    print('the final resized wavs(dataset) features shape',total_features_sized.shape)

    #save data into json file demos
    json_data = total_features_sized.tolist()
    # json_file_Path = os.path.join('../features', 'total_features_sized.json')
    # json_file_Path = os.path.join('../features', inputDataset+'_'+ dataset_usage +'_features.json')

    # part4: change feature saved name according to your need!
    json_file_Path = os.path.join('../features', inputDataset + '_complete_' + dataset_usage + '_mfcc_energy_585_40.json')
    store(json_data,json_file_Path)
    print('succeed save '+ inputDataset+'_'+ dataset_usage +'_features json file!')

    #结束计时
    end_time=time.time()
    print("程序运行时长",str(end_time-start_time))
    #######################################################################3

if __name__ == '__main__':
    main()

