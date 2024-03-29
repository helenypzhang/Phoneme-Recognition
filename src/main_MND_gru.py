from numpy.core.defchararray import index
from numpy.lib.function_base import average
from numpy.lib.shape_base import expand_dims
from DataLoader_initialpad import *
# from Model_deForMND_bigru_att import *
# from Model_deForMND_gru_att import *
from Model_deForMND_gru import *

from utils import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
# from torchmetrics import WER #for 0.6 version
from torchmetrics import WordErrorRate # WER updated as WordErrorRate 0.6-0.9 version change
from nltk.translate.bleu_score import sentence_bleu #nltk.translate.bleu_score.sentence_bleu()

# For demo dataset:
# features demo file is 'timit_train_features_demo.json'
# targets demo file is 'timit_train_phonemes_demo.txt'

features_path_train = '../features/timit_complete_train_mfcc_energy_gradient1_585_26.json' #all features with fixed size
targets_path_train = '../targets/timit_complete_train_phonemes_noh.txt' #all translations not padded

features_path_test = '../features/timit_core_test_mfcc_energy_gradient1_585_26.json'
targets_path_test = '../targets/timit_core_test_phonemes_noh.txt'

num_epochs = 300 # number of times training the whole train dataset
# batch_size = 64 # for sentenses
batch_size = 100 # for phonemes 30
initial_lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# shape of features: (4620, 371, 304) [num_sentences, num_frames, num_featureDim]
# shape of gt.txt: (4620, 18, 1) [num_sentences, num_words, num_featureDim]

def evaluate(model, asrdata):
    model.eval()
    iterator_train, iterator_test = asrdata.get_iter()
    criterion = nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))
    sosINDEX = 0
    eosINDEX = 1
    unkINDEX = 2
    padINDEX = 3
    epoch_loss = 0
    test_predictions = []
    test_truths = []

    with torch.no_grad():
        for i, data in enumerate(iterator_test):
            input, target = data
            softmax_cal, target_cal, outputs = model(input.float().transpose(0,1).to(device),
                                                     target.long().transpose(0,1).to(device),
                                                     teacher_forcing_ratio = 0)
            # model input:[100, 371, 304]; model target:[100, 68, 1]; model output:(100, 67, 1)
            loss = criterion(softmax_cal, target_cal)
            epoch_loss += loss.item()
            print("Test Loss at batch %d: %.2f" % (i, loss.item()))
            # Evaluation Metrics prepare
            token = target[:,1:,:]
            for j in range(token.shape[0]): # j == batch_size
                # token.shape = (batch_size,len(sentens),1), new token[i].shape =(18, 1)-->(18, 1, 1)
                token_clean = []
                outputs_clean = []
                token_show = []
                outputs_show = []
                for n in range(token[j].shape[0]): ## token[j] is a sentence # token[j][n] is a number, refer to the word id
                    if token[j][n] != padINDEX and token[j][n] != sosINDEX and token[j][n] != eosINDEX:
                        token_show.append([token[j][n].item()])
                        outputs_show.append(outputs[n][j])
                        if token[j][n] != unkINDEX:
                            token_clean.append([token[j][n].item()])
                            outputs_clean.append(outputs[n][j])
                # get all truth and predictions, except pad, sos, eos, show unk if any
                truth_show = asrdata.vec_to_sentence(np.expand_dims(token_show, axis = 2))
                prediction_show = asrdata.vec_to_sentence(np.expand_dims(outputs_show, axis=2))

                # get all truth and predictions, except pad, sos, eos, unk
                truth = asrdata.vec_to_sentence(np.expand_dims(token_clean, axis = 2))
                prediction = asrdata.vec_to_sentence(np.expand_dims(outputs_clean, axis=2))
                test_truths.append(truth)
                test_predictions.append(prediction)
                # for show, show the unk token
                if (i % 100 == 0):
                    print('The sentence', j,' is:')
                    print("Truth: \"%s\"" % truth_show) #str
                    print("Guess: \"%s\"\n" % prediction_show) #string # outputs.shape = [batch_size, 18, 1], str
        epoch_loss /= len(iterator_test)

    # Evaluation Metrics
    # 1. calculate the WER score using torchmetrics
    metric = WordErrorRate()
    score_wer = metric(test_predictions, test_truths)  # predictions, references
    # print('---------------------------------WER score of testing is: ', score_wer.data)
    # 2. calculate the BLEU score using nltk
    sum = 0
    num = 0
    for i in range(len(test_truths)):  # [0-(length(test_targets)-1)]
        test_prediction = test_predictions[i].split()
        test_truth = test_truths[i].split()
        score_bleu = sentence_bleu([test_truth], test_prediction)  # reference, predictions
        sum = sum + score_bleu
        num = num + 1
    score_bleu_average = sum / num
    # print('----------------------------------bleu score of testing is: ', score_bleu_average)

    return epoch_loss, score_wer, score_bleu_average

def test(model, iterator_test):
    model.eval()
    padINDEX = 3
    test_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=padINDEX)

    with torch.no_grad():
        for i, data in enumerate(iterator_test):
            input, target = data
            softmax_cal, target_cal, outputs = model(input.float().transpose(0,1).to(device),
                                                     target.long().transpose(0,1).to(device),
                                                     teacher_forcing_ratio = 0)
            loss = criterion(softmax_cal, target_cal)
            test_loss += loss.item()
        test_loss /= len(iterator_test)

    return test_loss

def train(model, asrdata):
    delete_path('./logs')
    writer = SummaryWriter('./logs')

    model.train()
    iterator_train, iterator_test = asrdata.get_iter()
    criterion = nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))  # 计算loss的时候不计算'pad' token
    # optimizer_1 = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    optimizer_1 = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # scheduler_1 = StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler_1 = MultiStepLR(optimizer_1, milestones=[60, 90], gamma=0.1) #in laptop
    scheduler_1 = MultiStepLR(optimizer_1, milestones=[150, 250], gamma=0.1) #in HPC
    # print("初始化的学习率：", optimizer.defaults['lr'])

    for epoch in range(num_epochs):
        model.train()
        print("=" * 50 + ("  EPOCH %i  " % epoch) + "=" * 50)
        Train_Loss = 0
        for i, data in enumerate(iterator_train):
            input, target = data
            optimizer_1.zero_grad()
            softmax_cal, target_cal, outputs = model(input.float().transpose(0,1).to(device),
                                                     target.long().transpose(0,1).to(device),
                                                     teacher_forcing_ratio = 0.5)
            #model input[100, 371, 304]; model target[100, 68, 1];  model outputs:(100, 67, 1)
            loss = criterion(softmax_cal, target_cal)
            loss.backward()
            optimizer_1.step()
            Train_Loss += loss.item()

            # token = target[:,1:,:]
            # if epoch > 0 and i % 100 == 0: ##print one step token and preditions to see the result
            #     for j in range(token.shape[0]):
            #         # token.shape = (batch_size,len(sentens),1), new token[i].shape =(18, 1)-->(18, 1, 1)
            #         print('The sentence', j, ' is:')
            #         print("Truth: \"%s\"" % asrdata.vec_to_sentence(np.expand_dims(token[j].cpu().detach(), axis=2)))
            #         print("Guess: \"%s\"\n" % asrdata.vec_to_sentence(np.expand_dims(outputs[:,j,:], axis=2)))

        print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
        scheduler_1.step()
        Train_Loss /= len(iterator_train)
        print("Train Loss at epoch %d: %.2f \n" % (epoch, Train_Loss))
        #save model after each epoch
        # torch.save(model, "./models/asr_model.pth")
        torch.save(model.state_dict(), "./models/per_gru_weights.pth")

        Test_Loss = test(model, iterator_test)
        print('Test Loss: %.2f' % (Test_Loss))

        writer.add_scalars('loss', {'train_loss': Train_Loss, 'test_loss': Test_Loss}, epoch)
        # terminal input: tensorboard --logdir=./logs

def main():
    ## load data
    start = time.time()
    data = AsrDataLoader(features_path_train, features_path_test, batch_size,
                         targets_path_train, targets_path_test)
    end = time.time()
    print('Time of loading features: %f' % (end - start))

    # Init the AsrModel model  #dict_size = 63
    model = AsrModel(data.dict_size, data.max_length, data.sos, data.pad).to(device)  # output_size=len(dictionary)
    print('Time of initializing model: %f' % (time.time() - end))

    ### train model
    # train(model,data)
    end = time.time()
    print('timit training time costs:', str(end - start))

    ### test model
    # model = torch.load("./models/asr_model.pth")
    model.load_state_dict(torch.load("./models/per_gru_weights_trainComG1_E300_B100_testComCor_wer.90._bleu.02..pth"))
    print('To eval model.pth found and loaded.')
    Test_Loss, Test_WER, Test_Bleu = evaluate(model, data)
    print('Test Loss: %.2f \nTest WER: %.2f \nTest BLEU: %.2f' % (Test_Loss, Test_WER, Test_Bleu))

if __name__ == "__main__":
    main()



