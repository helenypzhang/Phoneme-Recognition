from numpy.core.defchararray import index
from numpy.lib.function_base import average
from numpy.lib.shape_base import expand_dims
from DataLoader_initialpad_sample_ctc import *
from Model_deForMND_sample_bigru_ctc import *
from utils import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from torchmetrics import WER
from nltk.translate.bleu_score import sentence_bleu #nltk.translate.bleu_score.sentence_bleu()

# For demo dataset:
# features demo file is 'timit_train_features_demo.json'
# targets demo file is 'timit_train_phonemes_demo.txt'

features_path_train = '../features/timit_train_mfcc_585.json' #all features with fixed size
targets_path_train = '../targets/timit_train_phonemes.txt' #all translations not padded

features_path_test = '../features/timit_test_mfcc_585.json'
targets_path_test = '../targets/timit_test_phonemes.txt'

num_epochs = 100 # number of times training the whole train dataset
# batch_size = 64 # for sentenses
batch_size = 30 # for phonemes
initial_lr = 0.001
ctcRate = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# shape of features: (4620, 371, 304) [num_sentences, num_frames, num_featureDim]
# shape of gt.txt: (4620, 18, 1) [num_sentences, num_words, num_featureDim]

def evaluate(model, asrdata):
    model.eval()
    iterator_train, iterator_test = asrdata.get_iter()
    crossE_loss = nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))
    ctc_loss = nn.CTCLoss(blank=asrdata.dict_size-1, reduction='mean')
    # criterion = r * ctc_loss + (1-r) * cross_loss
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
            softmax_cal, target_cal, outputs, ctc_outputs, ctc_targets = model(input.float().transpose(0,1).to(device),
                                                     target.long().transpose(0,1).to(device),
                                                     teacher_forcing_ratio = 0)
            # model input:[100, 371, 304]; model target:[100, 68, 1]; model output:(100, 67, 1)

            # CTC LOSS
            ctcInput = ctc_outputs.log_softmax(2) #[frame,N,D], log_softmax didn't change shape
            ctcTarget = ctc_targets.squeeze(2) #[N,M]
            batch_size = ctc_outputs.shape[1]
            frame_size = ctc_outputs.shape[0]
            max_length = ctc_targets.shape[1]
            input_lengths = torch.full((batch_size,), frame_size, dtype=torch.long)
            target_length = torch.full((batch_size,), max_length, dtype=torch.long)

            loss1 = crossE_loss(softmax_cal, target_cal)
            loss2 = ctc_loss(ctcInput, ctcTarget, input_lengths, target_length)
            loss = ctcRate * loss2 + (1-ctcRate) * loss1

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
    metric = WER()
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

def test(model, iterator_test, dict_size):
    model.eval()
    padINDEX = 3
    test_loss = 0
    # criterion = nn.CrossEntropyLoss(ignore_index=padINDEX)
    crossE_loss = nn.CrossEntropyLoss(ignore_index=padINDEX)
    ctc_loss = nn.CTCLoss(blank=dict_size-1, reduction='mean')

    with torch.no_grad():
        for i, data in enumerate(iterator_test):
            input, target = data
            softmax_cal, target_cal, outputs, ctc_outputs, ctc_targets = model(input.float().transpose(0,1).to(device),
                                                     target.long().transpose(0,1).to(device),
                                                     teacher_forcing_ratio = 0)
            # loss = criterion(softmax_cal, target_cal)
            # CTC LOSS
            ctcInput = ctc_outputs.log_softmax(2) #[frame,N,D], log_softmax didn't change shape
            ctcTarget = ctc_targets.squeeze(2) #[N,M]
            batch_size = ctc_outputs.shape[1]
            frame_size = ctc_outputs.shape[0]
            max_length = ctc_targets.shape[1]
            input_lengths = torch.full((batch_size,), frame_size, dtype=torch.long)
            target_length = torch.full((batch_size,), max_length, dtype=torch.long)

            loss1 = crossE_loss(softmax_cal, target_cal)
            loss2 = ctc_loss(ctcInput, ctcTarget, input_lengths, target_length)
            loss = ctcRate * loss2 + (1-ctcRate) * loss1

            test_loss += loss.item()
        test_loss /= len(iterator_test)

    return test_loss

def train(model, asrdata):
    delete_path('./logs')
    writer = SummaryWriter('./logs')

    model.train()
    iterator_train, iterator_test = asrdata.get_iter()
    # criterion = nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))  # 计算loss的时候不计算'pad' token
    crossE_loss = nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))
    ctc_loss = nn.CTCLoss(blank=asrdata.dict_size-1, reduction='mean')
    # optimizer_1 = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    optimizer_1 = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # scheduler_1 = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler_1 = MultiStepLR(optimizer_1, milestones=[60, 90], gamma=0.1)
    # print("初始化的学习率：", optimizer.defaults['lr'])

    for epoch in range(num_epochs):
        model.train()
        print("=" * 50 + ("  EPOCH %i  " % epoch) + "=" * 50)
        Train_Loss = 0
        for i, data in enumerate(iterator_train):
            input, target = data
            optimizer_1.zero_grad()
            softmax_cal, target_cal, outputs, ctc_outputs, ctc_targets = model(input.float().transpose(0,1).to(device),
                                                     target.long().transpose(0,1).to(device),
                                                     teacher_forcing_ratio = 0.5)
            #model input[100, 371, 304]; model target[100, 68, 1];  model outputs:(100, 67, 1)
            # loss = criterion(softmax_cal, target_cal)
            # CTC LOSS
            ctcInput = ctc_outputs.log_softmax(2) #[frame,N,D], log_softmax didn't change shape
            ctcTarget = ctc_targets.squeeze(2) #[N,M]
            batch_size = ctc_outputs.shape[1]
            frame_size = ctc_outputs.shape[0]
            max_length = ctc_targets.shape[1]
            input_lengths = torch.full((batch_size,), frame_size, dtype=torch.long)
            target_length = torch.full((batch_size,), max_length, dtype=torch.long)

            loss1 = crossE_loss(softmax_cal, target_cal)
            loss2 = ctc_loss(ctcInput, ctcTarget, input_lengths, target_length)
            loss = ctcRate * loss2 + (1-ctcRate) * loss1

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
        torch.save(model.state_dict(), "./models/asr_model_weights.pth")

        Test_Loss = test(model, iterator_test, asrdata.dict_size)
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
    model = AsrModel(data.dict_size, data.frame_length, data.max_length, data.sos, data.pad).to(device)  # output_size=len(dictionary)
    print('Time of initializing model: %f' % (time.time() - end))

    ### train model
    train(model,data)
    end = time.time()
    print('timit training time costs:', str(end - start))

    ### test model
    # model = torch.load("./models/asr_model.pth")
    model.load_state_dict(torch.load("./models/asr_model_weights.pth"))
    print('To eval model.pth found and loaded.')
    Test_Loss, Test_WER, Test_Bleu = evaluate(model, data)
    print('Test Loss: %.2f \nTest WER: %.2f \nTest BLEU: %.2f' % (Test_Loss, Test_WER, Test_Bleu))

if __name__ == "__main__":
    main()



