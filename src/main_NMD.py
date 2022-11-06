from numpy.core.defchararray import index
from numpy.lib.function_base import average
from numpy.lib.shape_base import expand_dims
from DataLoader_initialpad import *
from Model_deForNMD_gru import *
from utils import *
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from torchmetrics import WER
from nltk.translate.bleu_score import sentence_bleu #nltk.translate.bleu_score.sentence_bleu()
# For demo dataset:
# features demo file is 'timit_train_features_demo.json'
# targets demo file is 'timit_train_phonemes_demo.txt'

features_path_train = '../features/timit_train_features_demo.json' #all features with fixed size
targets_path_train = 'F:/datasets/timit_train_phonemes_demo.txt' #all translations not padded

# features_path_test = '../features/timit_test_features.json'
# targets_path_test = 'F:/datasets/timit_test_phonemes.txt'
features_path_test = '../features/timit_train_features_demo.json'
targets_path_test = 'F:/datasets/timit_train_phonemes_demo.txt'

# max_length = 18 # control the maximum translation length.,NOT USED, INSTEAD USE MAX_LEN IN DATALOADER
num_epochs = 50 # number of times training the whole train dataset
batch_size = 32 # number of words/frames trained in each input
vocab_size = 70
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# shape of features: (4620, 371, 304) [num_sentences, num_frames, num_featureDim]
# shape of gt.txt: (4620, 18, 1) [num_sentences, num_words, num_featureDim]
# vocab_size = 5256 # size of the dictionary with 1 add for '<PAD>' 5255+1 # Not used in train/test
def evaluate(model, asrdata):
    model.eval()
    iterator_train, iterator_test = asrdata.get_iter()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))
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
            softmax_cal, target_cal, outputs = model(input.float().to(device),target.long().to(device),teacher_forcing_ratio = 0)
            # model input:[100, 371, 304]; model target:[100, 68, 1]; model output:(100, 67, 1)
            loss = criterion(softmax_cal, target_cal)
            epoch_loss += loss
            print("Test Loss at batch %d: %.2f" % (i, loss))
            # Evaluation Metrics prepare
            token = target[:,1:,:]
            for j in range(token.shape[0]):
                # token.shape = (batch_size,len(sentens),1), new token[i].shape =(18, 1)-->(18, 1, 1)
                token_clean = []
                outputs_clean = []
                for n in range(token[j].shape[0]): ## token[j] is a sentence # token[j][n] is a number, refer to the word id
                    if token[j][n] != padINDEX and token[j][n] != sosINDEX and token[j][n] != eosINDEX and token[j][n] != unkINDEX:
                        token_clean.append([token[j][n].item()])
                        outputs_clean.append(outputs[j][n])
                truth = asrdata.vec_to_sentence(np.expand_dims(token_clean, axis = 2))
                prediction = asrdata.vec_to_sentence(np.expand_dims(outputs_clean, axis=2))
                test_truths.append(truth)
                test_predictions.append(prediction)
                if (i % 100 == 0):
                    print('The sentence', j,' is:')
                    print("Truth: \"%s\"" % truth) #str
                    print("Guess: \"%s\"\n" % prediction) #string # outputs.shape = [batch_size, 18, 1], str
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



def train(model, asrdata):
    delete_path('./runs')
    writer = SummaryWriter('./runs')

    model.train()
    iterator_train, iterator_test = asrdata.get_iter()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(asrdata.pad[0][0]))  # 计算loss的时候不计算'pad' token
    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    for epoch in range(num_epochs):
        print("=" * 50 + ("  EPOCH %i  " % epoch) + "=" * 50)
        epoch_loss = 0
        for i, data in enumerate(iterator_train):
            input, target = data
            optimizer.zero_grad()
            softmax_cal, target_cal, outputs = model(input.float().to(device), target.long().to(device), teacher_forcing_ratio = 0)
            #model input[100, 371, 304]; model target[100, 68, 1];  model outputs:(100, 67, 1)
            loss = criterion(softmax_cal, target_cal)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            # print("Train Loss at batch %d: %.2f" % (i, loss))

            token = target[:,1:,:]
            if epoch > 0 and i % 100 == 0: ##print one step token and preditions to see the result
                for j in range(token.shape[0]):
                    # token.shape = (batch_size,len(sentens),1), new token[i].shape =(18, 1)-->(18, 1, 1)
                    print('The sentence', j, ' is:')
                    print("Truth: \"%s\"" % asrdata.vec_to_sentence(np.expand_dims(token[j].cpu().detach(), axis=2)))
                    print("Guess: \"%s\"\n" % asrdata.vec_to_sentence(np.expand_dims(outputs[j], axis=2)))

        epoch_loss /= len(iterator_train)
        print("Train Loss at epoch %d: %.2f \n" % (epoch, epoch_loss))
        writer.add_scalars('loss', {'training loss': epoch_loss}, epoch)
        # terminal input: tensorboard --logdir=./runs --port 8123

        #save model after each epoch
        # torch.save(model, "./models/asr_model.pth")
        torch.save(model.state_dict(), "./models/asr_model_weights.pth")

def main():
    ## load data
    start = time.time()
    data = AsrDataLoader(features_path_train, features_path_test, vocab_size, batch_size,
                         targets_path_train, targets_path_test)
    end = time.time()
    print('Time of loading features: %f' % (end - start))

    # Init the AsrModel model  #dict_size = 63
    model = AsrModel(data.dict_size, data.max_length, data.sos, data.pad).to(device)  # output_size=len(dictionary)
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



