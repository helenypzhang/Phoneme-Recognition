import zipfile
import torch
import ast
from sklearn import preprocessing
import numpy as np  
import json
import os

def read_data_zip(filename):
    z = zipfile.ZipFile(filename, 'r')

    lines = []
    with z.open(z.namelist()[0]) as f:
        i = 0
        for line in f:
            if i % 100 == 0:
                line = line.decode('utf-8').lower().replace("'", " ").replace(".", "").replace("?", "")\
                    .replace("!", "").replace(":", "").replace(";", "")
                lines.append(line)
            i += 1

    z.close()
    return lines

def read_int_txt(filePath):
    '''
    input: txt filepath; file has seperate spaces, so we should read in line.split()
    output: list type with int list elements in list with whole shape of (4620, 18, 1)
    '''
    with open(filePath) as f:
        myLines = []
        lines = f.readlines()
        for line in lines:
            myLine = []
            for word in line.split():
                # add a dimension to match the language loader 
                myWord = np.zeros((1,1), dtype=np.int32)
                myWord[0][0] = int(word)
                myLine.append(myWord)
            myLines.append(myLine)
    f.close()
    return myLines

def read_json(filePath):
    '''
    the output is list type
    '''
    with open(filePath, 'r') as f:
        data = json.load(f) 
    f.close()
    return data


def generate_dict(filePath):
    with open(filePath) as f:
        mydict = f.read()
        mydict = ast.literal_eval(mydict)
    f.close()
    return mydict

def read_words_txt(filePath):
    '''
    input: txt filepath; file has seperate spaces, so we should read in line.split()
    output: list type with int list elements in list with whole shape of (4620, 18, 1)
    '''
    with open(filePath) as f:
        myLines = []
        lines = f.readlines()
        for line in lines:
            myLine = []
            for word in line.split():
                myLine.append(word)
            myLines.append(myLine)
    f.close()
    return myLines

def read_data(filename):
    '''
    function: read txt file
    input: txt file path
    output: a list of sentences with each sentence for each index
    notice: sentence with no other punctuation mark except comma and space symbol
    '''
    lines = []
    with open(filename, encoding='utf-8') as f: #z.namelist()就是读取的ZipInfo中的filename，组成一个 list返回的
        i = 0
        for line in f:
            if i % 100 == 0:
                line = line.encode('utf-8').decode('utf-8').lower().replace("'", " ").replace(".", "").replace("?", "")\
                    .replace("!", "").replace(":", "").replace(";", "")
                lines.append(line)
            i += 1

    f.close()
    return lines

def read_sentences_txt(filePath):
    '''
    txt file has seperate spaces
    so we should read in line.split()
    the output is list type ['str(sentence1)', 'str(sentence2)']
    '''
    with open(filePath, encoding='utf-8') as f:
        myLines = []
        lines = f.readlines()
        for line in lines:
            line = line.encode('utf-8').decode('utf-8').lower().replace("'", " ").replace(".", "").replace("?", "") \
                .replace("!", "").replace(":", "").replace(";", "")
            myLines.append(line)
    f.close()
    return myLines

def delete_path(root):
    if os.path.isfile(root):
        try:
            os.remove(root)
        except:
            pass
    elif os.path.isdir(root):
        for item in os.listdir(root):
            file = os.path.join(root, item)
            delete_path(file)
            try:
                os.rmdir(root)
            except:
                pass

