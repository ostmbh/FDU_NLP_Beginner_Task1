from transformers import BertTokenizer
import csv
import os
import numpy
import matplotlib.pyplot as plt


file_path = 'train/train.tsv'

csv.field_size_limit(2 ** 20 * 2)

with open(file_path, mode='r', encoding='utf-8') as tsv_file:
    # 创建一个csv阅读器
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    
###-------------------------------------------------------------data_init

    str = ''

    ct = 0

    Gram = 1

    Data_div = 0.3

    Wordlist = dict()

    epoch = 200

    learn_rate = [10, 1, 0.1, 0.01]

    typenum = 5
    
    Datax = list()
    
    Datay = list()
    
    for row in tsv_reader:
        ct += 1
        if ct > 1:
            words = (row[2].upper()).split()
            Datax.append(row[2])
            Datay.append(row[3])
            for d in range(Gram):
                for i in range(len(words) - d):
                    temp = words[i:i + d +1]
                    temp = "_".join(temp)
                    if temp not in Wordlist:
                        Wordlist[temp] = len(Wordlist)
        if ct >= 12000:
            break
            
    sample = ct - 1

    test_num = int(sample * Data_div)

    train_num = sample - test_num

    train_matr = numpy.zeros((train_num, len(Wordlist)))

    train_y = numpy.zeros((train_num, typenum))

    test_matr = numpy.zeros((test_num, len(Wordlist)))

    test_y = numpy.zeros((test_num, typenum))
    
    train_acc = list()
    
    for ct in range(train_num):
        words = (Datax[ct].upper()).split()
        for d in range(Gram):
            for i in range(len(words) - d):
                temp = words[i:i + d + 1]
                temp = "_".join(temp)
                train_matr[ct][Wordlist[temp]] += 1
        train_y[ct][int(Datay[ct])] += 1
    
    for ct in range(test_num):
        words = (Datax[ct + train_num].upper()).split()
        for d in range(Gram):
            for i in range(len(words) - d):
                temp = words[i:i + d + 1]
                temp = "_".join(temp)
                test_matr[ct][Wordlist[temp]] += 1
        test_y[ct][int(Datay[ct])] += 1

###-------------------------------------------------------------clear

def clea():
    global W, B, train_acc, test_acc
    W = numpy.zeros((typenum, len(Wordlist)))

    B = numpy.zeros((typenum, 1))

    train_acc = []

    test_acc = []

###-------------------------------------------------------------gradient

def soft_calc(X):
    exp = numpy.exp(X - numpy.max(X))
    return exp/exp.sum()

def predic(X):
    global W
    global B
    
    Ypre = (W.dot(X.T)).reshape(-1,1) + B
    Ypre = soft_calc(Ypre)
    return Ypre

def gradient(X, Y):
    incrementW = numpy.zeros((len(Y[0]),len(X[0])))
    incrementB = numpy.zeros((len(Y[0]),1))
    for i in range(train_num):
        Yhat = predic(X[i])
        incrementW += (Y[i].reshape(-1,1) - Yhat)*X[i]
        incrementB += Y[i].reshape(-1,1) - Yhat

    return incrementW,incrementB

###-------------------------------------------------------------train

xepoch=[i for i in range(1, epoch + 1)]

ct = 0

for i in learn_rate:
    clea()
    print(i)
    print('\n')
    ct += 1
    for j in range(epoch):
        print(j)
        print('\n')
        train_cnt = 0
        test_cnt = 0

        for k in range(train_num):
            Yhat = predic(train_matr[k])
            if Yhat.argmax(axis=0) == train_y[k].argmax(axis=0):
                train_cnt += 1
        
        for k in range(test_num):
            Yhat = predic(test_matr[k])
            if Yhat.argmax(axis=0) == test_y[k].argmax(axis=0):
                test_cnt += 1

        iw,ib = gradient(train_matr, train_y)
        
        ##print(iw)
        ##print(ib)
        ##print('\n')
        
        W += iw * i /train_num
        B += ib * i /train_num

        train_acc.append(float(train_cnt)/float(train_num))
        test_acc.append(float(test_cnt)/float(test_num))
        
    plt.figure(ct)

    plt.plot(xepoch, train_acc, 'b--', test_acc, 'r--')

    plt.title('Simple Line Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.show()


###-------------------------------------------------------------draw