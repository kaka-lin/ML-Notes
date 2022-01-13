import numpy as np
import pandas as pd
import time
import csv

def file_processing(file_path, encode):
    data = []
    # 每一個維度儲存一種污染物的資訊
    for i in range(18):
	    data.append([])
    
    with open(file_path, encoding=encode) as file:
        rows = csv.reader(file , delimiter=",")
        n_row = 0
        for row in rows:
            if n_row != 0:
                for i in range(3, 27):
                    if row[i] != 'NR':
                        data[(n_row-1)%18].append(row[i]) 
                    else:
                        data[(n_row-1)%18].append(0)
                            
            n_row += 1    

    data = np.array(data)
    return data

def data_processing(data):
    # 資料有12個月 X 20天 ＝ 240
    # 一個月(20天)連續取10小時的資料 -> 有20x24-9 = 471筆
    # 一年連續取10小時資料 -> y 有240x24-9 = 5751筆
    #x = np.zeros((5751, 9))
    #y = np.zeros((5751, 1))
    x_total = []
    y_total = []

    for n in range(18):
        x = np.zeros((5751, 9))
        y = np.zeros((5751, 1))
        for i in range(5751):
            x[i] = data[n][i:i+9]
            y[i] = data[n][i+9]
        x_total.append(x)
        y_total.append(y)

    x_total, y_total = np.array(x_total), np.array(y_total)
    return x_total, y_total

def gradient_descent(x, y, w, b, lr, repeat):
    b_grad_last = 0
    w_grad_last = 0

    for i in range(repeat):
        hypo = np.dot(x, w) + b
        loss = y - hypo
        loss_delta = -1.0 * loss
        cost = np.sum(np.square(loss)) / len(loss)
        rmse_cost = np.sqrt(cost)
        print ('iteration: %d | Cost: %f  ' % (i, rmse_cost))

        b_grad = loss_delta
        b_grad = np.sum(b_grad) / len(b_grad)
        w_grad = np.dot(x.transpose(), loss_delta)
        
        b_grad_last = b_grad_last + b_grad ** 2
        w_grad_last = w_grad_last + w_grad ** 2
        w_grad_last = np.sum(w_grad_last) / len(w_grad_last)

        # Update parameters(Use Adagrad)
        b = b - lr/np.sqrt(b_grad_last) * b_grad
        w = w - lr/np.sqrt(w_grad_last) * w_grad

    return w, b

def stochastic_gradient_descent(x, y, w, b, lr, repeat):
    b_grad_last = 0
    w_grad_last = 0
    N = len(x)

    for i in range(repeat):
        x1, y1 = x[i % N], y[i % N]
        x1 = x1.reshape((1, len(x1)))

        hypo = np.dot(x[i % N], w) + b
        loss = y1 - hypo
        loss_delta = -1.0 * loss
        cost = np.sum(np.square(loss)) / len(loss)
        rmse_cost = np.sqrt(cost)
        print ('iteration: %d | Cost: %f  ' % (i, rmse_cost))
        
        b_grad = loss_delta
        b_grad = np.sum(b_grad) / len(b_grad)
        w_grad = np.dot(x1.transpose(), loss_delta)
        w_grad = w_grad.reshape((len(w_grad), 1))
         
        b_grad_last = b_grad_last + b_grad ** 2
        w_grad_last = w_grad_last + w_grad ** 2
        w_grad_last = np.sum(w_grad_last) / len(w_grad_last)

        # Update parameters(Use Adagrad)
        b = b - lr/np.sqrt(b_grad_last) * b_grad
        w = w - lr/np.sqrt(w_grad_last) * w_grad
        
    return w, b


###########################################################
def test_file_processing(file_path, encode):
    data = []

    with open(file_path, 'r') as file:
        rows = csv.reader(file, delimiter=',')
        for row in rows:
            for i in range(2, 11):
                if row[i] == 'NR':
                    row[i] = np.float(0)
                    
                else:
                    row[i] = np.float(row[i])
            data.append((row[2:]))
    
    data = np.array(data)
    return data

def test_data_processing(data):
    x_test_total = []
    # 每一個維度儲存一種污染物的資訊
    for i in range(18):
	    x_test_total.append([])
    for i in range(len(data)):
        x_test_total[i%18].append(data[i])
        
    x_test_total = np.array(x_test_total)
    return x_test_total


def gen_predict_data(x, w, b):
    output = []
    for i in range(len(x)):
        output.append(['id_' + str(i)])
        y_predict = np.float(np.dot(x[i], w)) + b
        output[i].append(y_predict)
    
    with open('predict3.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["id","value"])
        for i in range(len(output)):
            writer.writerow(output[i])
############################################################    


if __name__ == '__main__':
    start = time.time()
    train = file_processing('train.csv', 'big5')
    test = test_file_processing('test.csv', 'ascii')
    x, y = data_processing(train)
    x_test = test_data_processing(test)
    
    w = np.zeros((len(x[0][0]), 1))
    b = 1
    lr = 0.1
    repeat = 10000
    
    x_train, y_train, w_train = x[9], y[9], w
    x_t = x_test[9]

    x_train, w_train = np.hstack((x_train, x[1], x[3], x[4], x[6], x[5], x[8], x[7], x[2], x[10], x[0])), np.vstack((w_train, w, w, w, w, w, w, w, w, w, w))
    x_t = np.hstack((x_t, x_test[1], x_test[3], x_test[4], x_test[6], x_test[5], x_test[8], x_test[7], x_test[2], x_test[10], x_test[0]))

    w, b = gradient_descent(x_train, y_train, w_train, b, lr, repeat)
    #w, b = stochastic_gradient_descent(x_train, y_train, w_train, b, lr, repeat)
    gen_predict_data(x_t, w, b)    

    end = time.time()
    print('Total time: ', end - start)
    