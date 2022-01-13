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

def gradient_descent(x, y, w, b, repeat):
    lr_b = 0
    lr_w = 0
    lr = 1

    for i in range(repeat):
        b_grad = 0.0
        w_grad = 0.0
        
        for n in range(len(x)):
            b_grad = b_grad - 2.0 * (y[n] - b - np.dot(x[n], w)) * 1.0
            w_grad = w_grad - 2.0 * (y[n] - b - np.dot(x[n], w)) * x[n]
        
        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2

        # Update parameters(Use Adagrad)
        b = b - lr/np.sqrt(lr_b) * b_grad
        w = w - lr/np.sqrt(lr_w) * w_grad
        cost(x, y, w, b, i)
    
    return w, b

def cost(x, y, w, b, i):
    hypo = (np.dot(x, w) + b)
    hypo = hypo.reshape((len(hypo), 1))
    loss = hypo - y
    cost = np.sum(np.square(loss)) / len(x)
    rmse_cost = np.sqrt(cost)

    print ('iteration: %d | Cost: %f  ' % (i, rmse_cost))

###########################################################
def test_file_processing(file_path, encode):
    data = []

    with open(file_path, encoding=encode) as file:
        rows = csv.reader(file, delimiter=',')
        for row in rows:
            for i in range(len(row)):
                if row[i] == 'NR':
                    row[i] = 0
            data.append(row)
    
    data = np.array(data)

     # 計算有幾個id(PM2.5, CO, NO ... 等) -> 18
    _id = data[0][0]
    _id_count = 0
    for i in range(len(data)):
            if data[i][0] == _id:
                _id_count += 1
            else:
                break

    return data, _id_count

def gen_predict_data(data, _id_count, w, b):
    # 計算每一個feature有幾天的資料
    N = int(len(data) / _id_count)

    x = np.zeros((N, 9)) # 每一個feature的前九小時資料(240, 9)
    y = np.zeros((N, 1)) # 預測後一小時的值(240, 1)
    output = [['id', 'value']]
    
    for i in range(N):
        x[i] = data[9 + i * _id_count, 2:]
        #y[i] = np.dot(x[i], w.transpose()) + b
        y = float(np.dot(x[i], w.transpose()) + b)
        _id = data[9 + i * _id_count, 0]
        output.append([_id, y])

    with open('predict.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')

        for i in range(len(output)):
            #print(output[i])
            writer.writerow(output[i])

    return output
############################################################    


if __name__ == '__main__':
    
    start = time.time()
    train = file_processing('train.csv', 'big5')
    
    x, y = data_processing(train)

    w = np.ones(len(x[0][0]))
    b = 1
    repeat = 1000

    w, b = gradient_descent(x[9], y[9], w, b, repeat)

    end = time.time()
    print('Total time: ', end - start)
    
    test, _id_count = test_file_processing('test.csv', 'ascii')
    
    output = gen_predict_data(test, _id_count, w, b)    
    