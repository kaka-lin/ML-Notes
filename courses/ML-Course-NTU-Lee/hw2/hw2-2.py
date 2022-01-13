import time
import csv
import numpy as np
from math import floor


def file_processing(file_path, encode):
    data = []
    
    with open(file_path, encoding=encode) as file:
        rows = csv.reader(file , delimiter=",")
        n_row = 0
        for row in rows:
            if n_row != 0:
                for i in range(len(row)):
                    row[i] = np.float(row[i])
                data.append(row) 
            n_row += 1 

    data = np.array(data)
    return data

def _shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return (x[randomize], y[randomize])

def logistic(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.000000000000000001, 0.999999999999999999)

def gradient_descent(x, y, w, lr, epochs):
    w_grad_last = 0
    N = len(x)

    for i in range(epochs):
        batch_size = 32
        step = int(floor(N / batch_size))
        #x, y = _shuffle(x, y)

        for n in range(step):
            x_t = x[n * batch_size:(n+1) * batch_size]
            y_t = y[n * batch_size:(n+1) * batch_size]
            hypo = np.dot(x_t, w)
            hypo = logistic(hypo)
            loss = y_t - hypo
            w_grad = -1 * (np.dot(x_t.transpose(), loss))
            cost = error(hypo, y_t)
            print ('iteration: %d | Cost: %f  ' % (i, cost))
            
            w_grad_last = w_grad_last + w_grad ** 2
            w_grad_last = np.sum(w_grad_last) / len(w_grad_last)
            #print(w_grad_last)

            # Update parameters(Use Adagrad)
            w = w - lr/np.sqrt(w_grad_last) * w_grad
            #w = w - lr * w_grad

    return w

def gen_predict_data(x, w):
    output = []
    for i in range(len(x)):
        output.append([str(i + 1)])
        y_predict = np.float(np.dot(x[i], w))
        y_predict = logistic(y_predict)
        y_predict = int(np.around(y_predict))
        output[i].append(y_predict)
    
    with open('predict3.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["id","label"])
        for i in range(len(output)):
            writer.writerow(output[i])

def error(hypo, y):
    err = 0
    for i in range(len(hypo)):
        hypo[i] = np.around(hypo[i])
        if hypo[i] != y[i]:
            err += 1
    return err / len(hypo)
    

if __name__ == '__main__':
    start = time.time()
    x_train, y_train = file_processing('X_train', 'ascii'), file_processing('Y_train', 'ascii')
    x_test = file_processing('X_test', 'ascii')
    x_train = np.concatenate((np.ones((len(x_train), 1)), x_train), axis=1)
    x_test = np.concatenate((np.ones((len(x_test), 1)), x_test), axis=1)
    
    w = np.zeros((len(x_train[0]), 1))
    lr = 0.1
    epochs = 20

    w = gradient_descent(x_train, y_train, w, lr, epochs)
    
    gen_predict_data(x_test, w)

    end = time.time()
    print('Total time: ', end - start)
    