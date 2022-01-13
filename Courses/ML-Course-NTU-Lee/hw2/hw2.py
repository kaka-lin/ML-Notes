import time
import csv
import numpy as np


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

def gradient_descent(x, y, w, lr, repeat):
    w_grad_last = 0

    for i in range(repeat):
        
        hypo = np.dot(x, w)
        hypo = logistic(hypo)
        loss = y - hypo
        w_grad = -1 * (np.dot(x.transpose(), loss))
        cost = error(hypo, y)
        print ('iteration: %d | Cost: %f  ' % (i, cost))
        
        w_grad_last = w_grad_last + w_grad ** 2

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
    
    with open('predict.csv', 'w') as file:
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
    lr = 0.0001
    repeat = 2000

    w = gradient_descent(x_train, y_train, w, lr, repeat)
    
    gen_predict_data(x_test, w)

    end = time.time()
    print('Total time: ', end - start)
    