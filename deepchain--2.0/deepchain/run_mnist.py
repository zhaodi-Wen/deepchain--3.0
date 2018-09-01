import numpy as np
import hipsternet.input_data as input_data
import hipsternet.neuralnet as nn
from hipsternet.solver import *
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data
from hipsternet.split_mnist import Split_data
from hipsternet.constant import worker_num

n_iter = 1500
#alpha = 1e-2 adam
alpha =1e-1
mb_size = 64
n_experiment = 10
reg = 1e-5
print_after = 50
p_dropout = .8
loss = 'cross_ent'
nonlin = 'relu'
solver = 'sgd'
solver3 = 'sgd3'
solver4 = 'adam'
#worker_num =10

filename1 = './1.txt'
filename2 = './2.txt'
f1 = open(filename1,'w')
f2 = open(filename2,'w')

mnist = input_data.read_data_sets('MNIST_Data/',one_hot = True)

def prepro(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean


if __name__ == '__main__':
    if len(sys.argv) > 1:
        net_type = sys.argv[1]
        valid_nets = ('ff', 'cnn')

        if net_type not in valid_nets:
            raise Exception('Valid network type are {}'.format(valid_nets))
    else:
        net_type = 'cnn'

    mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)
    '''
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    X_train, X_val, X_test = prepro(X_train, X_val, X_test)
    '''
    
    val_set = [[] for i in range(worker_num)]
    data = Split_data(mnist,worker_num)
    X_train,y_train,X_test,y_test,X_val,y_val= data.split_data()
    print(X_train[0].shape)
    for i in range(worker_num):
        print(y_train[i].shape)
        M, D, C = X_train[i].shape[0], X_train[i].shape[1], y_train[i].max() + 1
        X_train[i], X_val[i], X_test[i] = prepro(X_train[i], X_val[i], X_test[i])
        val_set[i]=(X_val[i], y_val[i])
        print("M {} D {} C {} ".format(M,D,C))
    '''
    if net_type == 'cnn':
        img_shape = (1, 28, 28)
        X_train = X_train[0].reshape(-1, *img_shape)
        X_val = X_val[0].reshape(-1, *img_shape)
        X_test = X_test[0].reshape(-1, *img_shape)
    '''
    if net_type == 'cnn':
        img_shape = (1, 28, 28)
        for i in range(worker_num):
            X_train[i] = X_train[i].reshape(-1, *img_shape)
            X_val[i] = X_val[i].reshape(-1, *img_shape)
            #X_test[i] = X_test[i].reshape(-1, *img_shape)
            val_set[i]=(X_val[i], y_val[i])
    print(X_train[0].shape)
    X1_train,y1_train,X_test,y_test,X1_val,y1_val= mnist.train.images,mnist.train.labels,\
                                                   mnist.test.images,mnist.test.labels,\
                                                   mnist.validation.images,mnist.validation.labels
    
    X1_train,X_test ,X1_val= prepro(X1_train,X_test,X1_val)
    X1_train = X1_train.reshape(-1,*img_shape)
    X_test = X_test.reshape(-1,*img_shape)
    X1_val = X1_val.reshape(-1,*img_shape)
    val1_set = (X1_val,y1_val)
    np.set_printoptions(precision=8,threshold=np.NAN)
    f1.write('X_train {}{}'.format(X_train[0],'\n'))
    f1.write('y_train {}'.format(y_train[0]))
    solvers = dict(
        sgd=sgd,
        sgd3 = sgd3,
        momentum=momentum,
        momentum1=momentum1,
        nesterov=nesterov,
        adagrad=adagrad,
        rmsprop=rmsprop,
        adam=adam
    )

    solver_fun = solvers['momentum1'] #sgd3
    solver2_fun = solvers['momentum'] #sgd
    solver3_fun = solvers[solver4] #adam
    accs = []
    '''
    for i in range(4):
        accs.append(np.zeros(n_experiment))
        '''
    accs1 = np.zeros(n_experiment)
    accs2 = np.zeros(n_experiment)


    print()
    print('Experimenting on {}'.format(solver3))
    print()

    for k in range(n_experiment):
        print('Experiment-{}'.format(k + 1))

        # Reset model
        '''
        if net_type == 'ff':
            net = nn.FeedForwardNet(D, C, H=128, lam=reg, p_dropout=p_dropout, loss=loss, nonlin=nonlin)
        elif net_type == 'cnn':
            net = nn.ConvNet(10, C, H=128)

        
        net = solver_fun(
            net, X_train[0], y_train[0], val_set=(X_val[0], y_val[0]), mb_size=mb_size, alpha=alpha,
            n_iter=n_iter, print_after=print_after
        )

        y_pred = net.predict(X_test[0])
        accs[k] = np.mean(y_pred == y_test[0])
        '''
        #multi worker

        net = []
        for i in range(worker_num):
            if net_type == 'ff':
                net.append(nn.FeedForwardNet(D, C, H=128, lam=reg, p_dropout=p_dropout, loss=loss, nonlin=nonlin))
                net1 = nn.FeedForwardNet(D, C, H=128,lam=reg,p_dropout=p_dropout,loss = loss,nonlin = nonlin)
                net2 = nn.FeedForwardNet(D, C, H=128,lam=reg,p_dropout=p_dropout,loss = loss,nonlin = nonlin)
            elif net_type == 'cnn':
                net.append(nn.ConvNet_new(10, C, H=128))
                net1 = nn.ConvNet_new(10,C,H=128)
                net2 = nn.ConvNet_new(10,C,H=128)

        net = solver_fun(
            net, X_train, y_train, worker_num=worker_num,val_set=val1_set, mb_size=mb_size, alpha=alpha,
            n_iter=n_iter, print_after=print_after
        )
        y_pred = []
        accs=[]
        for i in range(worker_num):
            y_pred.append(net[i].predict(X_test))
            accs.append(np.mean(y_pred[i] == y_test))
        for i in range(worker_num):
            print('Mean accuracy {}: {:.4f}, std: {:.4f}'.format(i+1,accs[i].mean(), accs[i].std()))
        net1 = solver2_fun(
            net1,X_train[0],y_train[0],val_set = val1_set,mb_size=mb_size,alpha=alpha,
            n_iter = n_iter,print_after = print_after
            )
        y1_pred = net1.predict(X_test)
        accs1 = np.mean(y1_pred==y_test)
        print('Mean accuracy :{:.4f},std :{:.4f}'.format(accs1.mean(),accs1.std()))

        net2 = solver2_fun(
            net2,X1_train,y1_train,val_set = val1_set,mb_size = mb_size,alpha = alpha,
            n_iter = n_iter,print_after = print_after
            )
        y2_pred = net2.predict(X_test)
        accs2 = np.mean(y2_pred==y_test)
    print()
    
    for i in range(worker_num):
        print('Mean accuracy {}: {:.4f}, std: {:.4f}'.format(i+1,accs[i].mean(), accs[i].std()))
    
    print('Mean accuracy :{:.4f},std :{:.4f}'.format(accs1.mean(),accs1.std()))
    print('Mean accuracy :{:.4f},std :{:.4f}'.format(accs2.mean(),accs2.std()))
