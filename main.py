from utils1 import data_generator
from tcn import compiled_tcn
from sklearn.metrics import confusion_matrix
from swa import SWA
import pandas as pd
import numpy as np

from keras.utils import to_categorical



def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()

    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=32,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train[0:1].shape[1],
                         use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}') 
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')


    model.summary()
    
    epochs =100
    start_epoch =75
    swa = SWA(start_epoch=start_epoch, 
          lr_schedule='constant', 
          swa_lr=0.001, 
          verbose=1)
    
    # start_epoch =2
    # swa = SWA(start_epoch=start_epoch, 
    #       lr_schedule='cyclic', 
    #       swa_lr=0.001,
    #       swa_lr2=0.003,
    #       swa_freq=3,
    #       batch_size=32, # needed when using batch norm
    #       verbose=1)

    model.fit(x_train, y_train.squeeze().argmax(axis=1),callbacks=[swa],batch_size=64,epochs=epochs,validation_data=(x_test, y_test.squeeze().argmax(axis=1)))
    y_test=(y_test.squeeze().argmax(axis=1))
    pre=model.evaluate(x_test,y_test,batch_size=32)
    print('test_loss:',pre[0],'- test_acc:',pre[1])



if __name__ == '__main__':
    run_task()





