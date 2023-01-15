import numpy as np
from keras.utils import to_categorical
import scipy.io 
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras import layers 
import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras import layers 
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_selection import SelectKBest,chi2


def data_generator():



    

    data = scipy.io.loadmat("E:\\DEAP\\DEAP_ALLS\\ALLS01.mat")
    data = data['d']
    data=data.reshape(2400,4096)
    label= scipy.io.loadmat("E:\\DEAP\\DEAP_ALLS\\ALLS01.mat")
    label = label['labA']


    data = (data - data.mean(axis = 0))/(data.std(axis = 0))     
    x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state=42)


    img_rows, img_cols = 32,128
    data = data.reshape(-1, img_rows * img_cols, 1)
    
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    
    # data = scipy.io.loadmat("E:\\SEED\\DE\\d1.mat")
    # x_train = data['x_train1']
    # x_train = x_train.astype('float32')
    # x_test = data['x_test1']
    # x_test = x_test.astype('float32')
    # y_train = data['y_train1']
    # y_test = data['y_test1']
    
   
    # img_rows, img_cols = 62,5
    # x_train = x_train.reshape(-1, img_rows * img_cols, 1)
    # x_test = x_test.reshape(-1, img_rows * img_cols, 1)


    # x_train = x_train.astype('float32')
    # x_train=x_train.reshape(2010,62,5)
    # x_test = x_test.astype('float32')
    # x_test=x_test.reshape(1384,62,5)
 
    
    

    # y_train = y_train.astype('int64')
    # y_test = y_test.astype('int64')
 
    return (x_train, y_train), (x_test, y_test)




if __name__ == '__main__':
    print(data_generator())