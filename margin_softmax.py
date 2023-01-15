import keras.backend as K

import tensorflow as tf





def sparse_logits_categorical_crossentropy(y_true, y_pred, scale=30):

    return K.sparse_categorical_crossentropy(y_true, scale * y_pred, from_logits=True)





# AM-Softmax

def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):

    y_true = K.expand_dims(y_true[:, 0], 1) # 保证y_true的shape=(None, 1)

    y_true = K.cast(y_true, 'int32') # 保证y_true的dtype=int32

    batch_idxs = K.arange(0, K.shape(y_true)[0])

    batch_idxs = K.expand_dims(batch_idxs, 1)

    idxs = K.concatenate([batch_idxs, y_true], 1)

    y_true_pred = tf.gather_nd(y_pred, idxs) # 目标特征，用tf.gather_nd提取出来

    y_true_pred = K.expand_dims(y_true_pred, 1)

    y_true_pred_margin = y_true_pred - margin # 减去margin

    _Z = K.concatenate([y_pred, y_true_pred_margin], 1) # 为计算配分函数

    _Z = _Z * scale # 缩放结果，主要因为pred是cos值，范围[-1, 1]

    logZ = K.logsumexp(_Z, 1, keepdims=True) # 用logsumexp，保证梯度不消失

    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ)) # 从Z中减去exp(scale * y_true_pred)

    return - y_true_pred_margin * scale + logZ




def sparse_asoftmax_loss(y_true, y_pred,  margin=4):
    y_true = K.cast(y_true, 'float32') 
    y_pred= margin * y_pred
    y_pred = y_true * y_pred + (1 - y_true) * y_pred
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)