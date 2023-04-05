from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras import optimizers, losses, models, layers
from keras.applications.vgg16 import VGG16


def get_dense_model(input_dims,
                    num_layers,
                    layer_width,
                    loss,
                    regularization):
    input_layer = keras.layers.Input(input_dims)
    model = keras.models.Sequential()

    for _ in range(num_layers):
        model.add(keras.layers.Dense(layer_width, activation='relu'))

    model = model(input_layer)
    if loss == 'mle':
        mean = keras.layers.Dense(1)(model)
        var = keras.layers.Dense(1)(model)
        var = keras.layers.Activation(tf.math.softplus)(var)
        output = keras.layers.concatenate([mean, var])
    else:
        if regularization == 0:
            output = keras.layers.Dense(1)(model)
        else:
            reg = keras.regularizers.l1(regularization)
            output = keras.layers.Dense(1, kernel_regularizer=reg)(model)

    dense_net = keras.models.Model(inputs=input_layer, outputs=output)
    return dense_net



def create_model(loss_fn='mae', num_layers=10, layer_width=20, regularization=0, lr=0.01):

    model = get_dense_model(input_dims=(30,),
                            loss=loss_fn,
                            num_layers=num_layers,
                            layer_width=layer_width,
                            regularization=regularization)
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=.9, beta_2=.99)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model



def set_model_weights(model: models.Model, weight_list):
    for i, symbolic_weights in enumerate(model.weights):
        weight_values = weight_list[i]
        K.set_value(symbolic_weights, weight_values)
