from Neuron_Layer import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import timeit
# Spiral Dataset
def generateData(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y



if __name__ == '__main__':
    # 100 size-2 feature sets, 3 times for 3 labels

    X, y = generateData(100, 3)

    # a is the learning rate
    def forward(inputs):
        layer1.forward(inputs)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        return activation2.output


    delta = 0.025
    x_ = y_ = np.arange(-1.0, 1.0, delta)
    n = len(x_)
    X_, Y_ = np.meshgrid(x_, y_)
    X_ = X_.reshape(-1)
    Y_ = Y_.reshape(-1)
    Y_ = Y_[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")

    lossArray = []
    stepsArray = []
    for epoch in range(10001):
        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        data_loss = lossSoftmax.calculate(layer2.output, y)
        regularisation_loss = lossSoftmax.regularization_loss(layer1) + lossSoftmax.regularization_loss(layer2)
        loss = data_loss + regularisation_loss
        predictions = np.argmax(lossSoftmax.output, axis=1)
        accuracy = np.mean(predictions == y)

        if epoch % 500:
            lossArray.append(loss)
            stepsArray.append(epoch)
        # input the softmax outputs and true y values
        lossSoftmax.backward(lossSoftmax.output, y)
        layer2.backward(lossSoftmax.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)
        if epoch % 2000 == 0:
            print("epoch:", epoch)
            print("loss:", loss)
            print("data_loss:", data_loss)
            print("reg_loss:", regularisation_loss)
            print("acc:", accuracy)
            print("lr:", optimiser.current_learning_rate)
            print("-------------------------------------")

        optimiser.pre_update_params()
        optimiser.update_params(layer=layer1)
        optimiser.update_params(layer=layer2)
        optimiser.post_update_params()

    Z_ = forward(np.array(list(zip(X_, Y_))))
    Z_ = np.argmax(Z_, axis=1).reshape(n, n)
    im = ax1.imshow(Z_, interpolation="bilinear", cmap="brg", extent=[-1, 1, -1, 1], alpha=0.5)
    ax1.axis('scaled')
    ax2.plot(stepsArray, lossArray)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    plt.show()
    #plt.colorbar()

