import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
#load data
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

X = X.reshape([-1,28,28,1])
testX = testX.reshape([-1,28,28,1])



#Convolutional neural network


input = tflearn.input_data(shape=[None,28,28,1],name="input")

conv1 = conv_2d(input,32,3, activation="relu", regularizer="l2")

max_pool1 = max_pool_2d(conv1,2)

conv2 = conv_2d(max_pool1,64,3, activation="relu", regularizer="l2")

max_pool2 = max_pool_2d(conv2,2)

fully_connected1 = fully_connected(max_pool2,100,activation="relu")

fully_connected2 = fully_connected(fully_connected1,10,activation="softmax")

regression = regression(fully_connected2, optimizer="adam", learning_rate=0.01, loss="categorical_crossentropy", name="target")


#Training

model = tflearn.DNN(regression,tensorboard_verbose=0)
model.fit(X,Y, n_epoch=20, validation_set=(testX,testY),show_metric=True, run_id="convNet_model")
