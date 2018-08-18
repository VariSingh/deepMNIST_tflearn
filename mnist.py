import tflearn
 #load data
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
 #neural network
input = tflearn.input_data(shape=[None,784])
input = tflearn.fully_connected(input,100,activation='relu')
layer1 = tflearn.fully_connected(input,100,activation='relu')
layer2 = tflearn.fully_connected(layer1,100,activation='relu')
output = tflearn.fully_connected(layer2,10,activation='softmax')
sgd = tflearn.SGD(learning_rate=0.1)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(output, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')
 #training
model = tflearn.DNN(net,tensorboard_verbose=0)
model.fit(X,Y, n_epoch=20, validation_set=(testX,testY),show_metric=True, run_id="dense_model")
