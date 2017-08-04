import numpy, math, csv, time
import pandas as pd
import scipy.stats, scipy.special
import matplotlib.pyplot as plt
from random import randint
from scipy.misc import toimage

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, hiddennodesl2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodesl2 = hiddennodesl2
        self.onodes = outputnodes
        self.lr = learningrate

        #Activation function (sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)
        #self.activation_function = numpy.vectorize(self.cLogLog)

        #Weights of the neural network
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.whh = (numpy.random.rand(self.hnodesl2, self.hnodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodesl2) - 0.5)

    def query(self, inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into hidden layer
        hidden_inputs_l2 = numpy.dot(self.whh, hidden_outputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs_l2 = self.activation_function(hidden_inputs_l2)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs_l2)

        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list):
        targets = numpy.array(targets_list, ndmin=2).T
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into hidden layer 2
        hidden_inputs_l2 = numpy.dot(self.whh, hidden_outputs)
        #calculate the signals emerging from hidden layer 2
        hidden_outputs_l2 = self.activation_function(hidden_inputs_l2)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs_l2)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #output layer error is the (target - actual)
        output_errors = targets - final_outputs

        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #hidden layer 2 error
        hidden_errors_l2 = numpy.dot(self.whh.T, hidden_errors)

        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors_l2 * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        #update the weights for the links between the input and hidden layers
        self.whh += self.lr * numpy.dot((hidden_errors * hidden_outputs_l2 * (1.0 - hidden_outputs_l2)), numpy.transpose(hidden_outputs))

        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_inputs_l2))

    def reverseQuery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal out of the hidden layer
        hidden_outputs_l2 = numpy.dot(self.who.T, final_outputs)

        # calculate the signal out of the hidden layer 2
        hidden_outputs = numpy.dot(self.whh.T, hidden_outputs_l2)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_outputs)

        return inputs

    def save(self):
        numpy.save("weights/wih.npy", self.wih)
        numpy.save("weights/whh.npy", self.whh)
        numpy.save("weights/who.npy", self.who)

    def load(self):
        self.wih = numpy.load("weights/wih.npy")
        self.whh = numpy.load("weights/whh.npy")
        self.who = numpy.load("weights/who.npy")

#Convert the CIFAR-10 batch file to a dictionary
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

#Method to train the neural network
def train(n):
    for b in range(1, 5):
        batch = unpickle("_data/data_batch_" + str(b))

        #loop through batch data
        count = len(batch["labels"])
        for i in range(0, count):
            targets = numpy.zeros(10) + 0.01
            targets[batch["labels"][i]] = 0.99
            n.train(batch["data"][i] / 255, targets)

    n.save()

start_time = time.time()
print "Creating neural network..."
n = NeuralNetwork(3072, 1000, 300, 10, 0.1)
n.load()

print "Testing the neural network..."
batch = unpickle("_data/test_batch")
#loop through batch data
count = len(batch["labels"])
correct = 0
for i in range(0, count):
    result = n.query(batch["data"][i] / 255)
    target = batch["labels"][i]

    if numpy.argmax(result) == target:
        correct = correct + 1

percentage = float(correct) / float(count)
print str(correct) + " correct of " + str(count) + ": " + str(percentage)
