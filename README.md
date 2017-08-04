# CIFAR-10-NeuralNet
My journey to achieving >75% accuracy on the CIFAR-10 dataset. Trying different techniques and learning the basics of machine learning.

## Journal
Attemps in chronological order.
- (Deep) Feed Forward NN: 13.41% accuracy.

## (Deep) Feed Forward NN
Based on the book 'Make your own Neural Network' by Tariq Rashid. This Neural Network has an extra hidden layer. The setup is:
- Input: 3072
- Hidden l1: 1000
- Hidden l2: 300
- Output: 10
- Learning rate: 0.1

After training the neural network for 11 iterations (which took 03:09:50) it managed to achieve a poor maximum performance of 13.41% at 3 iterations. A performance of 13.41% is just above the chance of just guessing so there definitely should be some big adjustments for this model. See the plot below for the iterations vs performance:
 
![0](https://jenzus.com/assets/cifar-nn-plot.png)

The plot shows that every other iteration the network's performance drops significantly, this makes me believe that the learning rate could use some adjustments. 

## Deep Convolutional NN
After a bit of researching it seems that going for the Convolutional NN model should be a better approach for image classification, my next step will be to implement that and see the difference.
