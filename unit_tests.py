import unittest
from neuralnet import *

class TestNeuralNet(unittest.TestCase):
    def test_sigmoid(self):
        test_weights = [1.0, 1.0, 1.0]
        test_percep = Perceptron(len(test_weights), test_weights)