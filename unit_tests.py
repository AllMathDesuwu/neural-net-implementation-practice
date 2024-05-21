import unittest
from neuralnet import *

class TestNeuralNet(unittest.TestCase):
    def runTest(self):
        self.test_weighted_sum()
        self.test_sigmoid()
        self.test_sigmoid_activation_func()
        self.test_sigmoid_deriv()
        self.test_sigmoid_activ_deriv()
        self.test_update_weights()

    def test_weighted_sum(self):
        '''Tests for weighted_sum'''
        test_weights = [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 2.0, 3.0], [0, 0, 0], [-1, -2, -3]]
        test_acts = [[1.0, 1.0, 1.0],[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [42, 39, 69], [1.0, 1.0, 1.0]]
        test_sol = [3, 6, 14, 0, -6]

        for i in range(len(test_weights)):
            with self.subTest(i=i):
                test_percep = Perceptron(len(test_weights[i]) - 1, test_weights[i])
                the_sum = test_percep.get_weighted_sum(test_acts[i])
                self.assertEqual(the_sum, test_sol[i])

    def test_sigmoid(self):
        '''Tests for sigmoid'''
        test_in = [0, 1, -1, 24, -24]
        test_sol = [0.5, 0.731059, 0.268941, 1, 0]
        test_weights = [1.0, 1.0, 1.0]

        for i in range(len(test_in)):
            with self.subTest(i=i):
                test_percep = Perceptron(len(test_weights) - 1, test_weights)
                self.assertEqual(round(test_percep.sigmoid(test_in[i]), 6), test_sol[i])

    def test_sigmoid_activation_func(self):
        '''Tests for sigmoid_activation_func'''
        test_weights = [1.0, 1.0, 1.0]
        test_acts = [[1.0, 1.0], [2.0, 3.0], [-1, 0]]
        test_sol = [0.952574, 0.997527, 0.5]

        for i in range(len(test_sol)):
            with self.subTest(i=i):
                test_percep = Perceptron(len(test_weights) - 1, test_weights)
                activation = test_percep.sigmoid_activation_function(test_acts[i])
                self.assertEqual(round(activation, 6), test_sol[i])

    def test_sigmoid_deriv(self):
        '''Tests for sigmoid_derivative'''
        test_in = [0, 1, -1, 24, -24]
        test_sol = [0.25, 0.196612, 0.196612, 0, 0]
        test_weights = [1.0, 1.0, 1.0]

        for i in range(len(test_in)):
            with self.subTest(i=i):
                test_percep = Perceptron(len(test_weights) - 1, test_weights)
                self.assertEqual(round(test_percep.sigmoid_derivative(test_in[i]), 6), test_sol[i])

    def test_sigmoid_activ_deriv(self):
        '''Tests for sigmoid_activation_derivative'''
        test_weights = [1.0, 1.0, 1.0]
        test_acts = [[1.0, 1.0], [2.0, 3.0], [-1, 0]]
        test_sol = [0.045177, 0.002467 ,0.25]

        for i in range(len(test_acts)):
            with self.subTest(i=i):
                test_percep = Perceptron(len(test_weights) - 1, test_weights)
                self.assertEqual(round(test_percep.sigmoid_activation_derivative(test_acts[i]), 6), test_sol[i])

    def test_update_weights(self):
        '''Test for update_weights'''
        test_alpha = 0.1
        test_deltas = [1, 0.5, 0.1]
        test_acts = [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        test_sol = [[1.1, 1.1, 1.1, 1.1], [1.05, 1.05, 1.1, 1.15], [1.01, 1.01, 1.02, 1.03]]
        
        for i in range(len(test_acts)):
            with self.subTest(i=i):
                test_weights = [1.0, 1.0, 1.0, 1.0]
                test_percep = Perceptron(len(test_weights) - 1, test_weights)
                test_percep.update_weights(test_acts[i], test_alpha, test_deltas[i])
                self.assertCountEqual(test_percep.weights, test_sol[i])

    
        
unittest.main()

