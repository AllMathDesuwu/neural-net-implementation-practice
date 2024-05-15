import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice
from my_exceptions import *

class Perceptron(object):
    def __init__(self, in_size = 1, weights = None):
        self.in_size = in_size + 1
        if weights is None:
            self.weights = list()
            for i in range(self.in_size):
                weights.append[1.0]
            self.set_random_weights()
        else:
            self.weights = weights
            
    def get_weighted_sum(self, input_acts):
        weighted_sum = 0
        if (len(input_acts) != len(self.weights)):
            raise ArgLenError(input_acts, self.weights)
        for i in range(len(input_acts)):
            weighted_sum += self.weights[i] * input_acts[i]

        return weighted_sum
    
    def sigmoid(self, val):
        return 1 / (1 + exp(-1 * val))
    
    def sigmoid_activation_function(self, input_acts):
        input_w_bias = input_acts.copy()
        input_w_bias.insert(0, 1)

        weighted_sum = self.get_weighted_sum(input_w_bias)
        return self.sigmoid(weighted_sum)
    
    def sigmoid_derivative(self, val):
        sigm = self.sigmoid(val)
        return sigm * (1 - sigm)
    
    def sigmoid_activation_derivative(self, input_acts):
        input_w_bias = input_acts.copy()
        input_w_bias.insert(0,1)

        weighted_sum = self.get_weighted_sum(input_w_bias)
        return self.sigmoid_derivative(weighted_sum)
    
    def update_weights(self, input_acts, alpha, delta):
        input_w_bias = input_acts.copy()
        input_w_bias.insert(0, 1)

        if (len(input_acts) != len(self.weights)):
            raise ArgLenError(input_w_bias, self.weights)
        
        total_mod = 0
        for i in range(len(input_w_bias)):
            modification = alpha * delta * input_w_bias[i]
            self.weights[i] += modification

            total_mod += abs(modification)

        return total_mod
    
    def set_random_weights(self):
        for i in range(self.in_size):
            self.weights[i] = (random() + 0.0001) * choice([1,-1])

class NeuralNet(object):
    def __init__(self, layer_size, use_default_weights = False):
        self.layer_size = layer_size
        self.output_layer = list()
        self.num_hidden_layers = len(layer_size) - 2
        self.hidden_layers = [[]] * self.num_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        #create the hidden layers
        for h in range(self.num_hidden_layers):
            hidden_layer = list()
            weights = None
            if use_default_weights:
                weights = [0.1] * self.layer_size[h]

            for i in range(self.layer_size[h+1]):
                hidden_layer.append(Perceptron(self.layer_size[h], weights))

            self.hidden_layers.append(hidden_layer)

        #create output layer
        weights = None
        if use_default_weights:
            weights = [0.1] * self.layer_size[-2]

        for percep in range(self.layer_size[-1]):
            self.output_layer.append(Perceptron(self.layer_size[-2], weights))

        layers = self.hidden_layers.copy()
        layers.append(self.output_layer)
        self.layers = layers

    def feed_forward(self, input_acts):
        return_list = list()
        return_list.append(input_acts)

        #first work on our hidden layers
        for layer in self.hidden_layers:
            input_list = return_list[-1] #always use the most recently added list of activations-- remember, the output of the first hidden layer is the input of the second hidden layer
            output_list = list()
            for perceptron in layer:
                output_list.append(perceptron.sigmoid_activation_function(input_list))
            return_list.append(output_list)

        #work on output layer
        input_list = return_list[-1]
        output_list = list()
        for perceptron in self.output_layer:
            output_list.append(perceptron.sigmoid_activation_function(input_list))
        return_list.append(output_list)

        return return_list
    
    def backpropagation(self, examples, alpha):
        '''note: examples is a list of tuples-- first element is input vector, second element is output'''
        avg_error = 0
        avg_weight_change = 0
        num_weights = 0

        for example in examples:
            deltas = list()

            all_layer_output = self.feed_forward(example[0])
            last_layer_output = all_layer_output[-1]
            all_layer_output.pop()

            output_delta = list()
            for output_entry_num in range(len(example[1])):
                deriv = self.output_layer[output_entry_num].sigmoid_activation_deriv(all_layer_output[-1])
                error = example[1][output_entry_num] - last_layer_output[output_entry_num]
                delta = deriv * error
                average_error += 0.5 * error * error
                output_delta.append(delta)
            deltas.append(output_delta)

            for layer_num in range(self.num_hidden_layers - 1, -1, -1):
                layer_deltas = list()
                curr_layer = self.layers[layer_num]
                next_layer = self.layers[layer_num + 1]

                for hidden_neuron_num in range(len(curr_layer)):
                    deriv = curr_layer[hidden_neuron_num].sigmoid_activation_deriv(all_layer_output[layer_num])
                    pseudo_error = 0
                    for i in range(len(next_layer)):
                        pseudo_error += deltas[0][i] * next_layer[i].weights[hidden_neuron_num + 1] #add 1 because bias weight is always in first position-- so 0th neuron gets 1st weight
                    delta = deriv * error
                    average_error += 0.5 * error * error
                    layer_deltas.append(delta)

                deltas = [layer_deltas] + deltas

            for layer_num in range(self.num_layers):
                layer = self.layers[layer_num]
                for neuron_num in range(len(layer)):
                    weight_mod = layer[neuron_num].update_weights(all_layer_output[layer_num], alpha, deltas[layer_num][neuron_num])
                    avg_weight_change += weight_mod
                    num_weights += layer[neuron_num].in_size
                    
        avg_error /= (len(examples) * len(examples[0][1]))
        avg_weight_change /= num_weights
        return avg_error, avg_weight_change

def build_neural_net(examples, alpha = 0.1, weight_change_threshold = 0.00008, hidden_layer_list = [1], max_iter = sys.maxsize, start_net = None):
    examples_train, examples_test = examples
    in_len = len(examples_train[0][0])
    out_len = len(examples_train[0][1])
    right_now = datetime.now().time()
    if start_net is not None:
        hidden_layer_list = list()
        for layer in start_net.hidden_layers:
            hidden_layer_list.append(len(layer))

    #print a message about starting training
    layer_list = [in_len] + hidden_layer_list + [out_len]
    neural_net = NeuralNet(layer_list)
    if start_net is not None:
        neural_net = start_net

    iteration = 0
    train_error = 0
    weight_mod = 1

    while weight_mod > weight_change_threshold and iteration < max_iter:
        train_error, weight_mod = neural_net.backpropagation(examples_train, alpha)
        if iteration % 10 == 0:
            #print some message giving an update on training
            pass
        iteration += 1

    time = datetime.now().time()
    #print some message saying that training's concluded

    test_error = 0
    test_correct = 0

    for example in examples_test:
        are_equal = True
        example_features = example[0]
        expected_out = example[1]
        ff_results = neural_net.feed_forward(example_features)[-1]
        for i in range(expected_out):
            if round(ff_results[i]) != expected_out[i]:
                are_equal = False
                break
            if are_equal:
                test_correct += 1
            else:
                test_error += 1
    
    test_accuracy = test_correct / len(examples_test)

    #print the accuracy of the neural net

    return neural_net, test_accuracy