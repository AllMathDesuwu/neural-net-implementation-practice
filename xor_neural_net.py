from neuralnet import *

def main():
    layer_size = [2, 3, 1] #XOR should need only two perceptrons in a hidden layer and one output perceptron
    examples = ([([0,0], [0]), ([0,1], [1]), ([1,0], [1]), ([1,1], [0])], [([0,0], [0]), ([0,1], [1]), ([1,0], [1]), ([1,1], [0])])
    xor_net = NeuralNet(layer_size)
    xor_net, test_accuracy = build_neural_net(examples, max_iter = 5000, start_net = xor_net)    

if __name__ == "__main__":
    main()