import math
def neuron(inputs, weights, bias):
    y = 0
    for x, w in zip(inputs, weights):
        y += x * w
    y += bias
    return(1 / (1 + math.exp(-y)))
    

weights = [10.0, 10.0]
bias = -10.0  # choisis ce que tu veux




def mini_network(x1, x2):
    w = [0.5,1]
    w2 = [0,1]
    w3 = [0.8,0.7]
    b=1
    b2 = 0.4
    b3 = 0.5
    n1 = neuron([x1,x2],w,b)
    n2 = neuron([x1, x2], w2, b2)

    return(neuron([n1, n2], w, b3))

print(mini_network(4,5))