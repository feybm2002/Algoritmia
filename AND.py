# Maria Fernanda Barroso Monroy 
# Perceptron AND 
import numpy as np # Library for mathematical operations

# Perceptrón AND
def perceptron_AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7  # weights and theta
    suma = x1 * w1 + x2 * w2       # weighted sum
    if suma <= theta:              # activation function
        return 0
    else:
        return 1

# Print results
print(perceptron_AND(0,0))  # Output: 0
print(perceptron_AND(0,1))  # Output: 0
print(perceptron_AND(1,0))  # Output: 0
print(perceptron_AND(1,1))  # Output: 1
