import numpy as np

#logic OR
input_data_X = np.array([ [0, 0, 1],[0, 1, 1],[1, 0, 1],[1, 1, 1]]) #([[0, 0, 1],[0, 1, 1],[1, 0, 1],[1, 1, 1]])
print(input_data_X)
output_data_Y = np.array([[0,1,1,1]]).T
alpha =  0.01

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_to_derivative(output):
    return output*(1-output)


np.random.seed(1)

syn_0 = 2*np.random.random((3,4)) - 1  #случайные веса
syn_1 = 2*np.random.random((4,1)) - 1


print(syn_0, "\n")
print(syn_1, "\n")


for it in range(50000):
    lay_0 = input_data_X
    lay_1 = sigmoid(np.dot(lay_0,syn_0))
    lay_2 = sigmoid(np.dot(lay_1,syn_1))

    lay_2_lost = (lay_2 - output_data_Y)
    lay_2_delta = lay_2_lost*sigmoid_to_derivative(lay_2)

    lay_1_lost = lay_2_delta.dot(syn_1.T)
    lay_1_delta = lay_1_lost * sigmoid_to_derivative(lay_1)

    syn_1 -= alpha * (lay_1.T.dot(lay_2_delta))
    syn_0 -= alpha * (lay_0.T.dot(lay_1_delta))

    if divmod(it, 100)[1] == 0:
        print(it," iter, lost :", lay_2_lost)

print("lost: ", lay_1_lost, " \n_ \n", "weight: ", syn_1)


lay_0 = np.array([1,1,1])
lay_1 = sigmoid(np.dot(lay_0,syn_0))
lay_2 = sigmoid(np.dot(lay_1,syn_1))

print("\n-------\n input: ", lay_0, "\n hidden layer: ", lay_1, "\n output: ",lay_2)