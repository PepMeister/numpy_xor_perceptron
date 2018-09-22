from math import fabs
import numpy as np
import matplotlib.pyplot as plt

#logic OR
input_data_X = np.array([ [0.1, 0.01, 1],[0, 1, 1],[1, 0.05, 1],[0.99, 1, 1]]) #([[0, 0, 1],[0, 1, 1],[1, 0, 1],[1, 1, 1]])
output_data_Y = np.array([[0, 0.9, 0.99, 1]]).T
alpha =  0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_to_derivative(output):
    return output*(1-output)

np.random.RandomState(1)
#np.random.seed(1)
syn_0 = 2*np.random.random((3,4)) - 1
syn_1 = 2*np.random.random((4,1)) - 1
print(syn_0, "\n")
print(syn_1, "\n")


accuracy = ([], [], [], [])
epochnumb = []

for it in range(10000):
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
        accuracy[0].append(fabs(lay_2_lost[0]))
        accuracy[1].append(fabs(lay_2_lost[1]))
        accuracy[2].append(fabs(lay_2_lost[2]))
        accuracy[3].append(fabs(lay_2_lost[3]))

        epochnumb.append(it)
        print(it," iter, lost :", lay_2_lost)


print("lost: \n", lay_1_lost, " \n_ \n", "weight: \n", syn_1)

lay_0 = np.array([1,1,1])
lay_1 = sigmoid(np.dot(lay_0,syn_0))
lay_2 = sigmoid(np.dot(lay_1,syn_1))

print("\n-------\n input: ", lay_0)
print("-------------")
print("веса (1 слой): \n", syn_0)
print("-------------")
print("*результат перемножения матриц lay_0, syn_0 (lay_1):* \n")
print(np.dot(lay_0,syn_0), "\n _")
print("*after sigmoid func: \n",lay_1)

print("-------------")
print("веса (2 слой): \n", syn_1)
print("-------------")
print("*результат перемножения матриц lay_1, syn_1:* \n")
print(np.dot(lay_1,syn_1))

print("-------------")
print("*after sigmoid func: \n",lay_2)
print("output: ",lay_2)


plt.plot(np.array(epochnumb), np.array(accuracy[0]), np.array(epochnumb),\
         np.array(accuracy[1]), np.array(epochnumb), np.array(accuracy[2]), np.array(epochnumb), np.array(accuracy[3]))
plt.savefig('neural_accuracy'+'.png')
plt.show()
