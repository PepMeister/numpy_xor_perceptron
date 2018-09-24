import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math
lenth = 10000

input_train_set = []
input_test_set = []
list(map(lambda x: input_train_set.append([]), range(10)))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28)
x_train = x_train.astype('float32')
x_test = x_test.reshape(10000, 28 * 28)
x_test = x_test.astype('float32')


INDEX = 0
print("set making:")
for i in tqdm(range(lenth)):
	try:
		input_train_set[INDEX].append(list((y_train[0:lenth])).index(INDEX))
		y_train[(list((y_train[0:lenth])).index(INDEX))]=99
		INDEX+=1
		if INDEX==10:
			INDEX=0
	except ValueError:
		break

for i in tqdm(range(10)):
    input_test_set.append(list((y_test[0:(lenth-1)])).index(i))
    y_test[(list((y_test[0:(lenth-1)])).index(i))]=99


output_data_Y = np.array([[1,0,0,0,0],[1,0,0,0,1],\
                          [1,0,0,1,0],[1,0,0,1,1],\
                          [1,0,1,0,0],[1,0,1,0,1],\
                          [1,0,1,1,0],[1,0,1,1,1],\
                          [1,1,0,0,0],[1,1,0,0,1]])


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_to_derivative(output):
    return output*(1-output)

np.random.RandomState(1)
syn_0 = 2*np.random.random((784,111)) - 1
syn_1 = 2*np.random.random((111,5)) - 1
print(syn_0, "\n")
print(syn_1, "\n")


alpha =  0.1
accuracy = []
epochnumb = []
list(map(lambda x: accuracy.append([]), range(10)))

print("learning")
lenth = min(list(map(lambda x: len(input_train_set[x]), range(10))))

for it in tqdm(range( lenth )):
    input_data_X =  []
    np.array(list(map(lambda x: input_data_X.append(x_train[input_train_set[x][it]]*0.01), range(10))))

    lay_0 = np.array(input_data_X)
    lay_1 = sigmoid(np.dot(lay_0,syn_0))
    lay_2 = sigmoid(np.dot(lay_1,syn_1))

    lay_2_lost = (lay_2 - output_data_Y)
    lay_2_delta = lay_2_lost*sigmoid_to_derivative(lay_2)

    lay_1_lost = lay_2_delta.dot(syn_1.T)
    lay_1_delta = lay_1_lost * sigmoid_to_derivative(lay_1)

    syn_1 -= alpha * (lay_1.T.dot(lay_2_delta))
    syn_0 -= alpha * (lay_0.T.dot(lay_1_delta))

    if divmod(it, 1)[1] == 0:
        accuracy[0].append(math.fabs(lay_2_lost[0][0]))
        accuracy[1].append(math.fabs(lay_2_lost[1][0]))
        accuracy[2].append(math.fabs(lay_2_lost[2][0]))
        accuracy[3].append(math.fabs(lay_2_lost[3][0]))
        accuracy[4].append(math.fabs(lay_2_lost[4][0]))
        accuracy[5].append(math.fabs(lay_2_lost[5][0]))
        accuracy[6].append(math.fabs(lay_2_lost[6][0]))
        accuracy[7].append(math.fabs(lay_2_lost[7][0]))
        accuracy[8].append(math.fabs(lay_2_lost[8][0]))
        accuracy[9].append(math.fabs(lay_2_lost[9][0]))
        epochnumb.append(it)
        #print(it," iter, lost :", lay_2_lost)


print("lost: \n", lay_1_lost, " \n_ \n", "weight: \n", syn_1)

inp=str(input("\n[#] Show graph [Y/n]?"))
if(inp=="y" or inp=="Y"):
    plt.plot(np.array(epochnumb), np.array(accuracy[0]),  np.array(epochnumb),\
         np.array(accuracy[1]), np.array(epochnumb), np.array(accuracy[2]), np.array(epochnumb), np.array(accuracy[3]),\
         np.array(epochnumb), np.array(accuracy[4]), np.array(epochnumb),\
         np.array(accuracy[5]), np.array(epochnumb), np.array(accuracy[6]), np.array(epochnumb), np.array(accuracy[7]),\
         np.array(epochnumb), np.array(accuracy[8]), np.array(epochnumb), np.array(accuracy[9]),)
    plt.show()
    plt.savefig('neural_accuracy'+'.png')


input_train_data_ =  []
np.array(list(map(lambda x: input_train_data_.append(x_test[input_test_set[x]]*0.01), range(10))))


while(True):
    try:
        INPUT_NUMBER = int(input("\n[#] Enter the number [0..9]: "))
        lay_0 = input_train_data_[INPUT_NUMBER]                                                                             ##########input
        print("\n-------\n input: \n", (np.array(list(map(lambda x: int(round(x, 1)), input_train_data_[INPUT_NUMBER])))).reshape(1,28,28))
        print("-------------")

        lay_1 = sigmoid(np.dot(lay_0,syn_0))
        lay_2 = sigmoid(np.dot(lay_1,syn_1))

        print("output:\n",lay_2)
        out = list(map(lambda x: (int(round(x,0))), lay_2))
        out[0] = 0
        print("in binary: ",out)
        print("in decimal: ", int(''.join(str(x) for x in out),2))

    except KeyboardInterrupt:
        print("exit")
        break