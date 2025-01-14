import numpy as np

class Perceptron:

    def __init__(self, input_dim, layer_dims):
        self.layers = []

        dims = np.concatenate(([input_dim], layer_dims))
        for i in range(1, len(dims)):
            layer = []
            for j in range(dims[i]):
                layer.append(self.Neuron([np.random.random() for i in range(dims[i-1]+1)]))
            self.layers.append(layer)

    def process(self, inputs):
        values = inputs
        for layer in self.layers:
            results = []
            for neuron in layer:
                results.append(neuron.process(values))
            values = results

        for i in range(len(values)):
            values[i] = np.round(values[i])

        return values

    # private method for getting intermediate states of perceptron
    def __process_for_train(self, inputs):
        values = inputs
        interstates = [inputs]
        for layer in self.layers:
            interstate = []
            for neuron in layer:
                interstate.append(neuron.process(values))
            interstates.append(interstate)
            values = interstate

        return interstates

    def train(self, samples, targets, epochs, education_speed):
        for epoch in range(epochs):
            for s in range(len(samples)):

                #we start with reverse move
                sample = samples[s]
                target = targets[s]
                states = self.__process_for_train(sample)

                mistakes = []

                mistake = [] ##generate mistakes on the output layer
                state = states[-1]
                for i in range(len(state)):
                    mistake.append(state[i]*(1-state[i])*(target[i] - state[i]))
                mistakes.append(mistake)

                for i in range(len(states)-2, 0, -1):

                    state = states[i]
                    mistake = []

                    for j in range(len(state)):
                        m = 0
                        for w in range(len(mistakes[0])):
                            m += mistakes[0][w]*self.layers[i][w].get_weight(j)
                        mistake.append(state[j]*(1-state[j])*m)

                    mistakes.insert(0, mistake)

                ##now lets correct our weights with straight move
                for i in range(0, len(states)-1):
                    mistake = mistakes[i]
                    for j in range(len(mistake)):
                        weights = self.layers[i][j].get_weights()
                        for k in range(len(weights)-1):
                            weights[k] += education_speed*mistake[j]*states[i][k]
                        weights[-1] += education_speed*mistake[j]

                        self.layers[i][j].set_weights(weights)





    class Neuron:
        def __init__(self, weights):
            self.weights = weights

        def process(self, inputs):
            s = np.matmul(self.weights[:-1], inputs) + self.weights[-1]

            return 1.0/(1 + np.e**(-1*s))

        def set_weight(self, index, weight):
            self.weights[index] = weight

        def set_weights(self, weights):
            self.weights = weights

        def increment_weight(self, index, increment):
            self.weights[index] += increment

        def get_weight(self, index):
            return self.weights[index]

        def get_weights(self):
            return self.weights

#too slow + depends on significance of fine-function, not used for solving tasks
def svm(samples, targets): #needs number of dimensions of space and training set

    def scalarMult(x, y):
        r = 0.0
        for i in range(len(x)):
            r += x[i] * y[i]
        return r

    def H(w):
        r = 0.0
        for i in range(len(targets)):
            r += np.max([0, 1 - targets[i] * scalarMult(w, samples[i])]) ** 2
        return r

    def gradf(w):
        r = []
        for i in range(len(w)):
            v = 2 * w[i]
            for j in range(len(targets)):
                v += 100 * np.max([0, 1 - targets[j] * scalarMult(w, samples[j])]) * (-1 * targets[j] * samples[j][i])
            r.append(v)
        return np.array(r)

    def f(w):
        return scalarMult(w, w) + 50 * H(w)

    # теперь поехали градиентный спуск делать
    w0 = samples[-1]
    a = 1.0
    while scalarMult(gradf(w0), gradf(w0)) > 10 ** (-2) and a > 10 ** -8:
        control = gradf(w0)
        w1 = w0 - control * a
        if f(w1) < f(w0):
            w0 = w1
        else:
            a /= 2.0
            if a <= 10 ** -8:
                print('Gradient doesnt get lower')
                break

    #method returns coefs of devision plate (double array)
    return w0

def gradient(fun, der, x, e):

    def scalarMult(x, y):
        r = 0.0
        for i in range(len(x)):
            r += x[i] * y[i]
        return r

    a = 1.0
    while scalarMult(der(x), der(x)) > e and a > 10**(-8):
        grad = der(x)
        xn = x - grad*a
        if fun(xn) < fun(x):
            x = xn
        else:
            a /= 2.0
            if a <= 10 ** -8:
                print('Gradient doesnt get lower')
                break

    return x

#just for concrete task, not universal, don't use for specific purposes
def neuron_math(samples, targets, min, max, step):

    class Neuron:
        def __init__(self, weights):
            self.weights = weights

        def process(self, inputs):
            s = np.matmul(self.weights[:-1], inputs) + self.weights[-1]

            if s < 0: return 0
            else: return 1

        def set_weight(self, index, weight):
            self.weights[index] = weight

        def set_weights(self, weights):
            self.weights = weights

        def increment_weight(self, index, increment):
            self.weights[index] += increment

        def get_weight(self, index):
            return self.weights[index]

        def get_weights(self):
            return self.weights

    #метод производит перебор всех допустимых значений ко-в в заданных промежутках с заданным шагом, по факту изменения возвращает правда/ложь
    def nextValue(arr):
        for i in range(len(arr)-1, -1, -1):
            if arr[i] < max[i]:
                arr[i] += step[i]
                for j in range(i+1, len(arr)):
                    arr[j] = min[j]
                return True
        return False

    n1 = Neuron(np.copy(min))
    n2 = Neuron(np.copy(min))
    n3 = Neuron(np.copy(min[:-1]))

    while True:
        while True:
            while True:

                flag = True
                for i in range(len(targets)):
                    if n3.process([n1.process(samples[i]), n2.process(samples[i])]) - targets[i] != 0.0:
                        flag = False
                        break
                if flag: return [n1, n2, n3]

                if not nextValue(n3.get_weights()): break

            n3.set_weights(np.copy(min[:-1]))
            if not nextValue(n2.get_weights()): break

        n2.set_weights(np.copy(min))
        if not nextValue(n1.get_weights()): break
    print('Coeffs were not found')
    return [n1, n2, n3]
