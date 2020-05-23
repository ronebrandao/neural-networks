import numpy as np

inputs = [[-1, 0.1, 0.4, 0.7], [-1, 0.3, 0.7, 0.2], [-1, 0.6, 0.9, 0.8], [-1, 0.5, 0.7, 0.1]]
labels = [1, -1, -1, 1]
weights = [0, 0, 0, 0]

learningRate = 1
output = []
epochs = 0
maxEpochs = 1000

def sinal(elem):
    if elem > 0:
        return 1

    return -1


while len(output) < len(inputs) or epochs == maxEpochs:
    output = []
    epochs = epochs + 1

    for input, label in zip(inputs, labels):
        u = np.dot(input, weights)
        y = sinal(u)

        err = label - y

        if err != 0:
            for widx in range(len(inputs)):
                # w <- w + N * (d[i] - y) * x[i]
                weights[widx] = weights[widx] + learningRate * err * input[widx]
        else:
            output.append(y)


print("FINAL WEIGHTS", weights)



