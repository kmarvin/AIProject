import matplotlib.pyplot as plt

def plotLineData(header, yLabel, data, labels, colors=["b", "darkorange"], xLabel='epoch'):
    i = 0
    for d in data:
        plt.plot(data, color=colors[i])
        i += 1
    plt.title(header)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend(labels, loc='upper left')
    plt.show()

# example for plotLineData
plotLineData("test", "test1", [[2, 4, 8, 11], [1, 2, 3, 4]], ["first", "second"])
