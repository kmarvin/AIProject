import matplotlib.pyplot as plt


def plotLineData(header, yLabel, firstData, secondData, firstLabel, secondLabel, firstColor='b', secondColor='darkorange', xLabel='epoch'):
    plt.plot(firstData, color=firstColor)
    plt.plot(secondData, color=secondColor)
    plt.title(header)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend([firstLabel, secondLabel], loc='upper left')
    plt.show()

# example for plotLineData
plotLineData("test", "test1", [2, 4, 8, 11], [1, 2, 3, 4], "first", "second", firstColor='g')