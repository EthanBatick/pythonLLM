import numpy

def makeLayer(inputSize, nodeCount):
    # inputSize is the number of input nodes
    # nodeCount is the number of nodes in the layer
    layerList = []
    for i in range(nodeCount):
        weightList = [] #new blank weightlist for the new node
        for j in range(inputSize):
            weightList.append(1.0)
        layerList.append([weightList,0.0]) #appending the new node to the layer
    return layerList

def makeNetwork(listOfNodeCounts):
    # listOfNodeCounts is a list of integers, where each integer is the number of nodes in a layer
    # The first integer is the input layer size
    # The last integer is the number of output nodes
    # The integers in between are the number of nodes in hidden layers

    # THE OUTPUT NETWORK IS GOING TO HAVE ONE LESS LAYER THAN len(listOfNodeCounts) 
    # BECAUSE THE INPUT LAYER IS NOT A LAYER IN THE NETWORK

    network = []
    for i in range(1,len(listOfNodeCounts)):
        network.append(makeLayer(listOfNodeCounts[i-1],listOfNodeCounts[i])) # the next hidden layer added on
    return network.copy()

def rectLA(x):
    return max(0,x)

def crunchLayer(prevLayerOutput, currentLayer):
    # helper method for computeThrough() that does each layer individually
    outputThisCrunch = []
    for node in currentLayer:
        outputThisCrunch.append(rectLA(numpy.dot(prevLayerOutput,node[0])+node[1]))
        

def computeThrough(inputLayer, fullNetwork):
    # inputLayer is a list of values, where each value is an input node
    # fullNetwork is a list of lists of lists, where each 3rd degree list is the weights and biases (in tuple form "[weightList, bias]")
    # for a node

    # The output is a list of tuples, where each tuple is the output from a node
    if fullNetwork == []:
        return inputLayer
    else:
        output = []
        for node in fullNetwork[0]:
            output.append(rectLA(numpy.dot(inputLayer,node[0])+node[1]))
        return computeThrough(output,fullNetwork[1:])
    
