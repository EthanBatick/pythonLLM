import networkGenAndCompute as ngc
import copy


def computeLoss(network, data):
    loss = 0
    for dataPoint in data:
        output = copy.deepcopy(ngc.computeThrough(dataPoint[0], network))
        maxOutput = max(output)+0.0000001
        for i in range(len(output)):
            output[i] /= maxOutput
        loss += sum([(output[i]-dataPoint[1][i])**2 for i in range(len(output))])
    return loss

#USE ONE HOT ENCODING FOR ALL TRAINING
#DATA WILL BE PASSED AS A LIST OF TUPLES
#Each tuple will have two elements
#The first element will be a list of inputs
#The second element will be a list of expected outputs (one hot encoded)
#The number of elements in the input list must match the number of input nodes
#The number of elements in the output list must match the number of output nodes




#example data for XOR
practiceData = [([0,0],[1,0]),([0,1],[0,1]),([1,0],[0,1]),([1,1],[1,0])]

inputSize = len(practiceData[0][0])
outputSize = len(practiceData[0][1])

#NETWORK STRUCTURE
struc = [inputSize,10,10,outputSize]
##################

finalNetwork = ngc.makeNetwork(struc)

diff = .5

while diff > 0.1:
    changes = 11
    while changes > 10:

        changes = 0

        #weights
        for layerInd in range(len(finalNetwork)):
            for nodeInd in range(len(finalNetwork[layerInd])):
                for weightInd in range(len(finalNetwork[layerInd][nodeInd][0])):

                    #make the variations of the network
                    tryUp = copy.deepcopy(finalNetwork)
                    tryDown = copy.deepcopy(finalNetwork)
                    tryStay = copy.deepcopy(finalNetwork)
                    tryUp[layerInd][nodeInd][0][weightInd] += diff
                    tryDown[layerInd][nodeInd][0][weightInd] -= diff

                    #calculate the loss for each variation
                    upLoss = computeLoss(tryUp, practiceData)
                    downLoss = computeLoss(tryDown, practiceData)
                    stayLoss = computeLoss(tryStay, practiceData)
                    #print(min([upLoss, downLoss, stayLoss]), diff)
                    #print((upLoss, downLoss, stayLoss))

                    #if the loss is less for one of the variations, update the network
                    if upLoss < stayLoss and upLoss < downLoss:
                        finalNetwork = copy.deepcopy(tryUp)
                        changes += 1

                    elif downLoss < stayLoss and downLoss < upLoss:
                        finalNetwork = copy.deepcopy(tryDown)
                        changes += 1

        #biases
        for layerInd in range(len(finalNetwork)):
            for nodeInd in range(len(finalNetwork[layerInd])):
                for weightInd in range(len(finalNetwork[layerInd][nodeInd][0])):

                    #make the variations of the network
                    tryUp = copy.deepcopy(finalNetwork)
                    tryDown = copy.deepcopy(finalNetwork)
                    tryStay = copy.deepcopy(finalNetwork)
                    tryUp[layerInd][nodeInd][1] += diff
                    tryDown[layerInd][nodeInd][1] -= diff

                    #calculate the loss for each variation
                    upLoss = computeLoss(tryUp, practiceData)
                    downLoss = computeLoss(tryDown, practiceData)
                    stayLoss = computeLoss(tryStay, practiceData)

                    #print(min([upLoss, downLoss, stayLoss]), diff)
                    #print((upLoss, downLoss, stayLoss))

                    #if the loss is less for one of the variations, update the network
                    if upLoss < stayLoss and upLoss < downLoss:
                        finalNetwork = copy.deepcopy(tryUp)
                        changes += 1

                    elif downLoss < stayLoss and downLoss < upLoss:
                        finalNetwork = copy.deepcopy(tryDown)
                        changes += 1
        print(changes)
    print("diff:",diff, "loss:", computeLoss(finalNetwork, practiceData))
    diff /= 2

#test examples
print(ngc.computeThrough([0,0], finalNetwork))
print(ngc.computeThrough([0,1], finalNetwork))
print(ngc.computeThrough([1,0], finalNetwork))
print(ngc.computeThrough([1,1], finalNetwork))
