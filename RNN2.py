import tensorflow.compat.v1 as tf
import numpy as np
import urllib

tf.compat.v1.disable_eager_execution()

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# the number of iterations to train for
numTrainingIters = 10000

# the number of hidden neurons that hold the state of the RNN
hiddenUnits = 1000

# the number of classes that we are learning over
numClasses = 3

# the number of data points in a batch
batchSize = 100

# this function takes a dictionary (called data) which contains
# of (dataPointID, (classNumber, matrix)) entries.  Each matrix
# is a sequence of vectors; each vector has a one-hot-encoding of
# an ascii character, and the sequence of vectors corresponds to
# one line of text.  classNumber indicates which file the line of
# text came from.
#
# The argument maxSeqLen is the maximum length of a line of text
# seen so far.  fileName is the name of a file whose contents
# we want to add to data.  classNum is an indicator of the class
# we are going to associate with text from that file.  linesToUse
# tells us how many lines to sample from the file.
#
# The return val is the new maxSeqLen, as well as the new data
# dictionary with the additional lines of text added
def addToData(maxSeqLen, data, fileName, classNum, desired_num, shuffledIndices, start_idx=0):
    response = urllib.request.urlopen(fileName)
    content = response.readlines()
    totalLines = len(content)

    i = len(data)
    count = 0
    idx = start_idx

    while count < desired_num and idx < totalLines:
        whichLine = shuffledIndices[idx]
        line = content[whichLine].decode("utf-8")
        if line.isspace() or len(line) == 0:
            idx += 1
            continue

        if len(line) > maxSeqLen:
            maxSeqLen = len(line)

        temp = np.zeros((len(line), 256))
        j = 0
        for ch in line:
            if ord(ch) >= 256:
                continue
            temp[j][ord(ch)] = 1
            j += 1

        if j == 0:
            idx += 1
            continue

        data[i] = (classNum, temp)
        i += 1
        count += 1
        idx += 1

    if count < desired_num:
        raise ValueError(f"Not enough valid lines in {fileName}. Required: {desired_num}, Found: {count}")

    return maxSeqLen, data, idx


# this function takes as input a data set encoded as a dictionary
# (same encoding as the last function) and pre-pends every line of
# text with empty characters so that each line of text is exactly
# maxSeqLen characters in size
def pad (maxSeqLen, data):
   for i in data:
        #
        # access the matrix and the label
        temp = data[i][1]
        label = data[i][0]
        #
        # get the number of chatacters in this line
        len = temp.shape[0]
        #
        # and then pad so the line is the correct length
        padding = np.zeros ((maxSeqLen - len,256))
        data[i] = (label, np.transpose (np.concatenate ((padding, temp), axis = 0)))
   #
   # return the new data set
   return data


# this generates a new batch of training data of size batchSize from the
# list of lines of text data. This version of generateData is useful for
# an RNN because the data set x is a NumPy array with dimensions
# [batchSize, 256, maxSeqLen]; it can be unstacked into a series of
# matrices containing one-hot character encodings for each data point
# using tf.unstack(inputX, axis=2)
def generateDataRNN(maxSeqLen, data):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.randint(0, len(data), batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack([data[i][1] for i in myInts.flat])
    #
    # and stack all of the labels into a vector of labels
    y = np.stack([np.array((data[i][0])) for i in myInts.flat])
    #
    # return the pair
    return (x, y)


# this also generates a new batch of training data, but it represents
# the data as a NumPy array with dimensions [batchSize, 256 * maxSeqLen]
# where for each data point, all characters have been appended.  Useful
# for feed-forward network training
def generateDataFeedForward(maxSeqLen, data):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.randint(0, len(data), batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack(data[i][1].flatten() for i in myInts.flat)
    #
    # and stack all of the labels into a vector of labels
    y = np.stack(np.array((data[i][0])) for i in myInts.flat)
    #
    # return the pair
    return (x, y)


trainData = {}
testData = {}
maxSeqLen = 0


files = [
    ("https://s3.amazonaws.com/chrisjermainebucket/text/Holmes.txt", 0),
    ("https://s3.amazonaws.com/chrisjermainebucket/text/war.txt", 1),
    ("https://s3.amazonaws.com/chrisjermainebucket/text/william.txt", 2)
]

trainLinesPerFile = 9000
testLinesPerFile = 1000

for fileName, classNum in files:
    response = urllib.request.urlopen(fileName)
    content = response.readlines()
    totalLines = len(content)

    if totalLines < trainLinesPerFile + testLinesPerFile:
        raise ValueError(
            f"Not enough lines in {fileName}. Required: {trainLinesPerFile + testLinesPerFile}, Found: {totalLines}")

    shuffledIndices = np.random.permutation(totalLines)

    testIndices = shuffledIndices[:testLinesPerFile]
    trainIndices = shuffledIndices[testLinesPerFile:trainLinesPerFile + testLinesPerFile]

    maxSeqLen, testData, next_idx = addToData(
        maxSeqLen, testData, fileName, classNum, testLinesPerFile, shuffledIndices, start_idx=0
    )

    maxSeqLen, trainData, next_idx = addToData(
        maxSeqLen, trainData, fileName, classNum, trainLinesPerFile, shuffledIndices, start_idx=next_idx
    )

# Pad both training and test data
trainData = pad(maxSeqLen, trainData)
testData = pad(maxSeqLen, testData)

# Now we build the TensorFlow computation... there are two inputs,
# a batch of text lines and a batch of labels
inputX = tf.placeholder(tf.float32, [batchSize, 256, maxSeqLen])
inputY = tf.placeholder(tf.int32, [batchSize])

# the initial state of the RNN, before processing any data
initialState = tf.placeholder(tf.float32, [batchSize, hiddenUnits])

# the weight matrix that maps the inputs and hidden state to a set of values
Wfir = tf.Variable(np.random.normal(0, 0.01, (hiddenUnits + 256, hiddenUnits)), dtype=tf.float32)
Wsec = tf.Variable(np.random.normal(0, 0.01, (hiddenUnits, hiddenUnits)), dtype=tf.float32)

# weights and bias for the final classification
W2 = tf.Variable(np.random.normal(0, 0.05, (hiddenUnits, numClasses)), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, numClasses)), dtype=tf.float32)

# unpack the input sequences so that we have a series of matrices,
# each of which has a one-hot encoding of the current character from
# every input sequence
sequenceOfLetters = tf.unstack(inputX, axis=2)

# now we implement the forward pass
currentState = initialState
for timeTick in sequenceOfLetters:
    #
    # concatenate the state with the input, then compute the next state
    inputPlusState = tf.concat([timeTick, currentState], 1)
    next_state = tf.tanh(tf.matmul(inputPlusState, Wfir))
    last_state = tf.tanh(tf.matmul(next_state, Wsec))
    currentState = last_state

# compute the set of outputs
outputs = tf.matmul(currentState, W2) + b2

predictions = tf.nn.softmax(outputs)

# compute the loss
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputY)
totalLoss = tf.reduce_mean(losses)

# use gradient descent to train
trainingAlg = tf.compat.v1.train.AdagradOptimizer(0.01).minimize(totalLoss)

# and train!!
with tf.Session() as sess:
    #
    # initialize everything
    sess.run(tf.compat.v1.global_variables_initializer())
    #
    # and run the training iters
    for epoch in range(numTrainingIters):
        #
        # get some data
        x, y = generateDataRNN(maxSeqLen, trainData)
        #
        # do the training epoch
        _currentState = np.zeros((batchSize, hiddenUnits))
        _totalLoss, _trainingAlg, _currentState, _predictions, _outputs = sess.run(
            [totalLoss, trainingAlg, currentState, predictions, outputs],
            feed_dict={
                inputX: x,
                inputY: y,
                initialState: _currentState
            })


        numCorrect = np.sum(np.argmax(_predictions, axis=1) == y)
        #
        # print out to the screen
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print("Step", epoch, "Loss", _totalLoss, "Correct", numCorrect, "out of", batchSize)

    # ------------------------------------------------------
    testBatchSize = 100
    totalTestLoss = 0.0
    totalTestCorrect = 0
    numTestSamples = 3000
    numTestBatches = numTestSamples // testBatchSize

    for i in range(numTestBatches):

        batchStart = i * testBatchSize
        batchEnd = batchStart + testBatchSize
        testBatchKeys = list(testData.keys())[batchStart:batchEnd]
        x_test = np.stack([testData[k][1] for k in testBatchKeys])
        y_test = np.stack([testData[k][0] for k in testBatchKeys])

        _currentState = np.zeros((testBatchSize, hiddenUnits))

        _testLoss, _testPredictions = sess.run(
            [losses, predictions],
            feed_dict={
                inputX: x_test,
                inputY: y_test,
                initialState: _currentState
            })

        totalTestLoss += np.sum(_testLoss)
        totalTestCorrect += np.sum(np.argmax(_testPredictions, axis=1) == y_test)

    averageTestLoss = totalTestLoss / numTestSamples

    print(
        f"Loss for 3000 randomly chosen documents is {averageTestLoss:.4f}, number of correct labels is {totalTestCorrect} out of 3000")
