import tensorflow.compat.v1 as tf
import numpy as np
import urllib

tf.compat.v1.disable_eager_execution()

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Number of iterations to train for
numTrainingIters = 10000

windowLength = 10  # Number of characters in each window
numConvFilters = 8  # Number of convolutional filters

# Number of hidden neurons in each hidden layer
hiddenUnits1 = 1024  # First hidden layer
hiddenUnits2 = 512  # Second hidden layer

# Number of classes to predict
numClasses = 3

# Batch size
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



def apply_convolution(sequence, filters):
    conv_results = np.zeros(filters.shape[0])

    # Slide window
    for i in range(len(conv_results)):
        max_conv_val = float('-inf')
        for j in range(sequence.shape[0] - windowLength + 1):
            window = sequence[j:j + windowLength].flatten()
            conv_val = np.dot(window, filters[i])
            max_conv_val = max(max_conv_val, conv_val)
        conv_results[i] = max_conv_val

    return conv_results


# this function takes as input a data set encoded as a dictionary
# (same encoding as the last function) and pre-pends every line of
# text with empty characters so that each line of text is exactly
# maxSeqLen characters in size
def pad(maxSeqLen, data, filters):
    convolved_data = {}
    for i in data:
        #
        # access the matrix and the label
        temp = data[i][1]
        label = data[i][0]
        #
        # get the number of characters in this line
        len_seq = temp.shape[0]
        #
        # and then pad so the line is the correct length
        padding = np.zeros((maxSeqLen - len_seq, 256))
        padded = np.concatenate((padding, temp), axis=0)  # Pre-padding

        # Apply convolution
        convolved_features = apply_convolution(padded, filters)

        # Store the convolved features
        convolved_data[i] = (label, convolved_features)
    #
    # return the new data set
    return convolved_data


def generateDataFeedForward(maxSeqLen, data):
    myInts = np.random.randint(0, len(data), batchSize)
    x = np.stack([data[i][1] for i in myInts.flat])
    y = np.stack([data[i][0] for i in myInts.flat])
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

np.random.seed(42)
conv_filters = np.random.randn(numConvFilters, windowLength * 256)

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

# Pad and convolve both training and test data
trainData = pad(maxSeqLen, trainData, conv_filters)
testData = pad(maxSeqLen, testData, conv_filters)


inputX = tf.placeholder(tf.float32, [batchSize, numConvFilters], name='inputX')
inputY = tf.placeholder(tf.int32, [batchSize], name='inputY')


W1 = tf.Variable(tf.random_normal([numConvFilters, hiddenUnits1], stddev=0.01), name='W1')
b1 = tf.Variable(tf.zeros([hiddenUnits1]), name='b1')
hidden1 = tf.nn.relu(tf.matmul(inputX, W1) + b1)

W2 = tf.Variable(tf.random_normal([hiddenUnits1, hiddenUnits2], stddev=0.01), name='W2')
b2 = tf.Variable(tf.zeros([hiddenUnits2]), name='b2')
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

# Output Layer
W3 = tf.Variable(tf.random_normal([hiddenUnits2, numClasses], stddev=0.01), name='W3')
b3 = tf.Variable(tf.zeros([numClasses]), name='b3')
logits = tf.matmul(hidden2, W3) + b3

# Define loss and optimizer
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inputY)
totalLoss = tf.reduce_mean(losses)
optimizer = tf.train.AdagradOptimizer(0.01).minimize(totalLoss)

# Define predictions and accuracy
predictions = tf.nn.softmax(logits)
correctPredictions = tf.equal(tf.argmax(predictions, axis=1), tf.cast(inputY, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

init = tf.global_variables_initializer()

# and train!!
with tf.Session() as sess:
    sess.run(init)

    # To store the loss and accuracy for the last 20 iterations
    last20_losses = []
    last20_accuracies = []

    for epoch in range(1, numTrainingIters + 1):
        x_batch, y_batch = generateDataFeedForward(maxSeqLen, trainData)

        _, batch_loss, batch_accuracy = sess.run(
            [optimizer, totalLoss, accuracy],
            feed_dict={
                inputX: x_batch,
                inputY: y_batch
            }
        )

        if epoch > numTrainingIters - 20:
            last20_losses.append(batch_loss)
            last20_accuracies.append(batch_accuracy)

        if epoch % 1000 == 0 or epoch == 1:
            print(f"Iteration {epoch}: Loss = {batch_loss:.4f}, Accuracy = {batch_accuracy * 100:.2f}%")

    testBatchSize = 100
    numTestSamples = 3000
    numTestBatches = numTestSamples // testBatchSize
    totalTestLoss = 0.0
    totalTestCorrect = 0

    for i in range(numTestBatches):

        batchStart = i * testBatchSize
        batchEnd = batchStart + testBatchSize
        testBatchKeys = list(testData.keys())[batchStart:batchEnd]
        x_test = np.stack([testData[k][1] for k in testBatchKeys])
        y_test = np.stack([testData[k][0] for k in testBatchKeys])


        batch_loss, batch_accuracy = sess.run(
            [losses, correctPredictions],
            feed_dict={
                inputX: x_test,
                inputY: y_test
            }
        )


        totalTestLoss += np.sum(batch_loss)
        totalTestCorrect += np.sum(batch_accuracy)


    averageTestLoss = totalTestLoss / numTestSamples
    overallTestAccuracy = (totalTestCorrect / numTestSamples) * 100


    print("=== Training Complete ===")
    print(f"Last 20 Iterations Losses: {['{0:.4f}'.format(loss) for loss in last20_losses]}")
    print(f"Last 20 Iterations Accuracies: {['{0:.2f}%'.format(acc * 100) for acc in last20_accuracies]}")
    print(f"Test Set Loss: {averageTestLoss:.4f}")
    print(f"Test Set Accuracy: {overallTestAccuracy:.2f}%")

    # Investigate filter patterns
    print("=== Convolutional Filters Analysis ===")
    for i, filter_vals in enumerate(conv_filters):
        print(f"Filter {i} characteristics:")
        top_indices = np.argsort(filter_vals)[-10:]
        print("Top 10 character indices with highest dot product:")
        print(top_indices)