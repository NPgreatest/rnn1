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

# Number of hidden neurons in each hidden layer
# You can experiment with these values
hiddenUnits1 = 1024*2 # First hidden layer
hiddenUnits2 = 512 *2  # Second hidden layer (set to None if using only one hidden layer)

# Number of classes to predict
numClasses = 3

# Batch size
batchSize = 100


def addToData(maxSeqLen, data, fileName, classNum, desired_num, shuffledIndices, start_idx=0):
    """
    Adds a specific number of valid lines to the data dictionary from shuffled indices.

    Parameters:
    - maxSeqLen: Current maximum sequence length.
    - data: Dictionary to add data to.
    - fileName: URL of the text file.
    - classNum: Class label for the lines.
    - desired_num: Number of valid lines to add.
    - shuffledIndices: Shuffled list of line indices.
    - start_idx: Starting index in shuffledIndices.

    Returns:
    - Updated maxSeqLen, data dictionary, and next starting index.
    """
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
        # Update maxSeqLen if necessary
        if len(line) > maxSeqLen:
            maxSeqLen = len(line)
        # Create one-hot encoding matrix
        temp = np.zeros((len(line), 256))
        j = 0
        for ch in line:
            if ord(ch) >= 256:
                continue
            temp[j][ord(ch)] = 1
            j += 1
        # If no valid characters, skip
        if j == 0:
            idx += 1
            continue
        # Add to data
        data[i] = (classNum, temp)
        i += 1
        count += 1
        idx += 1

    if count < desired_num:
        raise ValueError(f"Not enough valid lines in {fileName}. Required: {desired_num}, Found: {count}")

    return maxSeqLen, data, idx


def pad(maxSeqLen, data):
    """
    Pads each data sample with zeros so that all have the same sequence length.

    Parameters:
    - maxSeqLen: The maximum sequence length in the dataset.
    - data: Dictionary containing the data samples.

    Returns:
    - Padded data dictionary.
    """
    for i in data:
        # Access the matrix and the label
        temp = data[i][1]
        label = data[i][0]
        # Get the number of characters in this line
        len_seq = temp.shape[0]
        # Pad so the line is the correct length
        padding = np.zeros((maxSeqLen - len_seq, 256))
        padded = np.concatenate((padding, temp), axis=0)  # Pre-padding
        data[i] = (label, padded.flatten())  # Flatten to (1, maxSeqLen * 256)
    return data


def generateDataFeedForward(maxSeqLen, data):
    """
    Generates a new batch of training data for the FFNN.

    Parameters:
    - maxSeqLen: The maximum sequence length.
    - data: Dictionary containing the data samples.

    Returns:
    - Tuple of (x, y) where:
        x is a NumPy array of shape [batchSize, 256 * maxSeqLen]
        y is a NumPy array of shape [batchSize]
    """
    # Randomly sample batchSize lines of text
    myInts = np.random.randint(0, len(data), batchSize)
    # Stack all of the text into a matrix of one-hot characters
    x = np.stack([data[i][1] for i in myInts.flat])
    # Stack all of the labels into a vector of labels
    y = np.stack([data[i][0] for i in myInts.flat])
    return (x, y)


# Initialize separate data dictionaries for training and testing
trainData = {}
testData = {}
maxSeqLen = 0

# List of input files and their class numbers
files = [
    ("https://s3.amazonaws.com/chrisjermainebucket/text/Holmes.txt", 0),
    ("https://s3.amazonaws.com/chrisjermainebucket/text/war.txt", 1),
    ("https://s3.amazonaws.com/chrisjermainebucket/text/william.txt", 2)
]

# Number of lines to use for training and testing per file
trainLinesPerFile = 9000
testLinesPerFile = 1000

# For each file, split the data into training and test sets
for fileName, classNum in files:
    # Open the file and read all lines
    response = urllib.request.urlopen(fileName)
    content = response.readlines()
    totalLines = len(content)

    if totalLines < trainLinesPerFile + testLinesPerFile:
        raise ValueError(
            f"Not enough lines in {fileName}. Required: {trainLinesPerFile + testLinesPerFile}, Found: {totalLines}")

    # Generate a random permutation of line indices
    shuffledIndices = np.random.permutation(totalLines)

    # Select test and training indices
    testIndices = shuffledIndices[:testLinesPerFile]
    trainIndices = shuffledIndices[testLinesPerFile:trainLinesPerFile + testLinesPerFile]

    # Add test lines
    maxSeqLen, testData, next_idx = addToData(
        maxSeqLen, testData, fileName, classNum, testLinesPerFile, shuffledIndices, start_idx=0
    )

    # Add training lines
    maxSeqLen, trainData, next_idx = addToData(
        maxSeqLen, trainData, fileName, classNum, trainLinesPerFile, shuffledIndices, start_idx=next_idx
    )

# Pad both training and test data
trainData = pad(maxSeqLen, trainData)
testData = pad(maxSeqLen, testData)

# Define placeholders for input and output
inputX = tf.placeholder(tf.float32, [batchSize, 256 * maxSeqLen], name='inputX')
inputY = tf.placeholder(tf.int32, [batchSize], name='inputY')

# Define the network architecture
# You can experiment with the number of hidden layers and neurons
# Here, we provide two configurations: one with a single hidden layer and another with two hidden layers

W1 = tf.Variable(tf.random_normal([256 * maxSeqLen, hiddenUnits1], stddev=0.01), name='W1')
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

# Define optimizer
optimizer = tf.train.AdagradOptimizer(0.01).minimize(totalLoss)

# Define predictions and accuracy
predictions = tf.nn.softmax(logits)
correctPredictions = tf.equal(tf.argmax(predictions, axis=1), tf.cast(inputY, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

# Initialize variables
init = tf.global_variables_initializer()

# Training and Evaluation
with tf.Session() as sess:
    sess.run(init)

    # To store the loss and accuracy for the last 20 iterations
    last20_losses = []
    last20_accuracies = []
    last20_predictions = []

    for epoch in range(1, numTrainingIters + 1):
        # Generate a batch of training data
        x_batch, y_batch = generateDataFeedForward(maxSeqLen, trainData)

        # Run the optimizer and compute loss and accuracy
        _, batch_loss, batch_accuracy,batch_predictions = sess.run(
            [optimizer, totalLoss, accuracy, predictions],
            feed_dict={
                inputX: x_batch,
                inputY: y_batch
            }
        )

        # Store the loss and accuracy
        if epoch > numTrainingIters - 20:
            last20_losses.append(batch_loss)
            last20_accuracies.append(batch_accuracy)
            last20_predictions.append(np.argmax(batch_predictions, axis=1))

        # Print progress every 1000 iterations
        if epoch % 1000 == 0 or epoch == 1:
            print(f"Iteration {epoch}: Loss = {batch_loss:.4f}, Accuracy = {batch_accuracy * 100:.2f}%")

    # After training, evaluate on the test set
    testBatchSize = 100  # Define batch size for testing
    numTestSamples = 3000
    numTestBatches = numTestSamples // testBatchSize
    totalTestLoss = 0.0
    totalTestCorrect = 0

    for i in range(numTestBatches):
        # Get batch data
        batchStart = i * testBatchSize
        batchEnd = batchStart + testBatchSize
        testBatchKeys = list(testData.keys())[batchStart:batchEnd]
        x_test = np.stack([testData[k][1] for k in testBatchKeys])
        y_test = np.stack([testData[k][0] for k in testBatchKeys])

        # Compute loss and predictions
        batch_loss, batch_accuracy = sess.run(
            [losses, correctPredictions],
            feed_dict={
                inputX: x_test,
                inputY: y_test
            }
        )

        # Accumulate loss and correct predictions
        totalTestLoss += np.sum(batch_loss)
        totalTestCorrect += np.sum(batch_accuracy)

    # Compute average loss and overall accuracy
    averageTestLoss = totalTestLoss / numTestSamples
    overallTestAccuracy = (totalTestCorrect / numTestSamples) * 100

    # Print the final evaluation message
    print("\n=== Training Complete ===")
    print(f"Last 20 Iterations Losses: {['{0:.4f}'.format(loss) for loss in last20_losses]}")
    print(f"Last 20 Iterations Accuracies: {['{0:.2f}%'.format(acc * 100) for acc in last20_accuracies]}")
    print(f"Last 20 Prediction {last20_predictions}")
    print(f"Test Set Loss: {averageTestLoss:.4f}")
    print(f"Test Set Accuracy: {overallTestAccuracy:.2f}%")
