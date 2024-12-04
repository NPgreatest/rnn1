# Deep Learning for Text Classification using Convolutional RNN

## Overview

This project involves implementing a text classification model using Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The goal is to classify raw text sequences into one of three categories based on the source file. The model leverages convolutional filters to extract features from character-level sequences and uses dense layers for classification. TensorFlow 1.x is used to build and train the model.

The following text files are used as inputs:
- **Holmes.txt**: Sherlock Holmes novels
- **war.txt**: Text related to the war (fictional or historical)
- **william.txt**: Text from works by William Shakespeare

## Setup Instructions

### Requirements:
- **TensorFlow (1.x)**: The code uses TensorFlow's 1.x API. We specifically disable eager execution for compatibility.
- **NumPy**: For data manipulation.
- **Python**: The code should be executed in Python 3.6+.

### Hardware:
- **GPU**: A GPU is strongly recommended for faster training. You can run this code on AWS EC2 or Google Colab for GPU support.

### Installation:
You can install the required dependencies by running:

```
pip install tensorflow==1.15.0 numpy
```


### Files:
- **Data**: The code downloads text data directly from the provided URLs.
- **Model Architecture**: The model architecture is composed of convolutional layers followed by fully connected layers.

### Running the Code:
1. Clone or download this repository.
2. Run the Python script to start training the model on the text data.
3. The model will train for 10,000 iterations and output the loss and accuracy every 1,000 iterations.
4. After training, the test set will be evaluated to check performance.

## Code Explanation

### 1. **Data Preprocessing**:
   The code loads text data from URLs (Holmes, War, and William) and prepares it for training by encoding characters as one-hot vectors. Each line of text is padded to the same length (`maxSeqLen`), and convolutional filters are applied to extract relevant features.

### 2. **Convolutional Layer**:
   The `apply_convolution` function performs a 1D convolution with the provided filters over each line of text, sliding a window over the sequence and applying dot products with the filters.

### 3. **Model Architecture**:
   The model consists of:
   - **Convolutional Filters**: Applied to text data to extract features.
   - **Dense Layers**: Two hidden layers with ReLU activations, followed by an output layer that classifies the data into one of three classes.

### 4. **Training and Evaluation**:
   - The model is trained using **Adagrad Optimizer** and **cross-entropy loss**.
   - Training results are reported every 1,000 iterations.
   - After training, the model is evaluated on a test set.

### 5. **Convolutional Filter Analysis**:
   - After training, the characteristics of each convolutional filter are printed, showing the top 10 character indices with the highest dot products.

## Key Functions

- **addToData**: Loads and processes text data from a file, converting it into one-hot encoded vectors. It also handles padding the sequences to the maximum length (`maxSeqLen`).
  
- **apply_convolution**: Applies the convolutional filters to the sequences of text, sliding over each window and applying dot products.

- **pad**: Pre-pads each line of text to ensure all sequences are of the same length, then applies the convolutional filters.

- **generateDataFeedForward**: Generates random batches from the training data for feeding into the neural network.

## Hyperparameters

- `numTrainingIters`: Number of iterations for training (10,000 iterations).
- `windowLength`: The size of the convolutional window (10 characters).
- `numConvFilters`: Number of convolutional filters (8 filters).
- `hiddenUnits1` and `hiddenUnits2`: Number of neurons in the two hidden layers (1024 and 512, respectively).
- `numClasses`: Number of output classes (3, corresponding to three source text files).
- `batchSize`: Number of samples per batch during training (100).

## Training Output

During training, the model will print the following every 1,000 iterations:


```
teration <epoch>: Loss = <loss_value>, Accuracy = <accuracy_value>%
```


At the end of training, the following results will be printed:
- **Losses and accuracies** for the last 20 iterations.
- **Test set loss** and **accuracy** after evaluating the model on a held-out test set of 3,000 samples.

Additionally, the model will print the **Convolutional Filters Analysis** showing the top 10 character indices for each filter.

## Conclusion

This project demonstrates the implementation of a convolutional recurrent network for text classification. By applying convolutional filters to character-level sequences, the model can extract meaningful features and classify the text based on its source. The results can be used to analyze which characters or character patterns are most indicative of a given text source.

## Future Enhancements:
- **Model Optimization**: Experiment with other optimizers such as Adam or RMSprop.
- **More Filters**: Increase the number of filters or experiment with different window sizes.
- **Different Architectures**: Implement more advanced architectures like LSTM or GRU for sequence learning.
