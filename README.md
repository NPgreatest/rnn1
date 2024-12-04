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

## Sample Result of RNN.py

```angular2html
Step 9989 Loss 0.8941822 Correct 52 out of 100
Step 9990 Loss 0.85714287 Correct 62 out of 100
Step 9991 Loss 0.9516181 Correct 46 out of 100
Step 9992 Loss 0.8229578 Correct 59 out of 100
Step 9993 Loss 0.9178457 Correct 52 out of 100
Step 9994 Loss 0.92544 Correct 54 out of 100
Step 9995 Loss 0.9578306 Correct 54 out of 100
Step 9996 Loss 0.91394764 Correct 56 out of 100
Step 9997 Loss 0.8548018 Correct 56 out of 100
Step 9998 Loss 0.91161484 Correct 49 out of 100
Step 9999 Loss 0.99025124 Correct 46 out of 100
```

## Sample Result of RNN2.py

```angular2html
Step 0 Loss 1.0982344 Correct 38 out of 100
Step 99 Loss 1.0990533 Correct 28 out of 100
Step 199 Loss 1.0988008 Correct 33 out of 100
Step 299 Loss 1.0980936 Correct 38 out of 100
Step 399 Loss 1.0981315 Correct 34 out of 100
Step 499 Loss 1.0980262 Correct 38 out of 100
Step 599 Loss 1.0992012 Correct 30 out of 100
Step 699 Loss 1.0982177 Correct 35 out of 100
Step 799 Loss 1.098663 Correct 33 out of 100
Step 899 Loss 1.0978649 Correct 36 out of 100
Step 999 Loss 1.0972435 Correct 38 out of 100
Step 1099 Loss 1.0985812 Correct 34 out of 100
Step 1199 Loss 1.0985538 Correct 34 out of 100
Step 1299 Loss 1.0980405 Correct 39 out of 100
Step 1399 Loss 1.0983216 Correct 38 out of 100
Step 1499 Loss 1.0985249 Correct 33 out of 100
Step 1599 Loss 1.0981117 Correct 41 out of 100
Step 1699 Loss 1.0991205 Correct 26 out of 100
Step 1799 Loss 1.0988781 Correct 35 out of 100
Step 1899 Loss 1.0982977 Correct 32 out of 100
Step 1999 Loss 1.0984768 Correct 35 out of 100
Step 2099 Loss 1.0986739 Correct 31 out of 100
Step 2199 Loss 1.0988882 Correct 31 out of 100
Step 2299 Loss 1.0981003 Correct 36 out of 100
Step 2399 Loss 1.0995277 Correct 29 out of 100
Step 2499 Loss 1.0970573 Correct 39 out of 100
Step 2599 Loss 1.0980613 Correct 33 out of 100
Step 2699 Loss 1.0984813 Correct 32 out of 100
Step 2799 Loss 1.098602 Correct 34 out of 100
Step 2899 Loss 1.097861 Correct 32 out of 100
Step 2999 Loss 1.0984517 Correct 33 out of 100
Step 3099 Loss 1.0984262 Correct 33 out of 100
Step 3199 Loss 1.0986503 Correct 31 out of 100
Step 3299 Loss 1.0981375 Correct 36 out of 100
Step 3399 Loss 1.098554 Correct 29 out of 100
Step 3499 Loss 1.0983381 Correct 31 out of 100
Step 3599 Loss 1.0991898 Correct 28 out of 100
Step 3699 Loss 1.0982169 Correct 35 out of 100
Step 3799 Loss 1.0975045 Correct 36 out of 100
Step 3899 Loss 1.0979544 Correct 35 out of 100
Step 3999 Loss 1.09828 Correct 35 out of 100
Step 4099 Loss 1.09827 Correct 35 out of 100
Step 4199 Loss 1.100017 Correct 27 out of 100
Step 4299 Loss 1.0995477 Correct 30 out of 100
Step 4399 Loss 1.0987415 Correct 44 out of 100
Step 4499 Loss 1.0981104 Correct 29 out of 100
Step 4599 Loss 1.0971028 Correct 38 out of 100
Step 4699 Loss 1.0978876 Correct 31 out of 100
Step 4799 Loss 1.0975962 Correct 35 out of 100
Step 4899 Loss 1.0981311 Correct 35 out of 100
Step 4999 Loss 1.0976707 Correct 44 out of 100
Step 5099 Loss 1.0984371 Correct 30 out of 100
Step 5199 Loss 1.098049 Correct 31 out of 100
Step 5299 Loss 1.0979567 Correct 34 out of 100
Step 5399 Loss 1.0975811 Correct 40 out of 100
Step 5499 Loss 1.0979954 Correct 28 out of 100
Step 5599 Loss 1.0959499 Correct 39 out of 100
Step 5699 Loss 1.0973632 Correct 36 out of 100
Step 5799 Loss 1.0973816 Correct 51 out of 100
Step 5899 Loss 1.0975325 Correct 32 out of 100
Step 5999 Loss 1.0967305 Correct 42 out of 100
Step 6099 Loss 1.0969627 Correct 32 out of 100
Step 6199 Loss 1.0966073 Correct 36 out of 100
Step 6299 Loss 1.0958519 Correct 37 out of 100
Step 6399 Loss 1.0927455 Correct 44 out of 100
Step 6499 Loss 1.093513 Correct 45 out of 100
Step 6599 Loss 1.0903836 Correct 43 out of 100
Step 6699 Loss 1.0848396 Correct 52 out of 100
Step 6799 Loss 1.0791459 Correct 49 out of 100
Step 6899 Loss 1.0606576 Correct 50 out of 100
Step 6999 Loss 1.0350548 Correct 47 out of 100
Step 7099 Loss 0.9435701 Correct 56 out of 100
Step 7199 Loss 0.99363685 Correct 51 out of 100
Step 7299 Loss 0.93108726 Correct 52 out of 100
Step 7399 Loss 0.87538433 Correct 59 out of 100
Step 7499 Loss 0.9117274 Correct 53 out of 100
Step 7599 Loss 0.92991364 Correct 50 out of 100
Step 7699 Loss 0.9406828 Correct 50 out of 100
Step 7799 Loss 0.9678236 Correct 49 out of 100
Step 7899 Loss 0.92468154 Correct 48 out of 100
Step 7999 Loss 0.856976 Correct 56 out of 100
Step 8099 Loss 0.95848006 Correct 45 out of 100
Step 8199 Loss 0.9132041 Correct 49 out of 100
Step 8299 Loss 0.9048245 Correct 56 out of 100
Step 8399 Loss 0.888523 Correct 55 out of 100
Step 8499 Loss 0.9009552 Correct 59 out of 100
Step 8599 Loss 0.9239778 Correct 58 out of 100
Step 8699 Loss 0.93146485 Correct 51 out of 100
Step 8799 Loss 0.93820274 Correct 58 out of 100
Step 8899 Loss 0.9421954 Correct 51 out of 100
Step 8999 Loss 0.9072086 Correct 56 out of 100
Step 9099 Loss 0.8811946 Correct 54 out of 100
Step 9199 Loss 0.82575125 Correct 59 out of 100
Step 9299 Loss 0.9893938 Correct 52 out of 100
Step 9399 Loss 0.8846419 Correct 54 out of 100
Step 9499 Loss 0.94012004 Correct 52 out of 100
Step 9599 Loss 0.91935855 Correct 52 out of 100
Step 9699 Loss 0.91689926 Correct 55 out of 100
Step 9799 Loss 0.9288069 Correct 55 out of 100
Step 9899 Loss 0.9327771 Correct 49 out of 100
Step 9999 Loss 0.9129634 Correct 58 out of 100
Loss for 3000 randomly chosen documents is 0.9149, number of correct labels is 1608 out of 3000
```


## Sample Result of RNN3.py

```angular2html
Step 0 Loss 1.099872 Correct 28 out of 100
Step 99 Loss 1.0988876 Correct 29 out of 100
Step 199 Loss 1.0981865 Correct 39 out of 100
Step 299 Loss 1.0977135 Correct 33 out of 100
Step 399 Loss 1.0999802 Correct 24 out of 100
Step 499 Loss 1.0980214 Correct 33 out of 100
Step 599 Loss 1.096695 Correct 33 out of 100
Step 699 Loss 1.0909448 Correct 39 out of 100
Step 799 Loss 1.092819 Correct 52 out of 100
Step 899 Loss 1.0945287 Correct 42 out of 100
Step 999 Loss 1.0933676 Correct 44 out of 100
Step 1099 Loss 1.087272 Correct 53 out of 100
Step 1199 Loss 1.08586 Correct 46 out of 100
Step 1299 Loss 1.0838754 Correct 42 out of 100
Step 1399 Loss 1.0565875 Correct 46 out of 100
Step 1499 Loss 0.57382584 Correct 70 out of 100
Step 1599 Loss 0.50999594 Correct 77 out of 100
Step 1699 Loss 0.524447 Correct 64 out of 100
Step 1799 Loss 0.6081777 Correct 64 out of 100
Step 1899 Loss 0.50243294 Correct 66 out of 100
Step 1999 Loss 0.38692182 Correct 78 out of 100
Step 2099 Loss 0.49083784 Correct 77 out of 100
Step 2199 Loss 0.45296684 Correct 79 out of 100
Step 2299 Loss 0.38013586 Correct 86 out of 100
Step 2399 Loss 0.41778094 Correct 82 out of 100
Step 2499 Loss 0.42993307 Correct 81 out of 100
Step 2599 Loss 0.40903404 Correct 83 out of 100
Step 2699 Loss 0.36014107 Correct 81 out of 100
Step 2799 Loss 0.47493902 Correct 79 out of 100
Step 2899 Loss 0.52155644 Correct 74 out of 100
Step 2999 Loss 0.51511735 Correct 68 out of 100
Step 3099 Loss 0.34551474 Correct 92 out of 100
Step 3199 Loss 0.4266561 Correct 85 out of 100
Step 3299 Loss 0.34042874 Correct 85 out of 100
Step 3399 Loss 0.40176696 Correct 85 out of 100
Step 3499 Loss 0.51074296 Correct 78 out of 100
Step 3599 Loss 0.2866212 Correct 90 out of 100
Step 3699 Loss 0.34491757 Correct 89 out of 100
Step 3799 Loss 0.31614187 Correct 85 out of 100
Step 3899 Loss 0.43173876 Correct 72 out of 100
Step 3999 Loss 0.27953112 Correct 89 out of 100
Step 4099 Loss 0.24874 Correct 89 out of 100
Step 4199 Loss 0.34916005 Correct 87 out of 100
Step 4299 Loss 0.32838166 Correct 86 out of 100
Step 4399 Loss 0.28703642 Correct 92 out of 100
Step 4499 Loss 0.38889852 Correct 86 out of 100
Step 4599 Loss 0.2920805 Correct 89 out of 100
Step 4699 Loss 0.33619198 Correct 90 out of 100
Step 4799 Loss 0.4108422 Correct 92 out of 100
Step 4899 Loss 0.27600288 Correct 90 out of 100
Step 4999 Loss 0.3551844 Correct 87 out of 100
Step 5099 Loss 0.3335621 Correct 93 out of 100
Step 5199 Loss 0.2796984 Correct 88 out of 100
Step 5299 Loss 0.368449 Correct 86 out of 100
Step 5399 Loss 0.2765218 Correct 90 out of 100
Step 5499 Loss 0.35004303 Correct 86 out of 100
Step 5599 Loss 0.32726803 Correct 90 out of 100
Step 5699 Loss 0.16626053 Correct 96 out of 100
Step 5799 Loss 0.28063512 Correct 88 out of 100
Step 5899 Loss 0.21180299 Correct 94 out of 100
Step 5999 Loss 0.29685122 Correct 89 out of 100
Step 6099 Loss 0.19503763 Correct 94 out of 100
Step 6199 Loss 0.285151 Correct 91 out of 100
Step 6299 Loss 0.3864403 Correct 85 out of 100
Step 6399 Loss 0.28364357 Correct 91 out of 100
Step 6499 Loss 0.16150059 Correct 97 out of 100
Step 6599 Loss 0.22524624 Correct 94 out of 100
Step 6699 Loss 0.2659482 Correct 89 out of 100
Step 6799 Loss 0.22307386 Correct 92 out of 100
Step 6899 Loss 0.108330645 Correct 97 out of 100
Step 6999 Loss 0.3924369 Correct 88 out of 100
Step 7099 Loss 0.1956816 Correct 96 out of 100
Step 7199 Loss 0.29133543 Correct 89 out of 100
Step 7299 Loss 0.368592 Correct 85 out of 100
Step 7399 Loss 0.33003733 Correct 88 out of 100
Step 7499 Loss 0.19516757 Correct 93 out of 100
Step 7599 Loss 0.32814395 Correct 87 out of 100
Step 7699 Loss 0.36313635 Correct 87 out of 100
Step 7799 Loss 0.20160685 Correct 92 out of 100
Step 7899 Loss 0.24577945 Correct 89 out of 100
Step 7999 Loss 0.2416502 Correct 91 out of 100
Step 8099 Loss 0.2941112 Correct 88 out of 100
Step 8199 Loss 0.25411427 Correct 91 out of 100
Step 8299 Loss 0.26164785 Correct 90 out of 100
Step 8399 Loss 0.19311717 Correct 90 out of 100
Step 8499 Loss 0.28512657 Correct 91 out of 100
Step 8599 Loss 0.22061996 Correct 93 out of 100
Step 8699 Loss 0.15318859 Correct 96 out of 100
Step 8799 Loss 0.3083164 Correct 92 out of 100
Step 8899 Loss 0.22311412 Correct 91 out of 100
Step 8999 Loss 0.22857648 Correct 90 out of 100
Step 9099 Loss 0.19275978 Correct 93 out of 100
Step 9199 Loss 0.16639215 Correct 93 out of 100
Step 9299 Loss 0.19173025 Correct 92 out of 100
Step 9399 Loss 0.17426395 Correct 92 out of 100
Step 9499 Loss 0.22748236 Correct 93 out of 100
Step 9599 Loss 0.29433525 Correct 91 out of 100
Step 9699 Loss 0.24244826 Correct 91 out of 100
Step 9799 Loss 0.28732777 Correct 87 out of 100
Step 9899 Loss 0.19196786 Correct 91 out of 100
Step 9999 Loss 0.200877 Correct 91 out of 100
Loss for 3000 randomly chosen documents is 0.2476, number of correct labels is 2751 out of 3000
```

## Sample Result of RNN4.py

```angular2html
Iteration 1: Loss = 1.0988, Accuracy = 32.00%
Iteration 1000: Loss = 0.5857, Accuracy = 75.00%
Iteration 2000: Loss = 0.3645, Accuracy = 88.00%
Iteration 3000: Loss = 0.1909, Accuracy = 92.00%
Iteration 4000: Loss = 0.1223, Accuracy = 96.00%
Iteration 5000: Loss = 0.1290, Accuracy = 96.00%
Iteration 6000: Loss = 0.0700, Accuracy = 98.00%
Iteration 7000: Loss = 0.0278, Accuracy = 100.00%
Iteration 8000: Loss = 0.0194, Accuracy = 100.00%
Iteration 9000: Loss = 0.0171, Accuracy = 100.00%
Iteration 10000: Loss = 0.0161, Accuracy = 100.00%
=== Training Complete ===
Last 20 Iterations Losses: ['0.0193', '0.0136', '0.0164', '0.0125', '0.0124', '0.0135', '0.0132', '0.0162', '0.0139', '0.0112', '0.0201', '0.0222', '0.0154', '0.0083', '0.0284', '0.0440', '0.0162', '0.0521', '0.0204', '0.0161']
Last 20 Iterations Accuracies: ['100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '99.00%', '100.00%', '100.00%', '99.00%', '99.00%', '100.00%', '99.00%', '100.00%', '100.00%']
Last 20 Prediction [array([0, 1, 0, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 0,
       0, 0, 0, 2, 1, 2, 2, 0, 0, 1, 0, 1, 1, 2, 1, 2, 0, 0, 0, 0, 2, 2,
       1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 2, 0, 1, 1, 1, 1, 0, 2, 2, 2,
       1, 0, 2, 0, 1, 1, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 1, 2, 0, 2, 0, 2,
       0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 1]), array([0, 1, 0, 0, 2, 2, 2, 0, 2, 0, 0, 1, 0, 2, 2, 0, 2, 2, 1, 0, 0, 0,
       1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 1,
       2, 1, 1, 2, 2, 2, 1, 2, 0, 1, 0, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1,
       1, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0, 1, 1, 2, 0, 2, 0, 2, 1, 1, 2, 2,
       2, 0, 2, 2, 1, 1, 2, 2, 0, 2, 0, 0]), array([2, 0, 0, 2, 0, 2, 0, 2, 2, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 0, 1, 1,
       1, 0, 0, 2, 0, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0, 2, 1, 1, 2, 0, 2, 2,
       2, 1, 1, 2, 1, 0, 1, 2, 0, 0, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 0,
       1, 0, 0, 1, 1, 1, 2, 0, 2, 1, 1, 2, 2, 1, 1, 0, 2, 2, 2, 0, 1, 0,
       0, 0, 0, 1, 0, 0, 1, 2, 1, 0, 1, 1]), array([2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 2, 2, 1, 2, 0, 2,
       0, 0, 0, 2, 1, 2, 1, 0, 1, 2, 0, 1, 1, 2, 0, 1, 1, 0, 0, 1, 1, 2,
       0, 1, 2, 1, 2, 2, 1, 1, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 1, 1, 2, 1,
       1, 2, 0, 0, 0, 2, 2, 1, 2, 1, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 0, 0,
       1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2]), array([1, 0, 1, 1, 2, 1, 1, 0, 2, 0, 2, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0,
       2, 2, 1, 0, 2, 0, 1, 1, 0, 1, 0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 0, 2,
       0, 0, 0, 2, 2, 0, 2, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1, 0, 0, 0, 1, 1,
       2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 2, 2, 0, 0, 2, 0, 1, 0, 0, 1, 2, 2,
       0, 0, 1, 0, 2, 0, 2, 2, 1, 0, 0, 2]), array([2, 2, 0, 0, 1, 0, 1, 1, 0, 2, 0, 2, 0, 2, 1, 0, 0, 0, 1, 2, 2, 2,
       0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 2, 1, 2, 2, 0, 2, 1,
       0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 1, 1, 2, 0, 0, 2, 2, 2, 2,
       1, 1, 2, 0, 2, 1, 0, 1, 1, 1, 2, 0, 2, 2, 0, 1, 1, 1, 2, 1, 1, 2,
       1, 1, 2, 0, 1, 2, 2, 2, 0, 0, 2, 0]), array([2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 0, 2, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
       0, 1, 1, 1, 0, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 1, 2, 1, 2, 2, 1,
       0, 2, 1, 1, 1, 2, 0, 0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 1, 0, 2, 2, 1,
       2, 2, 1, 0, 2, 0, 1, 0, 2, 2, 1, 2, 0, 1, 0, 0, 0, 1, 1, 0, 1, 2,
       0, 2, 1, 2, 2, 2, 1, 0, 2, 2, 2, 1]), array([0, 1, 1, 2, 0, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 0, 2, 2, 0,
       2, 2, 1, 1, 1, 2, 2, 1, 0, 2, 1, 2, 1, 0, 2, 2, 0, 0, 1, 2, 0, 0,
       2, 1, 2, 2, 1, 0, 1, 0, 0, 0, 1, 2, 1, 0, 2, 2, 0, 1, 1, 0, 0, 1,
       2, 0, 0, 2, 0, 2, 0, 2, 1, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 2,
       2, 2, 1, 2, 1, 0, 2, 0, 0, 0, 1, 0]), array([2, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 2, 1, 1, 0, 2, 1, 0, 2, 1, 1, 0,
       1, 2, 0, 0, 2, 2, 2, 0, 0, 2, 1, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2,
       1, 1, 0, 0, 2, 0, 2, 1, 2, 2, 0, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 1,
       2, 1, 1, 1, 1, 0, 2, 0, 2, 0, 0, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0,
       1, 0, 1, 0, 1, 0, 2, 1, 1, 1, 2, 2]), array([1, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 1, 2, 2, 2, 2,
       2, 2, 1, 0, 0, 1, 1, 1, 2, 2, 1, 0, 2, 0, 2, 0, 1, 1, 2, 2, 1, 0,
       2, 1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 0, 2, 1, 2, 0, 0,
       0, 2, 2, 2, 0, 1, 0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 2, 1, 0, 2, 1, 0,
       1, 0, 1, 1, 0, 2, 0, 2, 2, 0, 1, 2]), array([1, 2, 2, 0, 0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 2, 0, 0, 0, 2, 1,
       1, 2, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 0,
       2, 0, 2, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 0, 0, 1,
       0, 0, 1, 2, 1, 0, 1, 1, 0, 1, 0, 1, 2, 2, 0, 2, 0, 1, 2, 1, 0, 1,
       0, 0, 0, 2, 2, 2, 0, 2, 2, 1, 0, 0]), array([1, 1, 0, 0, 0, 2, 2, 0, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 0, 1, 1,
       2, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 0, 1, 1, 1, 2, 1, 2, 0, 2, 1, 1,
       0, 0, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 2, 1, 0, 2, 1, 0, 0,
       1, 1, 0, 2, 2, 0, 2, 2, 2, 1, 2, 2, 0, 0, 1, 2, 1, 2, 2, 2, 0, 1,
       2, 1, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0]), array([2, 2, 0, 2, 0, 2, 2, 0, 1, 2, 1, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2,
       0, 2, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 2, 2, 0, 2, 0, 2, 1, 1, 0,
       2, 1, 2, 1, 0, 2, 2, 2, 1, 2, 0, 1, 0, 2, 2, 1, 1, 2, 1, 1, 1, 0,
       1, 2, 0, 0, 1, 2, 1, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 1, 1, 1,
       1, 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 2]), array([1, 0, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 0, 2, 2, 0, 0, 1, 1, 1, 0,
       1, 0, 0, 0, 2, 2, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2, 2, 0, 2, 2, 2,
       2, 0, 2, 2, 1, 2, 2, 1, 2, 2, 2, 0, 1, 1, 2, 2, 0, 0, 0, 2, 1, 2,
       1, 1, 1, 1, 0, 2, 2, 1, 1, 2, 0, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 2,
       0, 0, 2, 2, 1, 2, 1, 0, 1, 2, 0, 2]), array([1, 1, 1, 2, 1, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 2, 0, 0, 2,
       2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 1, 0, 0, 2, 1, 1, 2, 1, 0, 2, 0,
       1, 2, 1, 2, 1, 1, 0, 2, 1, 0, 0, 2, 2, 2, 0, 1, 0, 0, 1, 0, 0, 0,
       2, 2, 2, 0, 1, 1, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 1, 2,
       0, 0, 1, 2, 1, 1, 2, 1, 1, 2, 0, 1]), array([2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2, 1, 0, 0, 0,
       1, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0,
       0, 1, 0, 2, 0, 1, 2, 0, 2, 2, 1, 1, 0, 2, 0, 2, 0, 1, 1, 1, 0, 1,
       2, 0, 0, 2, 0, 2, 1, 1, 0, 2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
       1, 0, 0, 2, 1, 0, 0, 1, 2, 1, 0, 0]), array([0, 0, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 0, 2, 0, 1, 0, 1, 2, 1, 2, 1, 0, 0, 2, 0, 1, 2, 2, 2, 0, 0, 2,
       0, 0, 1, 1, 2, 1, 0, 2, 2, 0, 0, 2, 0, 1, 1, 1, 2, 2, 2, 1, 0, 1,
       2, 2, 1, 0, 1, 2, 0, 1, 2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 2, 1, 1,
       2, 0, 1, 0, 0, 2, 1, 2, 0, 1, 0, 0]), array([0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 2, 1, 2,
       1, 0, 0, 1, 2, 1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 1, 0, 1, 2,
       2, 1, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 2, 2, 1, 2, 2, 0, 2, 1, 2,
       0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 2, 1, 0, 2, 0, 0, 2, 2, 0, 0, 0, 2,
       1, 2, 0, 1, 0, 0, 2, 1, 0, 1, 0, 1]), array([0, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 0, 2, 0, 2, 1, 2, 1, 2, 2, 0,
       2, 2, 1, 2, 0, 2, 0, 0, 0, 2, 1, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 2,
       1, 0, 2, 0, 0, 1, 2, 2, 2, 1, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0, 1, 1,
       0, 2, 1, 0, 2, 1, 2, 0, 0, 1, 0, 0, 1, 2, 2, 0, 2, 0, 2, 0, 0, 2,
       2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 1, 1]), array([0, 2, 1, 0, 2, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 2, 1, 0, 2, 2, 0,
       2, 2, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 2,
       1, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 0, 2, 2, 0, 2, 2, 1, 2, 2, 2, 0,
       0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 2, 2, 2, 0, 2, 1,
       0, 2, 1, 1, 1, 2, 1, 0, 0, 1, 1, 2])]
Test Set Loss: 0.4625
Test Set Accuracy: 88.67%
```

## Sample Result of RNN5.py

```angular2html
Iteration 1: Loss = 1.0989, Accuracy = 31.00%
Iteration 1000: Loss = 1.0819, Accuracy = 43.00%
Iteration 2000: Loss = 1.0832, Accuracy = 44.00%
Iteration 3000: Loss = 1.0657, Accuracy = 40.00%
Iteration 4000: Loss = 1.0933, Accuracy = 37.00%
Iteration 5000: Loss = 1.0666, Accuracy = 38.00%
Iteration 6000: Loss = 1.0732, Accuracy = 47.00%
Iteration 7000: Loss = 1.0432, Accuracy = 43.00%
Iteration 8000: Loss = 1.0233, Accuracy = 49.00%
Iteration 9000: Loss = 1.0236, Accuracy = 49.00%
Iteration 10000: Loss = 1.0259, Accuracy = 51.00%
=== Training Complete ===
Last 20 Iterations Losses: ['1.0609', '1.0962', '1.0400', '1.0342', '0.9832', '1.0030', '1.0874', '1.0136', '0.9607', '1.1002', '1.0077', '1.0089', '0.9947', '0.9947', '0.9681', '0.9844', '1.0432', '0.9799', '0.9652', '1.0259']
Last 20 Iterations Accuracies: ['47.00%', '39.00%', '45.00%', '49.00%', '58.00%', '50.00%', '43.00%', '46.00%', '55.00%', '39.00%', '46.00%', '48.00%', '45.00%', '47.00%', '55.00%', '48.00%', '50.00%', '47.00%', '55.00%', '51.00%']
Test Set Loss: 1.0278
Test Set Accuracy: 47.40%
=== Convolutional Filters Analysis ===
Filter 0 characteristics:
Top 10 character indices with highest dot product:
[1934  179 2035 2506 2521  478 2305 1957 1615  209]
Filter 1 characteristics:
Top 10 character indices with highest dot product:
[  14  241 2224 1487  681 2310 2437 1422 1156  335]
Filter 2 characteristics:
Top 10 character indices with highest dot product:
[2321  553   37  726 1618  104 2210  676 2103 1771]
Filter 3 characteristics:
Top 10 character indices with highest dot product:
[ 415  631   56 1578  194  192 2505  485 1046  568]
Filter 4 characteristics:
Top 10 character indices with highest dot product:
[1665 1371  217  986  341 2005 2353 2468 1224 1098]
Filter 5 characteristics:
Top 10 character indices with highest dot product:
[2408 1148 2183  113  369  406 2112  644 1146  425]
Filter 6 characteristics:
Top 10 character indices with highest dot product:
[ 948 2259  613  843  127 2267 1549 2020 2076  483]
Filter 7 characteristics:
Top 10 character indices with highest dot product:
[1459 1664  212 1318  498 1570  730  443 2206  931]
```
