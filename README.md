# CNN Model for MNIST Classification

This is a convolutional neural network (CNN) model implemented using PyTorch for classifying handwritten digits from the MNIST dataset. The model achieves an accuracy greater than **99.4%** on test dataset in less than **20 epochs** and has fewer than **20,000 parameters**, making it both efficient and accurate.

Number of parameters=18,970


## Model Architecture

The architecture of the model consists of the following layers:

1. **Conv1**: A 2D convolutional layer with 10 filters of size 3x3 and padding of 1.
   - **BatchNorm1**: Batch normalization is applied to the output of Conv1.
   
2. **Conv2**: A 2D convolutional layer with 20 filters of size 3x3 and padding of 1.
   - **BatchNorm2**: Batch normalization is applied to the output of Conv2.

3. **MaxPool1**: A max pooling layer with kernel size 2x2 and stride 2.

4. **Conv3**: A 2D convolutional layer with 10 filters of size 3x3 and padding of 1.
   - **BatchNorm3**: Batch normalization is applied to the output of Conv3.

5. **Conv4**: A 2D convolutional layer with 10 filters of size 3x3 and padding of 1.
   - **BatchNorm4**: Batch normalization is applied to the output of Conv4.

6. **MaxPool2**: A second max pooling layer with kernel size 2x2 and stride 2.

7. **Conv5**: A 2D convolutional layer with 10 filters of size 3x3.
   - **BatchNorm5**: Batch normalization is applied to the output of Conv5.

8. **Conv6**: A 2D convolutional layer with 10 filters of size 3x3.
   - **BatchNorm6**: Batch normalization is applied to the output of Conv6.

9. **Conv7**: A 2D convolutional layer with 120 filters of size 3x3.
   - **BatchNorm7**: Batch normalization is applied to the output of Conv7.

10. **Fully Connected Layer**: A fully connected (FC) layer that outputs 10 classes.

11. **Dropout**: A dropout layer with a probability of 0.1 applied after each major operation to prevent overfitting.

## Forward Pass

The forward pass of the network applies the following operations sequentially:

1. Convolution -> BatchNorm -> ReLU -> Pooling (for Conv1, Conv2, Conv3, Conv4)
2. Dropout after each major operation
3. Final Convolution -> Fully Connected Layer -> LogSoftmax



## Performance

- **Test Accuracy**: Greater than **99.4%** on the MNIST test dataset.
- **Parameters**: The model has fewer than **20,000 parameters**, making it efficient in terms of model size and computational requirements.
- Achieves a test accuracy of 99.47% at epoch 12

## Training Details

- **Dataset**: MNIST (28x28 grayscale images of handwritten digits).
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum 0.9 and a learning rate of 0.025.
- **Learning Rate Scheduler**: StepLR with a decay factor of 0.15 every 9 epochs.
- **Loss Function**: Negative Log-Likelihood Loss (NLLLoss).
- **Batch Size**: 128.
- **Epochs**: Trained for a maximum of 20 epochs.

### Data Augmentation
The model uses random rotation as a form of data augmentation to improve generalization:
- Random rotation between -7.0 and 7.0 degrees.
  
## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm

