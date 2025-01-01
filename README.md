# CNN Model for MNIST Classification

This is a convolutional neural network (CNN) model implemented using PyTorch for classifying handwritten digits from the MNIST dataset. The model achieves an accuracy greater than **99.4%** on test dataset in less than **20 epochs** and has fewer than **20,000 parameters**, making it both efficient and accurate.

## Model Architecture

The model consists of the following layers:
- **Conv1**: Convolutional layer with 10 output channels, kernel size 3, and padding of 1.
- **Conv2**: Convolutional layer with 20 output channels, kernel size 3, and padding of 1.
- **Max Pooling (pool1)**: 2D Max Pooling with kernel size 2 and stride 2.
- **Conv3**: Convolutional layer with 10 output channels, kernel size 3, and padding of 1.
- **Conv4**: Convolutional layer with 10 output channels, kernel size 3, and padding of 1.
- **Max Pooling (pool2)**: 2D Max Pooling with kernel size 2 and stride 2.
- **Conv5**: Convolutional layer with 10 output channels, kernel size 3.
- **Conv6**: Convolutional layer with 10 output channels, kernel size 3.
- **Conv7**: Convolutional layer with 120 output channels, kernel size 3.
- **Fully Connected (fc)**: Linear layer that outputs 10 values (corresponding to the 10 classes in MNIST).

### Dropout Layers
Dropout layers are used throughout the model with a dropout rate of 10% to prevent overfitting during training.

## Performance

- **Test Accuracy**: Greater than **99.4%** on the MNIST test dataset.
- **Parameters**: The model has fewer than **20,000 parameters**, making it efficient in terms of model size and computational requirements.

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

