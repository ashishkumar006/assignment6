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

Epoch 1, Learning Rate: 0.025
loss=0.12566810846328735 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 18.91it/s]

Test set: Average loss: 0.0634, Accuracy: 9786/10000 (98%)

Epoch 2, Learning Rate: 0.025
loss=0.12301745265722275 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.13it/s]

Test set: Average loss: 0.0417, Accuracy: 9871/10000 (99%)

Epoch 3, Learning Rate: 0.025
loss=0.16032250225543976 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.15it/s]

Test set: Average loss: 0.0365, Accuracy: 9886/10000 (99%)

Epoch 4, Learning Rate: 0.025
loss=0.13728077709674835 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.21it/s]

Test set: Average loss: 0.0335, Accuracy: 9893/10000 (99%)

Epoch 5, Learning Rate: 0.025
loss=0.2434483915567398 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.20it/s]

Test set: Average loss: 0.0291, Accuracy: 9899/10000 (99%)

Epoch 6, Learning Rate: 0.025
loss=0.22861331701278687 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.15it/s]

Test set: Average loss: 0.0305, Accuracy: 9907/10000 (99%)

Epoch 7, Learning Rate: 0.025
loss=0.11394951492547989 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.41it/s]

Test set: Average loss: 0.0302, Accuracy: 9907/10000 (99%)

Epoch 8, Learning Rate: 0.025
loss=0.10457319766283035 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.23it/s]

Test set: Average loss: 0.0234, Accuracy: 9928/10000 (99%)

Epoch 9, Learning Rate: 0.025
loss=0.14934244751930237 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.22it/s]

Test set: Average loss: 0.0232, Accuracy: 9928/10000 (99%)

Epoch 10, Learning Rate: 0.00375
loss=0.19503778219223022 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.22it/s]

Test set: Average loss: 0.0197, Accuracy: 9942/10000 (99%)

Epoch 11, Learning Rate: 0.00375
loss=0.11391881853342056 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.36it/s]

Test set: Average loss: 0.0196, Accuracy: 9943/10000 (99%)

Epoch 12, Learning Rate: 0.00375
loss=0.04922836646437645 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.75it/s]

Test set: Average loss: 0.0198, Accuracy: 9947/10000 (99%)

Epoch 13, Learning Rate: 0.00375
loss=0.14018218219280243 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.99it/s]

Test set: Average loss: 0.0196, Accuracy: 9940/10000 (99%)

Epoch 14, Learning Rate: 0.00375
loss=0.07895512133836746 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.84it/s]

Test set: Average loss: 0.0202, Accuracy: 9944/10000 (99%)

Epoch 15, Learning Rate: 0.00375
loss=0.0779527872800827 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.67it/s]

Test set: Average loss: 0.0197, Accuracy: 9944/10000 (99%)

Epoch 16, Learning Rate: 0.00375
loss=0.09693173319101334 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.89it/s]

Test set: Average loss: 0.0203, Accuracy: 9944/10000 (99%)

Epoch 17, Learning Rate: 0.00375
loss=0.07345765084028244 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.69it/s]

Test set: Average loss: 0.0197, Accuracy: 9945/10000 (99%)

Epoch 18, Learning Rate: 0.00375
loss=0.12088511139154434 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.78it/s]

Test set: Average loss: 0.0196, Accuracy: 9943/10000 (99%)

Epoch 19, Learning Rate: 0.0005625
loss=0.06774037331342697 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.35it/s]

Test set: Average loss: 0.0204, Accuracy: 9940/10000 (99%)
Epoch 1, Learning Rate: 0.025
loss=0.12566810846328735 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 18.91it/s]

Test set: Average loss: 0.0634, Accuracy: 9786/10000 (98%)

Epoch 2, Learning Rate: 0.025
loss=0.12301745265722275 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.13it/s]

Test set: Average loss: 0.0417, Accuracy: 9871/10000 (99%)

Epoch 3, Learning Rate: 0.025
loss=0.16032250225543976 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.15it/s]

Test set: Average loss: 0.0365, Accuracy: 9886/10000 (99%)

Epoch 4, Learning Rate: 0.025
loss=0.13728077709674835 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.21it/s]

Test set: Average loss: 0.0335, Accuracy: 9893/10000 (99%)

Epoch 5, Learning Rate: 0.025
loss=0.2434483915567398 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.20it/s]

Test set: Average loss: 0.0291, Accuracy: 9899/10000 (99%)

Epoch 6, Learning Rate: 0.025
loss=0.22861331701278687 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.15it/s]

Test set: Average loss: 0.0305, Accuracy: 9907/10000 (99%)

Epoch 7, Learning Rate: 0.025
loss=0.11394951492547989 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.41it/s]

Test set: Average loss: 0.0302, Accuracy: 9907/10000 (99%)

Epoch 8, Learning Rate: 0.025
loss=0.10457319766283035 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.23it/s]

Test set: Average loss: 0.0234, Accuracy: 9928/10000 (99%)

Epoch 9, Learning Rate: 0.025
loss=0.14934244751930237 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.22it/s]

Test set: Average loss: 0.0232, Accuracy: 9928/10000 (99%)

Epoch 10, Learning Rate: 0.00375
loss=0.19503778219223022 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.22it/s]

Test set: Average loss: 0.0197, Accuracy: 9942/10000 (99%)

Epoch 11, Learning Rate: 0.00375
loss=0.11391881853342056 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.36it/s]

Test set: Average loss: 0.0196, Accuracy: 9943/10000 (99%)

Epoch 12, Learning Rate: 0.00375
loss=0.04922836646437645 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.75it/s]

Test set: Average loss: 0.0198, Accuracy: 9947/10000 (99%)

Epoch 13, Learning Rate: 0.00375
loss=0.14018218219280243 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.99it/s]

Test set: Average loss: 0.0196, Accuracy: 9940/10000 (99%)

Epoch 14, Learning Rate: 0.00375
loss=0.07895512133836746 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.84it/s]

Test set: Average loss: 0.0202, Accuracy: 9944/10000 (99%)

Epoch 15, Learning Rate: 0.00375
loss=0.0779527872800827 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.67it/s]

Test set: Average loss: 0.0197, Accuracy: 9944/10000 (99%)

Epoch 16, Learning Rate: 0.00375
loss=0.09693173319101334 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.89it/s]

Test set: Average loss: 0.0203, Accuracy: 9944/10000 (99%)

Epoch 17, Learning Rate: 0.00375
loss=0.07345765084028244 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.69it/s]

Test set: Average loss: 0.0197, Accuracy: 9945/10000 (99%)

Epoch 18, Learning Rate: 0.00375
loss=0.12088511139154434 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.78it/s]

Test set: Average loss: 0.0196, Accuracy: 9943/10000 (99%)

Epoch 19, Learning Rate: 0.0005625
loss=0.06774037331342697 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.35it/s]

Test set: Average loss: 0.0204, Accuracy: 9940/10000 (99%)


