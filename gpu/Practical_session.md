# Practical Session on GPU Computation Class
Prepared by: Hieng MAO
## A First Network: MNIST
### Exercise 1: Research, installation, first training
1. Conduct a thorough search about the MNIST (Modified National Institute of Standards and Technology) database. 
Since the searching for MNIST database, I have landed to a [Kaggle's page](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) which shortly explains the dataset. The MNIST database / dataset is a subset of a larger set available from NIST (it's copied from http://yann.lecun.com/exdb/mnist/) which consists of handwritten digits which splitted into training set and testing set of 60,000 and 10,000 examples respectively. There are four files available for download:
    - `t10k-images-idx3-ubyte.gz` test set images (1,611 KB)
    - `t10k-labels-idx1-ubyte.gz` test set labels (5 KB)
    - `train-images-idx3-ubyte.gz` train set images (9,681 KB)
    - `train-labels-idx1-ubyte.gz` train set labels (29 KB)
2. After anaconda installation, we prepare the working environement and required packages installation as below:
    1. Create a new conda environment
```sh
conda create --name myenv python=3.11
```
The reason of python 3.11 is that I tried with python 3.12 but some package installation cannot be done with incompatibility error message.


    2. Activate the environment
```sh
conda activate myenv
```
After we have our working space, we installed the required package as below:

```
conda install numpy
```
```
conda install pandas
```
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
```
conda install matplotlib scikit-learn
```
Also install `h5py` for writing the `hdf5` file:
```
conda install h5py
``` 

>>> Note: for running on the GPU, you can install pytorch that compatible with CUDA version (e.g., `11.7`) using command below. Replace `x.x` with the desired CUDA version:
```
conda install pytorch torchvision torchaudio cudatoolkit=x.x -c pytorch
```

Since I use Windows, installing `soundfile` package to use as `torchaudio`'s backend as below:
```
conda install -c conda-forge pysoundfile
```

3. Executing the `main.py`
Since the `main.py` will us pytorch to download the MNIST dataset, for the first time it runs, the dataset will be downloaded, stored and extracted in the `data/MNIST/raw` directory. And by understanding the `main` function of the `main.py` file, We can run it with minimum requirement for the first time to download the dataset as command follow:
```
python main.py --epochs 1 --lr 1.1 
```
As the result, just one epoch training with learning rate 1.1, we receive a very good accuracy on the testing dataset of **9817/10000 (98%)** with average loss of **0.206047** which is quite good result. But our goal is not about accurary, make just ensure everything is functioning and the dataset is downloaded. The deeper code analysis and more experienment will be conducted as detail as describe below.

### Exercice 2: Code Analysis
1. Neural network structure: named `Net`, a subclass of pytorch's `nn` module which contains the following components:
    1. `self.conv1 = nn.Conv2d(1, 32, 3, 1)`: A convolutional layer with 1 input channel (since the data are grayscale image which has one value is intensity), 32 output channels (correspond to the 32 features map, e.g., vertical edge, horizontal edge, ...etc.), a kernel (convolutional filter) size of 3, and a stride of 1 (the filter move 1 pixel at a time during the convolution operation-a hyperparameter that control the step size of the kernel).
    2. `self.conv2 = nn.Conv2d(32, 64, 3, 1)`: An other convolutional layer with 32 input channels, 64 output channels, a kernel size of 3, and a stride of 1.
    3. `self.dropout1 = nn.Dropout2d(0.25)`: A dropout layer with a dropout probability of 0.25 that is used to avoid the model overfitting (The model / network performs very good with the training dataset but is not generalize for unseen data).
    4. `self.dropout2 = nn.Dropout2d(0.5)`: An other droput layer with a dropout probability of 0.5.
    5. `self.fc1 = nn.Linear(9216, 128)`: A fully connected layer with 9216 input features and 128 output features. Input of 9216 is not random picked. It is received from the flattened output from the previous layers which:
        - Input image of 28x28 pixels with 1 channel (grayscale).
        - First Conv layer of input (1, 28, 28), output (32, 26, 26) with calculation `output_size = (input_size - kernel_size) / stride + 1 = (28 - 3) / 1 +1 = 26`
        - Second Conv layer of input (32, 26, 26), output (64, 24, 24), with the same calculation `(26 - 3) / 1 + 1 = 24`.
        - Then, we do the max pooling (see `forward` function) `F.max_pool2d(x, 2)` of input (64, 24, 24), output (64, 12, 12) with calculation **pooling reduces each dimension by a factor of 2**
        - Therefore, the flattening the ouput into 1D tensor which is `64 * 12 * 12 = 9216`.
2. How the training and evaluation procedure operate?
- **Training Procedure**
The training procedure is defined in the `train` function which operates step by step as below:
    1. It sets the model to the training mode `True` using `model.train()` which affect to the dropout (`:class: Dropout`) and batch normalization (`:class: BatchNorm`).
    2. It iterates over the training data (images and labels) in batches on `train_loader` object of DataLoader to do:
        - It moves data to the appropriate device (CPU or GPU).
        - It clears the gradients of all optimized tensors before each backward pass, using `optimizer.zero_grad()`.
        - It does forward pass by computing the model's output passing through the network from the input, using `output = model(data)`.
        - It computes the Loss of negative log likihood loss between the model's output and the target labels, using `loss = F.nll_loss(output, target)` since the model is designed for classification tasks where the output is a probability distribution over multiple classes.
        - It computes the gradient of the loss with respect to the model parameters, using `loss.backward()`.
        - It updates the model parameters using the computed gradients, with `optimizer.step()`.
        - Lastly, it prints out the training epoch step and the loss value respectively to the given `args.log_interval`.

- **Evaluation Procedure**
The evaluation procedure is defined in the `test` function which operates step by step as below:
    1. It sets the model to the evaluation mode `True` using `model.eval()` which affect to the dropout and batch normalization.
    2. It records the average loss and the number of correct predictions.
    3. It disables the gradient calculation, which reduces memory consumption and speed up computation since gradients are not needed during evalutation, using `torch.no_grad()`.
    4. It iterates over the testing dataset to do:
        - It moves data to the appropriate device (CPU or GPU).
        - It does forwar pass by computing the model's output passing through the network from the input.
        - It computes the negative log-likelihood loss and summed up over the entire test set.
        - It predicts the class obtained by finding the index of the maximum log-probability.
        - The number of correct predictions is also accumulated.
    5. It averages the total loss by dividing by the number of samples in the test set.
    6. Finally, it prints out the evaluation results.

3. Optimizer
In this practice source code, the Adadelta optimizer is used which is an adaptive learning rate optimization algorithm designed to improve upon Adagrad. In detail, base on [keras.io](https://keras.io/api/optimizers/adadelta/), Adadelta is a stochastic gradient descent method that is based on adaptive learning rate per dimension to address two drawbaks:
- The continual decay of learning rates throughout training.
Adadelta dynamically adjuss the learning rate for each parameter based on a moving window of gradient updates, which helps in dealing with the diminishing learning rates problem faced by Adagrad.
- The need for a manually selected global learning rate.
Unlike some other optimizers, Adadelta does not require an initial learning rate to be set manually. Based on the adaptive learning rate which can simplify the hyperparameter tuning process.

One more advantage of Adadelta, since it uses a moving window of gradient updates instead of accumulating all past gradients, which makes it more memory efficient compared to Adagrad.

In summary, using Adadelta optimizer helps in automatically adjusting the learning rates during training, which can lead to better convergence and performance without the need for extensive hyperparameter tuning. 

### Exercise 3: Analysis of Loss Evolution During Training 
1. Modify the `train` and `test` function
Since we just want the average loss per epoch, we modify the `train` and `test` function to returns their total losses divided by total training and testing dataset respectively for `train` and `test` function. Therefor, we store all average losses of training and testing in a list corresponding to the epoches (`args.epoches`). 

In order to keep tracking the working history and to effeciently plot the graph, these losses are saved to a csv file in an `exports` folder. 

For the sake of simplicity, we run `14` epoches and learning rate of `1.1` through out these experiments. 