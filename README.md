# AutoEncoder

Autoencoders are a type of networks that the input is the same as the output. They compress the input into a lower-dimensional code and then reconstruct the output from this representation.

## Denoising AutoEncoders
Autoencoders present an efficient way to learn a representation of your data, which helps with tasks such as dimensionality reduction or feature extraction. You can even train an autoencoder to identify and remove noise from your data.
The purpose of a DAE is to remove noise. You can also think of it as a customised denoising algorithm tuned to your data.

![image](https://miro.medium.com/max/828/1*iXCORmu7vWolNrcqCTMB0A.png)

[More](https://towardsdatascience.com/denoising-autoencoders-dae-how-to-use-neural-networks-to-clean-up-your-data-cd9c19bc6915)

# Dataset
The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 samples.
I used pytorch datasets for downloading dataset : 
```
train_dataset = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
```

# Model

The critical components of Autoencoders are:

* **Input layer** — to pass input data into the network
* **Hidden layer** consisting of Encoder and Decoder — to process information by applying weights, * biases and activation functions
* **Output layer** — typically matches the input neurons

# Train
Trainer class Does the main part of code which is training model, plot the training process and save model each n epochs.

I Defined `Adam` Optimizer with learning rate 0.0002.

In ```add_noise``` randomly add gaussian and uniform noise with below configs to image: 
* {"type":"gaussian", "mean":0, "var":0.1},
* {"type":"gaussian", "mean":0, "var":0.2},
* {"type":"gaussian", "mean":0, "var":0.3},
* {"type":"gaussian", "mean":0, "var":0.4},
* {"type":"gaussian", "mean":0, "var":0.05},
* {"type":"gaussian", "mean":0, "var":0.01},
* {"type":"uniform", "min":-0.01, "max":0.01},
* {"type":"uniform", "min":-0.1, "max":0.1},
* {"type":"uniform", "min":-0.2, "max":0.2}, 

## Some Configurations
 
*   You can set epoch size : `EPOCHS` and batch size : `BATCH_SIZE`.
*   Set `device` that you want to train model on it : `device`(default runs on cuda if it's available)
*   You can set one of three `verboses` that prints info you want => 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters size.
*   Each time you train model weights and plot(if `save_plots` == True) will be saved in `save_dir`.
*   You can find a `configs` file in `save_dir` that contains some information about run. 
*   You can choose Optimizer: `OPTIMIZER` 

# Results

![AE](https://user-images.githubusercontent.com/47561760/193319563-1ed228c1-cb61-4cb7-97be-9a52fd16aedb.jpg)
