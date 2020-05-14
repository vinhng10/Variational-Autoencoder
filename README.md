# Variational Autoencoder
A side project on apply Variational Autoencoder and Pytorch to learn latent representation of human face and generate new faces that don't exist.
Please take a look at the jupyter notebook for demonstration.
# Data
In this project, face data is acquired from FIFA20 dataset, which consists of 16,176 face images of soccer players. The dataset is downloaded from [https://sofifa.com/](https://sofifa.com/). 
The file **utils.py** provides a function 
- **download_images** for convenient downloading 
- **remove_error_images** for removing any errors in the downloaded data.

