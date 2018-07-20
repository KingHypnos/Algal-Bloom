# Algal-Bloom
The goal of this porject is to create a Convolutional Neural Network that will examine the month by month progress of algal blooms in the Great Lakes and create a predicative algorithm that you can input a picture of the great lakes to as well as some other data (ex: time of year). It will be able to ouput a photo that is a prediction of what the algal bloom in that photo will look like in a month, 3 months, and a year. We are using http://www.greatlakesremotesensing.org/ for our data.

___

# Prerequisites
- Install Theano  
`pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git`

- Install TensorFlow  
`pip install tensorflow`

- Install Keras  
`pip install --upgrade keras`

- Download the [dataset](https://drive.google.com/open?id=1XaFM8BJFligrqeQdE-_5Id0V_SubJAZe)

### File structure:
```
master
  -cnn.py
  -dataset
    -training_set
      -cat
      -dog
    -testing_set
      -cat
      -dog
```
If the project does not run you may also have to install Pillow:
`pip install Pillow`
