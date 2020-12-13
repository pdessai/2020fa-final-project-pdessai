# Final Project- Name the Country!

## Repo Structure:

- [Problem Statement/Project Overview](#Project Overview)
- [Tools/Technologies Used and Advanced Python Concepts](#Tools/Technologies Used and Advanced Python Conceptsw)
- [About NLP](#About NLP)
- [About ChartJS](#About ChartJS)
- [Workflow](#Workflow)
- [Django Visualization](#Django Visualization)



### Project Overview
The goal of our project is to input a list of random names, 
and predict which is the country of origin for that name using NLP 
and three different neural network models. We use a basic character-level 
RNN which reads words as a series of characters - outputting a prediction 
and “hidden state” at each step, feeding its previous hidden state into 
each next step. We have utilized luigi to design this workflow, and also 
visualized the ouput through Django Chart JS. 



### Tools/Technologies Used and Advanced Python Concepts

We have utilized a bunch of advanced python concepts/tools which learned from or beyond class:
- Pytorch Library for retrieving training/testing data
- Numpy and Pandas for storing and modifying data frames
- Data cleaning, scaling and manipulating principles- functional programming.
- Django web based application for data visualization 
- Python class composition and inheritance
- Recreatable pipenv virtual environments
- CSCI Cookiecutter to develop base project repo
- Django Cookiecutter for Django framework
- Luigi Task for pipeline flow

### About NLP

In this project, we explored NLP (Natural Language Processing) methodologies to address the country name classification problem. 
We have referenced the basic concepts of NLP and RNN (Recurrent Neural Network) based on this [Pytorch tutorial](#https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
.

We've adopted three methods to horse racing the best results:

- Basic RNN: 
a basic kind of recurrent neural network that specialize in processing sequences data

- Multi-layer RNN:
Based on method 1), multi-layer RNN add one more hidden i2o layer inside the network

- LSTM:
a type of RNN that uses special units in addition to standard units. LSTM units include a 'memory cell' that 
can maintain information in memory for long periods of time. 

We will explain more on the code details in later steps.


### About ChartJS
- Chart.js is a free open-source JavaScript library for data visualization, which supports 8 chart types: bar, line, area, pie, bubble, radar, polar, and scatter plots
- It works will with Django applications
- Installation- add the Chart.js lib to our HTML page:
- Best way to retrieve data is to make a Http Get Request from a Chart Html file and provide response from a method in your views.py file


### Workflow
We have utilized luigi to design the workflow for our project. The steps and detailed codes are as bellow:

- Step 1: Load datasets and perform transformation

  The datasets are downloaded from [here](#https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#preparing-the-data)
  
  There are in total 18 country name categories, e.g. English, German, Arabic, etc. Some country names include different special characters, so firstly we need to 
  transform them into ASCII.(see data cleaning functions like: `unicode_2_Ascii`,  ) 
  
  Also, in order to input the name text to the model, we need to convert them into tensor types. And after the prediction output, we need to 
  convert the tensor back to text format.(see functions: `read_lines`, 
  `letter_to_index`, `letter_to_tensor`, `line_to_tensor`, `category_from_output`)
  
  Please refer to `country_name_recognize.data_utils.py`, we have wrapped up all the data cleaning functions into this utils module.

- Step 2: Random sampling to get training datasets
  
  To improve model training efficiency, we need to sample the training datasets each iteration. 
  (see function `sample_trainning` in data_utils.py)
  
- Step 3: Implement different types of RNN models

  We have defined three classes for the different models. The classes is inherited based on torch.nn.modules, and we have customized 
  the network structure based on our own needs. Please see classes `RNN`, `RNN_multi_1ayer`, `LSTM` in data_utils.py
  
  Also, we have defined `train_batch` function to accomondate auto training based on three different model type. (please see `country_name_recognize.rnn.py`)
  
- Predict input names using well-trained model:
  
  After training the model, we designed function `predict_country_name` in `country_name_recognize.rnn.py` to predict the input names 
  using the arguements passed in main function. 
  
  Besides, we designed a luigi workflow on the prediction process. We have utilized the luigi module in csci_utils package which we did in class
   and defined the external tasks to input data and model, then defined `Predict_names` task to output the predictions and save to target path. 
  (please see `rnn_tasks.rnn.py` for more information) 
  

### Django Visualization

We created a Django application to run our model 
and add inputs and then visualize our results using ChartJs. The django app can 
be run by navigating to the src/ directory where the `manage.py` file is. Then run
the command: `python manage.py runserver`. Within the `/src` directory, the `/charts`
directory contains the `setup.py` file where the django app, templates, and installed
apps are setup. Within the `src/` directory there is also a templates directory 
that contains css, js and charts.html files. The `charts.html` file is where the
data visualization tables and charts are created. An api request is made in this file
to retrieve data from a get function. The `views.py` contains the functions to load 
the data and return an http response to pass into the `charts.html` file.  

The visualization is like this:

![alt text](https://github.com/pdessai/2020fa-final-project-pdessai/blob/develop4/data/barchart.png?raw=true)
![alt text](https://github.com/pdessai/2020fa-final-project-pdessai/blob/develop4/data/country_category_chart.png?raw=true)
