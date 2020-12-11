Final Project- Name the Country!

The goal of our project is to input a list of random names, 
and predict which is the country of origin for that name using NLP 
and three different neural network models. We use a basic character-level 
RNN which reads words as a series of characters - outputting a prediction 
and “hidden state” at each step, feeding its previous hidden state into 
each next step. 

We have retrieved training data from the pytorch library. 
We have received few thousand words from 18 different languages of origin. 
We use this to train the model and then when we feed it new words we use it 
to predict the output. The data can be found in the /data directory with separate
/input and /output directories. The /input directory contains the 18 different
.txt files with thousands of records on names from 18 different countires. 
The /output directory contains the predicted output data for each of the 3 models.


We then created a Django application to run our model 
and add inputs and then visualize our results using ChartJs. The django app can 
be run by navigating to the src/ directory where the manage.py file is. Then run
the command: python manage.py runserver. Within the /src directory, the /charts
directory contains the setup.py file where the django app, templates, and installed
apps are setup. Within the src/ directory there is also a templates directory 
that contains css, js and charts.html files. The charts.html file is where the
data visualization tables and charts are created. An api request is made in this file
to retrieve data from a get function. The views.py contains the functions to load 
the data and return an http response to pass into the charts.html file.  


