# Revenue Prediction using IBM AI Enterprise Workflow
This repository contains a time series prediction application for the IBM AI Enterprise Workflow Capstone project. 
The application uses the Scikit Learn Gaussian Process Regressor as well as datasets provided by the course.

This project contains 
* Unit tests for API, logging, and model
* run_tests.py for running all tests with a single script
* monitoring.py for performance monitoring
* Model_Evaluation.ipynb for model comparison
* Model_Visualization.ipynb for data analysis
* Docker deployment

Usage notes
===============

All commands are from this directory.

To test app.py
---------------------

    ~$ python app.py
    
To test the model directly
----------------------------



    ~$ python GaussianRegressorModely

To build the docker container
--------------------------------

    ~$ docker build --tag gaussian_app .

Check that the image is there.

    ~$ docker image ls
    
You may notice images that you no longer use. You may delete them with

    ~$ docker image rm IMAGE_ID_OR_NAME

And every once and a while if you want clean up you can

    ~$ docker system prune


To run the unittests
-------------------

Before running the unit tests launch the `app.py`.

To run only the api tests

    ~$ python unittests/ApiTests.py

To run only the model tests

    ~$ python unittests/ModelTests.py


To run all of the tests

    ~$ python run-tests.py

To run the container 
--------------------    

    ~$ docker run -p 4000:8080 gaussian_app

 
