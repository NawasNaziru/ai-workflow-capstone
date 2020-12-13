import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np

port = 8080

try:
    requests.post('http://0.0.0.0:{}/test'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    Test the essential API functionality
    """

    @unittest.skipUnless(server_available,"local server is not running")
    def test_train(self):
        """
        test the train functionality
        """
      
        request_json = {'mode':'test'}
        r = requests.post('http://0.0.0.0:{}/train'.format(port),json=request_json)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_predict_empty(self):
        """
        ensure appropriate failure types
        """
    
        ## test with no data at all 
        r = requests.post('http://0.0.0.0:{}/predict'.format(port))
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

        ## test withe improperly formatted data
        r = requests.post('http://0.0.0.0:{}/predict'.format(port),json={"key":"value"})     
        self.assertEqual(re.sub('\n|"','',r.text),"[]")
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_predict(self):
        """
        test the predict functionality
        """

        query_data = {'country': "all",
                      'year': "2019",
                      'month': "11",
                      'day': '19'
        }

        

        r = requests.post('http://0.0.0.0:{}/predict'.format(port),json=query_data)
        self.assertTrue(r)
### Run the tests
if __name__ == '__main__':
    unittest.main()
