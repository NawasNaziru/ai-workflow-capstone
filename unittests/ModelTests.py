import unittest

## import model specific functions and variables
from model.model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(os.path.join("data","cs-train"),test=True)
        saved_model = os.path.join("models","sl-netherlands-0_1.joblib")
        self.assertTrue(os.path.exists(saved_model))

    def test_load(self):
        """
        test the load functionality
        """
                        
        ## train the model
        all_data, all_models = model_load()
        model = all_models['united kingdom']
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_predict(self):
        """
        test the predict function input
        """

        ## ensure that a list can be passed        
        result = model_predict('netherlands','2018','05','02',test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred >= 0.0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
