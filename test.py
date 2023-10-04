# tests.py

import unittest
from train_model import X, y, model

class TestModelTraining(unittest.TestCase):
    def test_data_shape(self):
        self.assertEqual(X.shape[0], y.shape[0])

    def test_model_accuracy(self):
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        self.assertTrue(mse < 1.0)  # Simple accuracy check

if __name__ == '__main__':
    unittest.main()
