from communex.module import Module, endpoint
from communex.key import generate_keypair
from keylimiter import TokenBucketLimiter
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime

class BTCPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor()

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)

    def preprocess_data(self):
        # Placeholder for additional preprocessing if needed
        pass

    def train_model(self):
        X = self.data.drop('Price', axis=1)
        y = self.data['Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        rmse = sqrt(mean_squared_error(self.y_test, predictions))
        return rmse

    def predict(self, future_timestamp):
        # Convert future_timestamp to the same format as the training data
        # Extract features from future_timestamp
        # Make prediction with the trained model
        # This is a simplified example, in a real-world scenario you would need to preprocess the timestamp
        # and potentially use more features for the prediction
        features = [datetime.datetime.strptime(future_timestamp, '%Y-%m-%d %H:%M:%S').timestamp()]
        return self.model.predict([features])[0]

class Miner(Module):
    """
    A module class for mining and generating responses to prompts.
    """

    def __init__(self):
        self.predictor = BTCPricePredictor()
        # Load and preprocess the data
        self.predictor.load_data('path_to_preprocessed_csv_file.csv')
        self.predictor.preprocess_data()
        # Train the model
        self.predictor.train_model()

    @endpoint
    def generate(self, prompt: str, model: str = "btc-price-predictor"):
        """
        Generates a response to a given prompt using a specified model.
        """
        # Extract the timestamp from the prompt
        timestamp = prompt.split(' ')[-1]
        # Predict the BTC price for the given timestamp
        predicted_price = self.predictor.predict(timestamp)
        return f"The predicted BTC price at {timestamp} UTC is {predicted_price}"

if __name__ == "__main__":
    key = generate_keypair()
    miner = Miner()
    refill_rate = 1 / 400
    bucket = TokenBucketLimiter(2, refill_rate)
    server = ModuleServer(miner, key, ip_limiter=bucket, subnets_whitelist=[3])
    app = server.get_fastapi_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
