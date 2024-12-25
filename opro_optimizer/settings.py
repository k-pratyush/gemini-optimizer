from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai


class Settings():
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()

    optimization_config = {
        "temperature": 1.0,
        "batch_size": 1,
        "num_servers": 1,
        "max_decode_steps": 1024,
        "model": os.getenv("MODEL_NAME") or "gemini-2.0-flash-exp"
    }


class RegressionSettings(Settings):
    def __init__(self, w_true, b_true, num_points=50):
        super().__init__()
        self.w_true = w_true
        self.b_true = b_true
        self.num_points = num_points
        self.num_reps = 5
        self.max_steps = 500
        self.num_generated_points_in_each_step = 8
        self.num_input_decimals = 0
        self.num_output_decimals = 0
        self.max_num_pairs = 20

        # data
        np.random.seed(0)
        self.X = np.arange(self.num_points).astype(float) + 1
        self.y = self.X * self.w_true + self.b_true + np.random.randn(self.num_points)

    def get_training_data(self):
        return self.X, self.y

    def init_params(self,num_starting_points=5):
        params = []
        np.random.seed(42)
        init_w = np.random.uniform(low=10, high=20, size=num_starting_points)
        np.random.seed(54)
        init_b = np.random.uniform(low=10, high=20, size=num_starting_points)

        rounded_inits = [
        (np.round(w, self.num_input_decimals), np.round(b, self.num_input_decimals))
        for w, b in zip(init_w, init_b)
        ]
        rounded_inits = [
            tuple(item) for item in list(np.unique(rounded_inits, axis=0))
        ]
        for w, b in rounded_inits:
            z = RegressionSettings.evaluate_loss(self.X, self.y, w, b)
            params.append((w, b, z))

        return params

    @staticmethod
    def evaluate_loss(X, y, w, b):
        residual = y - (X * w + b)
        return np.linalg.norm(residual) ** 2


settings = Settings()
