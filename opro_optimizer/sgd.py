import os
import json
import numpy as np
from settings import RegressionSettings, settings
from utils import NpEncoder

class SGDRegression(RegressionSettings):
    def __init__(self, w_true, b_true, learning_rate=0.000001, tolerance=0.1, num_points=50):
        super().__init__(w_true, b_true, num_points)
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def fit(self):
        X, y = self.get_training_data()
        expected_loss = self.evaluate_loss(X, y, self.w_true, self.b_true)
        w, b = self.init_params(num_starting_points=1)[0][:2]

        config_dict = {
            "w_true": self.w_true,
            "b_true": self.b_true,
            "num_points": self.num_points,
            "init_w": w,
            "init_b": b,
            "learning_rate": self.learning_rate,
            "tolerance": self.tolerance,
            "num_iterations": 0,
            "w_b_pairs": set(),
            "loss_curve": [],
            "loss": None
        }

        loss = self.evaluate_loss(X, y, w, b)

        while abs(loss - expected_loss) > self.tolerance:
            y_preds = X * w + b
            dw =  np.sum(np.dot(X, y_preds - y))
            db = np.sum(np.sum(y_preds - y))

            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            config_dict["num_iterations"] += 1
            config_dict["w_b_pairs"].add((w, b))

            loss = self.evaluate_loss(X, y, w, b)
            config_dict["loss_curve"].append({"iter": config_dict["num_iterations"], "loss": loss})
            config_dict["loss"] = loss

        config_dict["num_unique_w_b_pairs"] = len(config_dict["w_b_pairs"])

        return config_dict

def main():
    save_folder = os.path.join(settings.ROOT_PATH, "outputs", "sgd-results")
    os.makedirs(save_folder, exist_ok=True)

    test_pairs = [(15,14), (17,17), (16,10), (3,5), (25,23), (2,30), (36,-1)]
    for w_b_pair in test_pairs:
        sgd = SGDRegression(*w_b_pair)
        config_dict = sgd.fit()
        
        with open(os.path.join(save_folder, f"sgd-w_{w_b_pair[0]}-b_{w_b_pair[1]}.json"), "w") as f:
            json.dump(config_dict, f, indent=4, cls=NpEncoder)


if __name__ == "__main__":
    main()
