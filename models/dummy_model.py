import random

class DummyModel:
    def __init__(self, noise_level=0.5):
        self.noise_level = noise_level

    def _add_noise(self, val, scale=1.0):
        return val + random.uniform(-self.noise_level, self.noise_level) * scale

    def predict(self, input_data):
        predictions = []

        # Predict cars from input_data
        for car in ["CarA", "CarB"]:
            location = input_data[f"{car}_Location"]
            rotation = input_data[f"{car}_Rotation"]
            dimension = input_data[f"{car}_Dimension"]

            noisy_location = [self._add_noise(x, 1.0) for x in location]
            noisy_rotation = self._add_noise(rotation, 5.0)
            noisy_dimension = [self._add_noise(x, 0.2) for x in dimension]

            predictions.append({
                "Location": noisy_location,
                "Rotation": noisy_rotation,
                "Dimension": noisy_dimension,
                "object": "Car"
            })

        # Random number of pedestrians
        for _ in range(random.randint(1, 3)):
            predictions.append({
                "Location": [self._add_noise(0, 5.0), self._add_noise(0, 5.0)],
                "Rotation": self._add_noise(0, 20.0),
                "Dimension": [0.5 + random.random() * 0.3, 0.5 + random.random() * 0.3],
                "object": "Pedestrian"
            })

        return predictions
