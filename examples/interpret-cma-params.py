from dataclasses import dataclass

@dataclass
class Parameter:
    """
    Base data class for CMA model parameters.
    """
    value: float

    def describe(self):
        raise NotImplementedError("This method should be implemented by subclasses")


class CircadianAlignment(Parameter):
    def describe(self):
        if self.value < -0.5:
            return "Significant circadian misalignment."
        elif -0.5 <= self.value < -0.2:
            return "Slight circadian misalignment."
        return "Good circadian alignment."


class LightExposure(Parameter):
    def describe(self):
        if self.value > 4:
            return "Very long exposure to artificial light."
        return "Normal light exposure."


class GlucoseResponse(Parameter):
    def describe(self):
        if self.value < 0.9:
            return "Fast glucose response."
        return "Normal glucose response."


class GlucoseBias(Parameter):
    def describe(self):
        if self.value > 0.1:
            return "Tendency towards higher glucose levels."
        return "Balanced glucose levels."


class CortisolSensitivity(Parameter):
    def describe(self):
        if self.value > 0.7:
            return "Higher cortisol sensitivity."
        return "Low-normal cortisol sensitivity."


class SolarNoonAlignment(Parameter):
    def describe(self):
        if abs(self.value) > 0.5:
            return "Significant solar noon misalignment."
        return "Good solar noon alignment."


class CMATrainingSampleGenerator:
    def __init__(self, params):
        self.params = params
        self.param_objects = {
            "d": CircadianAlignment(params["d"]),
            "taup": LightExposure(params["taup"]),
            "taug": GlucoseResponse(params["taug"]),
            "B": GlucoseBias(params["B"]),
            "Cm": CortisolSensitivity(params["Cm"]),
            "toff": SolarNoonAlignment(params["toff"])
        }

    def generate_sample(self):
        descriptions = [param_obj.describe() for param_obj in self.param_objects.values()]
        return " ".join(descriptions)

# Example usage
params = {
    "d": -0.21,
    "taup": 4.67,
    "taug": 1.10,
    "B": 0.13,
    "Cm": 0.00,
    "toff": 0.00
}

generator = CMATrainingSampleGenerator(params)
print(generator.generate_sample())
