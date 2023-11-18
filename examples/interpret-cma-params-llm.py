import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CMATrainingSampleGenerator:
    """
    A class to generate training samples for CMA model parameters using an NLP encoder model.
    """

    def __init__(self, model_name="gpt2"):
        """
        Initializes the generator with a specified NLP model.

        :param model_name: Name of the pre-trained model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.to('cuda')

    def generate_sample(self, params):
        """
        Generates a response based on CMA model parameters using the NLP model.

        :param params: Dictionary containing CMA model parameters.
        :return: A string containing the model-generated response.
        """
        input_text = f"Parameters: {json.dumps(params)}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

# Example usage
params = {
    "d": -0.21,
    "taup": 4.67,
    "taug": 1.10,
    "B": 0.13,
    "Cm": 0.00,
    "toff": 0.00
}

generator = CMATrainingSampleGenerator()
print(generator.generate_sample(params))
