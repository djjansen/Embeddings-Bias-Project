from openai import OpenAI
from collections import defaultdict


class ApiConnection:
    def __init__(self, key, prompt, model):
        self.api_key = key
        self.prompt = prompt
        self.model = model


# Child inherits from Parent
class OpenAiConnection(ApiConnection):
    def __init__(self, key, prompt, model):
        self.key = key
        super().__init__(key, prompt, model)
        self.client = OpenAI(api_key=self.api_key)

    def request(self, payload, n_logprobs=20):
        response = self.client.responses.create(
        model=self.model,
        input=f'{self.prompt}\n{payload}',
        top_logprobs=n_logprobs,
        include=["message.output_text.logprobs"] 
        )

        log_probs_dict = defaultdict(float)
        output = response.output

        for tok in output[0].content:
            log_probs = tok.logprobs[0].top_logprobs
            for candidate in log_probs:
                log_probs_dict[candidate.token] = candidate.logprob

        return log_probs_dict
