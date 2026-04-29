from openai import OpenAI
from transformers import pipeline

import numpy as np


class ApiClient:
    def __init__(self, model, key=None, prompt=None):
        self.api_key = key
        self.prompt = prompt
        self.model = model


    def get_token_prob(self, sentence, token):
        log_probs = self.request(sentence)

        try:
            lg_prob = log_probs[token]
        except KeyError:
            lg_prob = 0
        
        return lg_prob
    

    def add_scores_to_df(self, df):
        target_probabilities = []
        prior_probabilities = []
        scores = []

        for row in df.iloc:  # remember, anything with an index is iterable
            target_probabilities.append( self.get_token_prob(row['Sent_TM'], row['Person']) )
            prior_probabilities.append( self.get_token_prob(row['Sent_TAM'], row['Person']) )
            scores.append( np.log( np.exp(target_probabilities[-1]) / np.exp(prior_probabilities[-1]) ) )

        df['scores'] = scores

        return df


# Child inherits from Parent
class OpenAiClient(ApiClient):
    def __init__(self, key, prompt, model):
        self.key = key
        super().__init__(model, key, prompt)
        self.client = OpenAI(api_key=self.api_key)


    def request(self, payload, n_logprobs=20):
        response = self.client.responses.create(
        model=self.model,
        input=f'{self.prompt}\n{payload}',
        top_logprobs=n_logprobs,
        include=["message.output_text.logprobs"] 
        )

        log_probs_dict = {}
        output = response.output

        for tok in output[0].content:
            log_probs = tok.logprobs[0].top_logprobs
            for candidate in log_probs:
                log_probs_dict[candidate.token] = candidate.logprob

        return log_probs_dict
    
# Child inherits from Parent
class BertClient(ApiClient):
    def __init__(self, model):
        super().__init__(model)
        self.client = pipeline('fill-mask', model=model)


    def get_token_prob(self, sentence, token):
        try:
            prob = self.client(sentence, targets=token)[0]['score']
        except TypeError:
            prob = self.client(sentence, targets=token)[0][0]['score']
        
        return np.log(prob)
