from openai import OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

import numpy as np
import gc
from tqdm import tqdm


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
            # return a very low log probability ~= to prob=0.0 if the token is not in the log_probs dictionary
            lg_prob = -float('inf')
        
        return lg_prob
    

    def add_scores_to_df(self, df, score_col):
        target_probabilities = []
        prior_probabilities = []
        scores = []

        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            target_probabilities.append( self.get_token_prob(row['Sent_TM'], row['Person']) )
            prior_probabilities.append( self.get_token_prob(row['Sent_TAM'], row['Person']) )
            scores.append( np.log( np.exp(target_probabilities[-1]) / np.exp(prior_probabilities[-1]) ) )

        df[score_col] = scores

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
    

class BertClient(ApiClient):
    def __init__(self, model):
        super().__init__(model)
        self.client = pipeline('fill-mask', model=model)


    def get_token_prob(self, sentence, token):
        try:
            prob = self.client(sentence, targets=token.lower())[0]['score']
        except TypeError:
            prob = self.client(sentence, targets=token.lower())[0][0]['score']
        
        return np.log(prob)
    

class LlamaClient(ApiClient):
    def __init__(self, model, key):
        super().__init__(model, key)
        self.client = AutoModelForCausalLM.from_pretrained(model, device_map="auto", token=key)
        self.tokenizer = AutoTokenizer.from_pretrained(model, device_map="auto", token=key)


    def request(self, payload, n_logprobs=20):
        # clear memory before each request to avoid OOM errors
        torch.cuda.empty_cache()
        gc.collect()
        inputs = self.tokenizer(f'{self.prompt}\n{payload}', return_tensors="pt").to(self.client.device)
        outputs = self.client.generate(
            **inputs,
            max_new_tokens=10,
            output_scores=True,            #
            return_dict_in_generate=True   #
        )

        # scores is a tuple of (batch_size, vocab_size) with logits
        next_token_logits = outputs.scores[0]
        # apply softmax to get log probabilities
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1) #

        # Get top n
        top_logprobs, top_indices = torch.topk(next_token_logprobs, n_logprobs, dim=-1) #
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices[0])

        log_probs_dict = {}

        for token, logprob in zip(top_tokens, top_logprobs[0]):
            # clean space token from Llama 3 output and add to dictionary
            log_probs_dict[token.strip().replace('Ġ', '')] = logprob.item()

        return log_probs_dict
