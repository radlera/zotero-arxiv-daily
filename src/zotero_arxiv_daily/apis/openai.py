from abc import abstractmethod

import numpy as np
from openai import OpenAI
import tiktoken
from loguru import logger

from zotero_arxiv_daily.apis.base import BaseAPI



class OpenAIWrapper(BaseAPI):

    def __init__(self, config):
        super().__init__(config)

        self.client = OpenAI(api_key=self.config.api.key)


    def _api_call(self, system_prompt, prompt, model, max_input_tokens) -> np.ndarray:
        
        enc = tiktoken.encoding_for_model(model)
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:max_input_tokens]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            # **self.config.llm_params.get('generation_kwargs', {})
            model=model,
            max_tokens=1024,  # max output tokens
        )
        response = response.choices[0].message.content
        return response

    def get_tldr(self, system_prompt, prompt):
        return self._api_call(
            system_prompt, 
            prompt, 
            model=self.config.tldr.get('model', "gpt-4o-mini"),
            max_input_tokens=self.config.tldr.get('max_tokens', 4000)
        )

    def get_affiliations(self, system_prompt, prompt) -> np.ndarray:
        return self._api_call(
            system_prompt, 
            prompt, 
            model=self.config.affiliations.get('model', "gpt-4o"),
            max_input_tokens=self.config.affiliations.get('max_tokens', 2000)
        )
    
    def get_embedding(self, all_texts) -> np.ndarray:

        batch_size = self.config.embeddings.get('batch_size', 64)

        all_embeddings = []

        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.config.embeddings.model
            )
            all_embeddings.extend([r.embedding for r in response.data])

        return all_embeddings