from abc import abstractmethod

import numpy as np
from mistralai.client import Mistral
import tiktoken
from loguru import logger

from zotero_arxiv_daily.apis.base import BaseAPI



class MistralWrapper(BaseAPI):

    def __init__(self, config):
        super().__init__(config)

        self.client = Mistral(api_key=self.config.api.key)


    def _api_call(self, system_prompt, prompt, model, max_input_tokens) -> np.ndarray:
        
        # enc = tiktoken.encoding_for_model(model)
        # prompt_tokens = enc.encode(prompt)
        # prompt_tokens = prompt_tokens[:max_input_tokens]
        # prompt = enc.decode(prompt_tokens)
        prompt = prompt[:max_input_tokens]

        response = self.client.chat.complete(
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
            stream=False, 
            response_format={"type": "text"}
        )
        response = response.choices[0].message.content
        return response

    def get_tldr(self, system_prompt, prompt):

        system_prompt += ' Do not include the text "TL;DR" or similar. '
        'Do not add ** or * around the TLDR test.'
        'Individual parts could be highlighted though.'

        return self._api_call(
            system_prompt, 
            prompt, 
            model=self.config.tldr.model,
            max_input_tokens=5000
        )

    def get_affiliations(self, system_prompt, prompt) -> np.ndarray:

        system_prompt += ' The affiliations are usually marked with superscripts or subscripts at the authors. '
        'Typical affiliations are universities, research institutions or companies.'

        return self._api_call(
            system_prompt, 
            prompt, 
            model=self.config.affiliations.model,
            max_input_tokens=5000
        )
    
    def get_embedding(self, all_texts) -> np.ndarray:

        batch_size = self.config.embeddings.get('batch_size', 32)

        all_embeddings = []

        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            response = self.client.embeddings.create(
                inputs=batch,
                model=self.config.embeddings.model
            )
            all_embeddings.extend([r.embedding for r in response.data])

        return all_embeddings
    

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../config", config_name="default")
    def main(cfg: DictConfig):
        wrapper = MistralWrapper(cfg.llm)
        # Example usage
        system_prompt = "You are a helpful assistant."
        prompt = "Summarize the following document: ..."

        # Test get_tldr
        tldr = wrapper.get_tldr(system_prompt, prompt)
        logger.info(f"TLDR: {tldr}")

        # Test get_affiliations
        affiliations = wrapper.get_affiliations(system_prompt, prompt)
        logger.info(f"Affiliations: {affiliations}")

        # Test get_embedding
        texts = ["This is a test document.", "Another test document."]
        embeddings = wrapper.get_embedding(texts)
        logger.info(f"Embeddings shape: {np.array(embeddings).shape}")

    main()
