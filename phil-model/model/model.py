"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
CHECKPOINT = 'mistralai/Mistral-7B-Instruct-v0.3'
class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.tokenizer = None
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self._secrets["hf_access_token"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT,
            token=self._secrets["hf_access_token"],
        )

    def predict(self, request:dict):
        # Run model inference here
        prompt = request.pop("prompt")
        generate_args = {
            "max_new_tokens": request.get("max_new_tokens", 128),
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 0.95),
            "top_k": request.get("top_k", 50),
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,

        }
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            output = self.model.generate(inputs=input_ids, **generate_args)
            return self.tokenizer.decode(output[0])
        
        # return model_input
