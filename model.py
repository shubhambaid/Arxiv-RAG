from langchain_community.llms import CTransformers
from ctransformers import AutoModelForCausalLM


class LLMModel:
    def __init__(self):

        self.llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Tinyllama-2-1b-miniguanaco-GGUF",
            model_file="tinyllama-2-1b-miniguanaco.Q4_K_M.gguf",
            model_type="llama",
            gpu_layers=0,
        )
        print(f"Model loaded successfully!")

    def getLLM(self):
        return self.llm
