from langchain_community.llms import CTransformers
from langchain_community.llms import LlamaCpp
from ctransformers import AutoModelForCausalLM


class LLMModel:

    def __init__(
        self,
        model_name="TheBloke/Tinyllama-2-1b-miniguanaco-GGUF",
        model_path="./tinyllama-2-1b-miniguanaco.Q4_K_M.gguf",
        n_gpu_layers=0,
        n_batch=8,
        temperature=0.1,
        verbose=True,
        max_tokens=512,
        top_p=1,
    ):

        self.__model_name = model_name
        self.__model_path = model_path
        self.__n_gpu_layers = n_gpu_layers
        self.__n_batch = n_batch
        self.__temperature = temperature
        self.__verbose = verbose
        self.__max_tokens = max_tokens
        self.__top_p = top_p

        try:
            self.__llm = LlamaCpp(
                model_path=self.__model_path,
                temperature=self.__temperature,
                max_tokens=self.__max_tokens,
                top_p=self.__top_p,
                n_gpu_layers=self.__n_gpu_layers,
                n_batch=self.__n_batch,
                verbose=self.__verbose,
            )
            print(f"Model {self.__model_name} loaded successfully!")

        except Exception as e:
            print("Couldn't load the model. Exception : ", e)

    def getLLM(self):
        return self.__llm
