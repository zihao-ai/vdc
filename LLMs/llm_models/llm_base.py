import torch
from PIL import Image


class LLM_base(object):
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def process_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def process_prompt(self, prompt):
        return prompt

    def generate_mm(self, image_path, prompt):
        pass

    def generate_nlp(self, prompt):
        pass


def load_llm(llm, device=None):
    if llm == "BLIP2":
        from LLMs.llm_models.blip2 import BLIP2

        return BLIP2(device)
    elif llm == "InstructBLIP":
        from LLMs.llm_models.instructblip import InstructBLIP

        return InstructBLIP(device=device)
    elif llm == "LLaMA_Adapter_v2":
        from LLMs.llm_models.llama_adapter_v2 import LLaMA_Adapter_v2

        return LLaMA_Adapter_v2(device)
    elif llm == "llava":
        from LLMs.llm_models.llava1_5 import LLaVA

        return LLaVA(device)
    elif llm == "MiniGPT4":
        from LLMs.llm_models.minigpt4 import MiniGPT4

        return MiniGPT4(device)
    elif llm == "Otter":
        from LLMs.llm_models.otter import Otter

        return Otter(device)
    elif llm == "mPLUG":
        from LLMs.llm_models.mplug import mPLUG

        return mPLUG(device)
    elif llm == "Vicuna":
        from LLMs.llm_models.vicuna import Vicuna

        return Vicuna(device)

    elif llm == "gpt4":
        from LLMs.llm_models.gpt import GPT

        return GPT(device)

    elif llm == "qwen":
        from LLMs.llm_models.qwen import QWen

        return QWen(device)
