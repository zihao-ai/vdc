from PIL import Image

from LLaMA_Adapter.llama_adapter_v2_multimodal import llama
from llm_models.llm_base import LLM_base


class LLaMA_Adapter_v2(LLM_base):
    def __init__(self):
        super(LLaMA_Adapter_v2, self).__init__()
        llama_dir = "/workspace2/zhuzihao/mllm/ckpts/LLaMA/"
        self.model, self.preprocess = llama.load("BIAS-7B", llama_dir, self.device)

    def process_image(self, image_path):
        import cv2

        return Image.fromarray(cv2.imread(image_path))

    def process_prompt(self, prompt):
        return llama.format_prompt(prompt)

    def generate_mm(self, image_path, prompt):
        image = self.process_image(image_path)
        prompt = self.process_prompt(prompt)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        generated_text = self.model.generate(image, [prompt])[0]
        return generated_text
