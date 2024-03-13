import torch
from mPLUG.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mPLUG.mplug_owl.processing_mplug_owl import (MplugOwlImageProcessor,
                                                  MplugOwlProcessor)
from PIL import Image
from transformers import AutoTokenizer

from LLMs.llm_models.llm_base import LLM_base


class mPLUG(LLM_base):
    def __init__(self):
        super().__init__()
        pretrained_ckpt = "MAGAer13/mplug-owl-llama-7b"
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)

    def process_prompt(self, prompt):
        prompts = [
            f"""The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <image>
        Human: {prompt}.
        AI: """
        ]
        return prompts

    def generate_mm(self, image_path, prompt):
        prompts = self.process_prompt(prompt)
        image_list = [image_path]

        # generate kwargs (the same in transformers) can be passed in the do_generate()
        generate_kwargs = {"do_sample": True, "top_k": 5, "max_length": 512}

        images = [Image.open(_) for _ in image_list]
        inputs = self.processor(text=prompts, images=images, return_tensors="pt")
        inputs = {
            k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()
        }
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        return sentence
