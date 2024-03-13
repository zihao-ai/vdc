import lavis

from LLMs.llm_models.llm_base import LLM_base


class InstructBLIP(LLM_base):
    def __init__(self,device):
        super(InstructBLIP, self).__init__(device)
        self.model, self.vis_processors, _ = lavis.models.load_model_and_preprocess(
            name="blip2_vicuna_instruct",
            model_type="vicuna7b",
            is_eval=True,
            device=self.device,
        )

    def generate_mm(self, image_path, prompt):
        image = self.process_image(image_path)
        prompt = self.process_prompt(prompt)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        generated_text = self.model.generate({"image": image, "prompt": prompt})[
            0
        ].strip()
        return generated_text
