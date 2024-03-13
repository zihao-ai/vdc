import transformers
from PIL import Image

from llm_models.llm_base import LLM_base
from LLMs.Otter.otter.modeling_otter import OtterForConditionalGeneration


class Otter(LLM_base):
    def __init__(self):
        super().__init__()
        self.model = OtterForConditionalGeneration.from_pretrained(
            "luodian/OTTER-9B-LA-InContext", device_map="auto"
        )
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        self.model.eval()

    def process_image(self, image_path):
        return Image.open(image_path)

    def generate_mm(self, image_path, prompt):
        image = self.process_image(image_path)
        prompt = self.process_prompt(prompt)
        encoded_frames_list = [image]
        response = self.get_response(
            encoded_frames_list, prompt, self.model, self.image_processor
        )
        return response

    def get_formatted_prompt(self, prompt: str, in_context_prompts: list = []) -> str:
        in_context_string = ""
        for in_context_prompt, in_context_answer in in_context_prompts:
            in_context_string += f"<image>User: {in_context_prompt} GPT:<answer> {in_context_answer}<|endofchunk|>"
        return f"{in_context_string}<image>User: {prompt} GPT:<answer>"

    def get_response(
        self,
        image_list,
        prompt: str,
        model=None,
        image_processor=None,
        in_context_prompts: list = [],
    ) -> str:
        input_data = image_list

        if isinstance(input_data, Image.Image):
            vision_x = (
                image_processor.preprocess([input_data], return_tensors="pt")[
                    "pixel_values"
                ]
                .unsqueeze(1)
                .unsqueeze(0)
            )
        elif isinstance(input_data, list):  # list of video frames
            vision_x = (
                image_processor.preprocess(input_data, return_tensors="pt")[
                    "pixel_values"
                ]
                .unsqueeze(1)
                .unsqueeze(0)
            )
        else:
            raise ValueError(
                "Invalid input data. Expected PIL Image or list of video frames."
            )

        lang_x = model.text_tokenizer(
            [
                self.get_formatted_prompt(prompt, in_context_prompts),
            ],
            return_tensors="pt",
        )
        bad_words_id = self.tokenizer(
            ["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False
        ).input_ids
        generated_text = model.generate(
            vision_x=vision_x.to(model.device),
            lang_x=lang_x["input_ids"].to(model.device),
            attention_mask=lang_x["attention_mask"].to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_id,
        )
        parsed_output = (
            model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        return parsed_output
