import base64
import json

import requests

from LLMs.llm_models.llm_base import LLM_base
from LLMs.llm_models.openai_api_pool import *


class GPT(LLM_base):
    def __init__(self, args=None):
        super(LLM_base, self).__init__()
        self.url, self.api_key = get_openai_api()
        self.args = args

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_nlp(self, prompt):
        model_name = "gpt-3.5-turbo"
        parameters = {"model": model_name, "messages": {"role": "user", "content": prompt}}
        if self.args is not None:
            for key in self.args.keys():
                parameters[key] = self.args[key]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        raw_response = requests.post(self.url, headers=headers, json=parameters, timeout=5)
        response = json.loads(raw_response.content.decode("utf-8"))

        try:
            content = response["choices"][0]["message"]["content"]
            flag = True
            return content
        except:
            content = response["error"]["code"]
            flag = False
            return content

    def generate_mm(self, image_path, prompt):
        base64_image = self.encode_image(image_path)
        model_name = "gpt-4-vision-preview"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ]
        parameters = {
            "model": model_name,
            "messages": message,
            "max_tokens": 300,
        }
        if self.args is not None:
            for key in self.args.keys():
                parameters[key] = self.args[key]

        response = requests.post(self.url, headers=headers, json=parameters)

        raw_response = requests.post(self.url, headers=headers, json=parameters, timeout=5)
        response = json.loads(raw_response.content.decode("utf-8"))

        try:
            content = response["choices"][0]["message"]["content"]
            flag = True
            return content
        except:
            content = response["error"]["code"]
            flag = False
            return content
