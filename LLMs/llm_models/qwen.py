import base64
import json
import os
from http import HTTPStatus

import dashscope
import requests

from LLMs.llm_models.llm_base import LLM_base

dashscope.api_key = "sk-xx"


class QWen(LLM_base):
    def __init__(self):
        super(LLM_base, self).__init__()

    def generate_mm(self, image_path, prompt):
        messages = [
            {
                "role": "user",
                "content": [{"image": f"file://{image_path}"}, {"text": prompt}],
            }
        ]
        response = dashscope.MultiModalConversation.call(model="qwen-vl-plus", messages=messages)

        if response.status_code == HTTPStatus.OK:
            output = response.output.choices[0].message.content[0]["text"]
            return output

        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
            return None


