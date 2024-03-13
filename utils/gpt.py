import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from LLMs.llm_models.openai_api_pool import *


class GPT:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.history_message = []

    def request_chatgpt(self, url, openai_key, parameters):
        url = url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }

        raw_response = requests.post(
            url, headers=headers, json=parameters, verify=False, timeout=10
        )
        response = json.loads(raw_response.content.decode("utf-8"))

        try:
            content = response["choices"][0]["message"]["content"]
            flag = True
        except:
            content = response["error"]["code"]
            flag = False
        return flag, content

    def generate(self, url, api_key, new_message, role=None, args=None):
        if role is not None:
            role = {
                "role": "system",
                "content": role,
            }
        if len(self.history_message) == 0 and role is not None:
            self.history_message.append(role)
        temp_message = self.history_message
        temp_message.append({"role": "user", "content": new_message})
        parameters = {"model": self.model_name, "messages": temp_message}
        if args is not None:
            for key in args.keys():
                parameters[key] = args[key]
        flag, response = self.request_chatgpt(url, api_key, parameters)
        if flag == True:
            self.history_message.append({"role": "assistant", "content": response})
        return flag, response