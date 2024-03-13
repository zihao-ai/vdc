import argparse

from LLMs.llm_models.llm_base import LLM_base
from MiniGPT4.minigpt4.common.config import Config
from MiniGPT4.minigpt4.common.dist_utils import get_rank
from MiniGPT4.minigpt4.common.registry import registry
from MiniGPT4.minigpt4.conversation.conversation import CONV_VISION, Chat
from MiniGPT4.minigpt4.datasets.builders import *
from MiniGPT4.minigpt4.models import *
from MiniGPT4.minigpt4.processors import *
from MiniGPT4.minigpt4.runners import *
from MiniGPT4.minigpt4.tasks import *


class MiniGPT4(LLM_base):
    def __init__(self):
        super(MiniGPT4, self).__init__()
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument(
            "--cfg-path",
            type=str,
            default="/workspace2/zhuzihao/mllm/MiniGPT4/eval_configs/minigpt4_eval.yaml",
        )
        parser.add_argument("--gpu-id", type=int, default=0)
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args()

        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))

    def generate_mm(self, image_path, prompt):
        image = self.process_image(image_path)
        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = self.chat.upload_img(image, chat_state, img_list)
        prompt = self.process_prompt(prompt)
        self.chat.ask(prompt, chat_state)
        llm_message = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=2000,
        )[0]
        return llm_message
