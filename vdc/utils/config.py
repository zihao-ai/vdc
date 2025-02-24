import os
from dataclasses import dataclass


@dataclass
class VDCConfig:
    """配置类"""

    llm_base_url: str
    llm_api_key: str
    mllm_base_url: str
    mllm_api_key: str
    llm_model: str = "gpt-4"  # 默认LLM模型
    mllm_model: str = "gpt-4-vision-preview"  # 默认MLLM模型

    @staticmethod
    def from_env():
        """从环境变量中读取配置"""
        return VDCConfig(
            llm_base_url=os.getenv("VDC_LLM_BASE_URL"),
            llm_api_key=os.getenv("VDC_LLM_API_KEY"),
            mllm_base_url=os.getenv("VDC_MLLM_BASE_URL"),
            mllm_api_key=os.getenv("VDC_MLLM_API_KEY"),
            llm_model=os.getenv("VDC_LLM_MODEL", "gpt-4"),
            mllm_model=os.getenv("VDC_MLLM_MODEL", "gpt-4-vision-preview"),
        )

    def validate(self):
        """验证配置是否完整"""
        required_vars = {
            "VDC_LLM_BASE_URL": self.llm_base_url,
            "VDC_LLM_API_KEY": self.llm_api_key,
            "VDC_MLLM_BASE_URL": self.mllm_base_url,
            "VDC_MLLM_API_KEY": self.mllm_api_key,
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
