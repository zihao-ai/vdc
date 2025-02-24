from typing import List, Optional
from .base import BaseDataPair, QAPair

class VideoTextPair(BaseDataPair):
    """视频文本对类，用于存储和处理视频文本对数据"""
    
    def __init__(self, video_path: str, text: str):
        super().__init__(text)
        self.video_path: str = video_path 