from typing import List, Optional
from .base import BaseDataPair, QAPair

class ImageTextPair(BaseDataPair):
    """图文对类，用于存储和处理图文对数据"""
    
    def __init__(self, img_path: str, text: str):
        super().__init__(text)
        self.img_path: str = img_path 
    
    def __str__(self):
        s=f"Image_path={self.img_path}, text={self.text}"
        for i, qa in enumerate(self.qa_pairs):
            s+=f"\n{i+1}. Question: {qa.question}\nExpected Answer: {qa.expected_answer}\nActual Answer: {qa.actual_answer}\nMatched: {qa.is_matched}\n"
        s+=f"\nConsistency Score: {self.consistency_score}\nIs Consistent: {self.is_consistent}"
        return s
    
    def __repr__(self):
        return self.__str__()
    
    