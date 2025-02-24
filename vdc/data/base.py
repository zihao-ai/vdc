from typing import Optional, List
from dataclasses import dataclass

@dataclass
class QAPair:
    """存储问答对的数据类"""
    question: str
    expected_answer: str
    actual_answer: Optional[str] = None
    is_matched: Optional[bool] = None

class BaseDataPair:
    """数据对的基类"""
    def __init__(self, text: str):
        self.text: str = text
        self.qa_pairs: List[QAPair] = []
        self.consistency_score: float = 0.0
        self.is_consistent: Optional[bool] = None
    
    def add_qa_pair(self, question: str, expected_answer: str):
        """添加一个问答对"""
        self.qa_pairs.append(QAPair(question=question, expected_answer=expected_answer))
    
    def update_qa_result(self, index: int, actual_answer: str, is_matched: bool):
        """更新问答结果"""
        if 0 <= index < len(self.qa_pairs):
            self.qa_pairs[index].actual_answer = actual_answer
            self.qa_pairs[index].is_matched = is_matched
    
    def calculate_consistency_score(self):
        """计算一致性得分"""
        if not self.qa_pairs:
            return 0.0
        matched_count = sum(1 for qa in self.qa_pairs if qa.is_matched)
        self.consistency_score = matched_count / len(self.qa_pairs)
        return self.consistency_score 

    def is_consistent(self, threshold: float = 0.5  ):
        """判断是否一致"""
        self.is_consistent = self.consistency_score >= threshold
        return self.is_consistent