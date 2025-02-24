from typing import Optional, Union, Tuple
from .data.image_text_pair import ImageTextPair
from .data.video_text_pair import VideoTextPair
from .models.openai_api import generate_text_with_images, generate_text, generate_text_with_video
from .prompts import (
    ImageQGPrompt,
    VideoQGPrompt,
    ImageQAPrompt,
    VideoQAPrompt,
    ImageBatchQAPrompt,
    VideoBatchQAPrompt,
)
from .utils.config import VDCConfig


class DataCleanser:
    """数据清洗器类"""

    def __init__(
        self,
        config: Optional[VDCConfig] = None,
        llm_model: str = "gpt-4o-mini",
        mllm_model: str = "gpt-4o-mini",
    ):
        if config is None:
            self.config = VDCConfig.from_env()
        else:
            self.config = config
        self.config.validate()

        self.llm_model = llm_model
        self.mllm_model = mllm_model

    def process_image_text_pair(
        self, img_path: str, text: str, num_questions: int, batch_qa_size: int = 1
    ) -> ImageTextPair:
        """处理图文对数据

        Args:
            img_path: 图片路径
            text: 文本内容
            num_questions: 需要生成的问题数量
            batch_qa_size: 批量处理的问题数量
        Returns:
            ImageTextPair: 处理后的图文对实例
        """
        # 创建图文对实例
        pair = ImageTextPair(img_path, text)

        # 使用LLM生成问题和预期答案
        print(f"Generating questions and expected answers for image: {img_path}")
        qg_prompt = ImageQGPrompt.format(text=text, num_questions=num_questions)
        qa_pairs_text = generate_text(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            model=self.llm_model,
            prompt=qg_prompt,
        )

        # 解析生成的问答对并添加到pair中
        qa_pairs = self._parse_question_generation_results(qa_pairs_text)
        for i, (question, expected_answer) in enumerate(qa_pairs):
            if expected_answer.lower() in ["yes", "no"]:
                pair.add_qa_pair(question, expected_answer)

        print(f"There are {len(pair.qa_pairs)} valid QA pairs:")
        for i, qa_pair in enumerate(pair.qa_pairs):
            print(f"\t{i+1}. Q: {qa_pair.question} A: {qa_pair.expected_answer}")

        if batch_qa_size == -1:
            batch_qa_size = len(pair.qa_pairs)

        # if batch_qa_size is 1, 顺序处理所有问题
        if batch_qa_size == 1:
            results = []
            print(f">>>>>>>>>>> Answering all {len(pair.qa_pairs)} QA pairs")
            for i, qa_pair in enumerate(pair.qa_pairs):
                # 使用MLLM回答问题
                qa_prompt = ImageQAPrompt.format(question=qa_pair.question)
                actual_answer = generate_text_with_images(
                    base_url=self.config.mllm_base_url,
                    api_key=self.config.mllm_api_key,
                    model=self.mllm_model,
                    prompt=qa_prompt,
                    img_path_list=[img_path],
                )
                # 判断是否回答正确
                is_matched = qa_pair.expected_answer.lower()  in actual_answer.lower()

                print(f"\n{i}. Question: {qa_pair.question}")
                print(f"   Expected: {qa_pair.expected_answer}")
                print(f"   Actual: {actual_answer}")
                print(f"   Matched: {is_matched}")

                results.append((actual_answer, is_matched))
        else:
            results = []
            print(f">>>>>>>>>>> Answering {batch_qa_size} QA pairs in each batch")
            for i in range(0, len(pair.qa_pairs), batch_qa_size):
                batch_qa_pairs = pair.qa_pairs[i : i + batch_qa_size]
                qa_prompt = ImageBatchQAPrompt.format(questions=[qa_pair.question for qa_pair in batch_qa_pairs])
                actual_answers = generate_text_with_images(
                    base_url=self.config.mllm_base_url,
                    api_key=self.config.mllm_api_key,
                    model=self.mllm_model,
                    prompt=qa_prompt,
                    img_path_list=[img_path],
                )
                actual_answers = actual_answers.strip().split("\n")
                for j, actual_answer in enumerate(actual_answers):
                    is_matched = batch_qa_pairs[j].expected_answer.lower() in actual_answer.lower()

                    print(f"\n{i+j}. Question: {batch_qa_pairs[j].question}")
                    print(f"   Expected: {batch_qa_pairs[j].expected_answer}")
                    print(f"   Actual: {actual_answer}")
                    print(f"   Matched: {is_matched}")

                    results.append((actual_answer, is_matched))

        # 更新结果
        for i, (actual_answer, is_matched) in enumerate(results):
            pair.update_qa_result(i, actual_answer, is_matched)

        # 计算一致性得分
        pair.calculate_consistency_score()
        return pair

    def process_video_text_pair(
        self, video_path: str, text: str, num_questions: int, batch_qa_size: int = -1, frame_interval: int = 50
    ) -> VideoTextPair:
        """处理视频文本对数据

        Args:
            video_path: 视频路径
            text: 文本内容
            num_questions: 需要生成的问题数量
            batch_qa_size: 批量处理的问题数量
            frame_interval: 抽取帧的间隔
        Returns:
            VideoTextPair: 处理后的视频文本对实例
        """
        # 创建视频文本对实例
        pair = VideoTextPair(video_path, text)

        # 使用LLM生成问题和预期答案
        print(f">>>>>>>>>>> Generating questions and expected answers for video: {video_path}")
        qg_prompt = VideoQGPrompt.format(text=text, num_questions=num_questions)
        qa_pairs_text = generate_text(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            model=self.llm_model,
            prompt=qg_prompt,
        )

        # 解析生成的问答对并添加到pair中
        qa_pairs = self._parse_question_generation_results(qa_pairs_text)
        for i, (question, expected_answer) in enumerate(qa_pairs):
            if expected_answer.lower() in ["yes", "no"]:
                pair.add_qa_pair(question, expected_answer)

        print(f"There are {len(pair.qa_pairs)} valid QA pairs:")
        # for i, qa_pair in enumerate(pair.qa_pairs):
        #     print(f"\t{i+1}. Q: {qa_pair.question} A: {qa_pair.expected_answer}")

        # if batch_qa_size is 1, 顺序处理所有问题
        if batch_qa_size == -1:
            batch_qa_size = len(pair.qa_pairs)

        if batch_qa_size == 1:
            results = []
            print(f">>>>>>>>>>> Answering all {len(pair.qa_pairs)} QA pairs")
            for i, qa_pair in enumerate(pair.qa_pairs):
                # 使用MLLM回答问题
                qa_prompt = VideoQAPrompt.format(question=qa_pair.question)
                actual_answer = generate_text_with_video(
                    base_url=self.config.mllm_base_url,
                    api_key=self.config.mllm_api_key,
                    model=self.mllm_model,
                    prompt=qa_prompt,
                    video_path=video_path,
                    frame_interval=frame_interval,
                )
                # 判断是否回答正确
                is_matched = qa_pair.expected_answer.lower() in actual_answer.lower()

                print(f"\n{i}. Question: {qa_pair.question}")
                print(f"   Expected: {qa_pair.expected_answer}")
                print(f"   Actual: {actual_answer}")
                print(f"   Matched: {is_matched}")

                results.append((actual_answer, is_matched))
        else:
            results = []
            print(f">>>>>>>>>>> Answering {batch_qa_size} QA pairs in each batch")
            for i in range(0, len(pair.qa_pairs), batch_qa_size):
                batch_qa_pairs = pair.qa_pairs[i : i + batch_qa_size]
                qa_prompt = VideoBatchQAPrompt.format(questions=[qa_pair.question for qa_pair in batch_qa_pairs])
                actual_answers = generate_text_with_video(
                    base_url=self.config.mllm_base_url,
                    api_key=self.config.mllm_api_key,
                    model=self.mllm_model,
                    prompt=qa_prompt,
                    video_path=video_path,
                    frame_interval=frame_interval,
                )
                actual_answers = actual_answers.strip().split("\n")
                for j, actual_answer in enumerate(actual_answers):
                    is_matched = batch_qa_pairs[j].expected_answer.lower() in actual_answer.lower()

                    print(f"\n{i+1+j}. Question: {batch_qa_pairs[j].question}")
                    print(f"   Expected: {batch_qa_pairs[j].expected_answer}")
                    print(f"   Actual: {actual_answer}")
                    print(f"   Matched: {is_matched}")

                    results.append((actual_answer, is_matched))

        # 更新结果
        for i, (actual_answer, is_matched) in enumerate(results):
            pair.update_qa_result(i, actual_answer, is_matched)

        # 计算一致性得分
        pair.calculate_consistency_score()
        return pair

    def _parse_question_generation_results(self, qa_pairs_text: str) -> list[tuple[str, str]]:
        """Parse the QA pairs text from LLM output

        Args:
            qa_pairs_text: LLM generated QA pairs text, each line in format "Q: question A: Yes"

        Returns:
            list[tuple[str, str]]: List of (question, answer) pairs
        """
        qa_pairs = []

        for line in qa_pairs_text.strip().split("\n"):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Split line into question and answer parts
            try:
                # Find the position of "A:" to split the line
                answer_start = line.find("A:")
                if answer_start == -1:
                    continue

                # Extract question and answer
                question = line[2:answer_start].strip()  # Remove "Q:" and trailing spaces
                answer = line[answer_start + 2 :].strip()  # Remove "A:" and spaces

                if question and answer:
                    qa_pairs.append((question, answer))
            except Exception:
                # Skip malformed lines
                continue
        return qa_pairs
