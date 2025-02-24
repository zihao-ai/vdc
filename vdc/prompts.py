class ImageQGPrompt:
    """Template for generating image-text pair verification questions"""

    @staticmethod
    def format(text: str, num_questions: int) -> str:
        return f"""Based on the following text caption of one image, generate {num_questions} yes/no questions to verify the consistency between the text and an image. The questions should:
1. Cover all key information points from the text.  
2. Be clear and specific.
3. Be answerable by observing an image.
4. Be diverse and not repeat the same information.
5. The expected answer must be "yes" since we assume the text and image are consistent.
6. Use "Q:" and "A:" format. Each line should contain a single question and answer pair.

Text Caption:
{text}

Example output format:
Q: Is there [something mentioned in text] in the image? A: Yes
Q: Does the image contain a [something mentioned in text]? A: Yes

Please only output the Q&A pairs, do not include any other text."""


class ImageQAPrompt:
    """Template for image-based question answering"""

    @staticmethod
    def format(question: str) -> str:
        return f"""Please carefully observe the image and answer the following question. Your answer should be:
1. Start with a clear "Yes" or "No", followed by a brief explanation.
2. Accurate and objective
3. Based only on visible information in the image

Question: {question}

Answer:"""


class VideoQGPrompt:
    """Template for video-text pair question generation"""

    @staticmethod
    def format(text: str, num_questions: int) -> str:
        return f"""Based on the following text description of one video, generate {num_questions} yes/no questions to verify the consistency between the text and the video. The questions should:
1. Cover key information points from the text, including actions, events and temporal relationships
2. Be clear and specific
3. Be answerable by watching the video
4. Be diverse and not repeat the same information
5. The expected answer must be "yes" since we assume the text and video are consistent
6. Consider temporal aspects and sequence of events where relevant
7. Use "Q:" and "A:" format. Each line should contain a single question and answer pair

Text Description:
{text}

Example output format:
Q: Does the video show [something mentioned in text]? A: Yes
Q: Can we see [an action/event mentioned in text] happening in the video? A: Yes

Please only output the Q&A pairs, do not include any other text."""


class VideoQAPrompt:
    """Template for video-based question answering"""

    @staticmethod
    def format(question: str) -> str:
        return f"""Please carefully observe the frames from a video and answer the following question. Your answer should be:
1. Start with a clear "Yes" or "No", followed by a brief explanation.
2. Accurate and objective
3. Based only on visible information and events in the video
4. Consider temporal aspects and sequence of events where relevant

Question: {question}

Answer:"""


class ImageBatchQAPrompt:
    """Template for batch image-based question answering"""

    @staticmethod
    def format(questions: list) -> str:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return f"""Please carefully observe the image and answer the following questions. Your answers should be:
1. Start with a clear "Yes" or "No", followed by a brief explanation.
2. Accurate and objective
3. Based only on visible information in the image
5. One answer per line, in the same order as questions

Questions:
{questions_text}

Please provide answers in the following format:
1. [Answer to first question]
2. [Answer to second question]
etc.

Answers:"""


class VideoBatchQAPrompt:
    """Template for batch video-based question answering"""

    @staticmethod
    def format(questions: list) -> str:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return f"""Please carefully observe the frames from a video and answer the following questions. Your answers should be:
1. Start with a clear "Yes" or "No", followed by a brief explanation.
2. Accurate and objective
3. Based only on visible information and events in the video
4. Consider temporal aspects and sequence of events where relevant
5. One answer per line, in the same order as questions

Questions:
{questions_text}

Please provide answers in the following format:
1. [Answer to first question]
2. [Answer to second question]
etc.

Answers:"""
