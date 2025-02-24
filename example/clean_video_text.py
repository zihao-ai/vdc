from vdc.cleanser import DataCleanser
from vdc.utils.config import VDCConfig

config = VDCConfig(
    llm_base_url="https://hk.uniapi.io/v1/",
    llm_api_key="sk-T0e7qj7yQJEKXDu_srXOAQWQ4dY0ucJrda9enRj5nvbA3rYwKwvSl8CApVA",
    mllm_base_url="https://hk.uniapi.io/v1/",
    mllm_api_key="sk-T0e7qj7yQJEKXDu_srXOAQWQ4dY0ucJrda9enRj5nvbA3rYwKwvSl8CApVA",
)

cleanser = DataCleanser(config=config, llm_model="gpt-4o-mini", mllm_model="gpt-4o-mini")

res = cleanser.process_video_text_pair(
    video_path="example/test_video.mp4",
    text="It shows a wooden table with several items: a bouquet of flowers wrapped in newspaper, some fruits including tomatoes in a clear plastic container, and some other fruits (possibly mangoes on a white plate.",
    num_questions=10,
    frame_interval=50,
    batch_qa_size=-1
)

consistency_score = res.consistency_score
is_consistent = res.is_consistent

print(res)
