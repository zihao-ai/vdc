from vdc.cleanser import DataCleanser
from vdc.utils.config import VDCConfig

config = VDCConfig(
    llm_base_url="https://hk.uniapi.io/v1/",
    llm_api_key="sk-xxx",
    mllm_base_url="https://hk.uniapi.io/v1/",
    mllm_api_key="sk-xxx",
)

cleanser = DataCleanser(config=config, llm_model="gpt-4o-mini", mllm_model="gpt-4o-mini")

res = cleanser.process_image_text_pair(
    img_path="example/test.png",
    text="cat",
    num_questions=5,
    batch_qa_size=-1
)

consistency_score = res.consistency_score
is_consistent = res.is_consistent

print(res)
