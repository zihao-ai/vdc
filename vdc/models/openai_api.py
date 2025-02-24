import openai
import base64
import os
import cv2


def get_mime_type(file_path: str) -> str:
    """根据文件扩展名获取MIME类型"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")  # 默认返回 image/jpeg


def encode_image(image_path: str) -> str:
    """将图像编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_text(base_url, api_key, model, prompt):
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
        top_p=1.0,
    )
    return response.choices[0].message.content.strip()


def generate_text_with_images(base_url, api_key, model, prompt, img_path_list):
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    content = [{"type": "text", "text": prompt}]

    for img_path in img_path_list:
        base64_image = encode_image(img_path)
        mime_type = get_mime_type(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=1000, top_p=1.0
    )
    return response.choices[0].message.content.strip()


def generate_text_with_video(base_url, api_key, model, prompt, video_path, frame_interval=50):
    """
    从视频生成文本描述

    Args:
        base_url: OpenAI API的基础URL
        api_key: OpenAI API密钥
        model: 使用的模型名称
        prompt: 提示文本
        video_path: 视频文件路径
        frame_interval: 抽取帧的间隔(默认每50帧取一帧)

    Returns:
        str: 生成的文本描述
    """
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    # 读取视频帧
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    frame_count = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        # 每frame_interval帧保存一帧
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            base64Frames.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        frame_count += 1

    video.release()

    # 构建消息
    content = [{"type": "text", "text": prompt}]
    content.extend(base64Frames)

    messages = [{"role": "user", "content": content}]

    # 调用API
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=512, top_p=1.0
    )

    return response.choices[0].message.content.strip()

