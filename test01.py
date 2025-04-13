import base64
import requests


def process_image(api_url, input_image_path):
    # 1. 转换图片为 Base64
    with open(input_image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    # 2. 发送请求
    payload = {"image": base64_image, "task": "enhance_image"}
    response = requests.post(api_url, json=payload)
    response_data = response.json()

    # 3. 检查返回的是 Base64 还是 URL
    if "output_image" in response_data["result"]:  # Base64 情况
        output_base64 = response_data["result"]["output_image"]
        if output_base64.startswith("data:image"):
            output_base64 = output_base64.split(",")[1]
        output_bytes = base64.b64decode(output_base64)
        with open("output.jpg", "wb") as f:
            f.write(output_bytes)
        print("图片已保存为 output.jpg")

    elif "image_url" in response_data["result"]:  # URL 情况
        image_url = response_data["result"]["image_url"]
        image_data = requests.get(image_url).content
        with open("downloaded_image.jpg", "wb") as f:
            f.write(image_data)
        print("图片已下载为 downloaded_image.jpg")


# 使用示例
process_image("https://api.doubao.com/vision", "test.jpg")