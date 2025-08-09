import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from project_paths import paths


def download_with_retry(model_name, local_path, max_retries=3):
    """多镜像重试下载"""
    mirrors = [
        "https://hf-mirror.com",
        "https://huggingface.co",
        "https://huggingface.co.cn"
    ]

    for retry in range(max_retries):
        for mirror in mirrors:
            try:
                print(f"尝试 {retry + 1}/{max_retries}，镜像: {mirror}")
                os.environ['HF_ENDPOINT'] = mirror

                # 下载tokenizer和模型
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=local_path, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=local_path, torch_dtype=torch.float16,
                    trust_remote_code=True, low_cpu_mem_usage=True,
                    resume_download=True  # 断点续传
                )

                # 保存到本地
                tokenizer.save_pretrained(local_path)
                model.save_pretrained(local_path)

                print(f"{model_name} 下载成功")
                del model
                torch.cuda.empty_cache()
                return True

            except Exception as e:
                print(f"镜像失败: {e}")
                continue

        if retry < max_retries - 1:
            print("等待60秒后重试...")
            time.sleep(60)

    print(f"{model_name} 所有尝试失败")
    return False


def main():
    models = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b"),
        ("microsoft/Phi-3-mini-4k-instruct", "phi3-mini-3.8b"),
        ("meta-llama/Llama-2-7b-chat-hf", "llama2-7b")
    ]

    for model_name, local_name in models:
        local_path = paths.get_model_path(local_name)
        os.makedirs(local_path, exist_ok=True)

        success = download_with_retry(model_name, local_path)
        if not success:
            print(f"{model_name} 下载失败，可继续其他模型")
        print("-" * 50)


if __name__ == "__main__":
    main()