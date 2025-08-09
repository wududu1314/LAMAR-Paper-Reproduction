import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import gc
from project_paths import paths


class UnifiedLLMInterface:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None

    def load_model(self, model_path, model_name):
        """自适应加载模型（显存不足时8bit量化）"""
        # 清理之前的模型
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        print(f"加载模型：{model_name}")

        # 加载tokenizer
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.current_tokenizer.pad_token is None:
            self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

        # 尝试fp16加载
        try:
            if "7b" in model_name.lower():
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16,
                    device_map="auto", low_cpu_mem_usage=True
                )
            else:
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16
                ).to("cuda")

            print(f"{model_name} fp16加载成功")
            self.current_model_name = model_name
            return True

        except torch.cuda.OutOfMemoryError:
            print(f"显存不足，尝试8bit量化...")
            torch.cuda.empty_cache()

            try:
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path, load_in_8bit=True,
                    device_map="auto", low_cpu_mem_usage=True
                )
                print(f"{model_name} 8bit加载成功")
                self.current_model_name = model_name
                return True

            except Exception as e:
                print(f"{model_name} 加载失败: {e}")
                return False

        except Exception as e:
            print(f"{model_name} 加载失败: {e}")
            return False

    def generate_rating_prediction(self, prompt):
        """生成评分预测"""
        inputs = self.current_tokenizer(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=1500
        )

        if not "7b" in self.current_model_name.lower():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.current_tokenizer.pad_token_id,
            "repetition_penalty": 1.15
        }

        with torch.inference_mode():
            generated = self.current_model.generate(
                inputs["input_ids"], attention_mask=inputs["attention_mask"],
                **generation_config
            )

        output = self.current_tokenizer.decode(generated[0], skip_special_tokens=True)
        response = output[len(prompt):].strip()
        return response

    def extract_rating_from_output(self, output):
        """从输出中解析评分"""
        patterns = [
            r"rating:\s*(\d+(?:\.\d+)?)\s*stars?",
            r"rating:\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*stars?",
            r"rate.*?(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output.lower())
            if match:
                rating = float(match.group(1))
                if 1 <= rating <= 5:
                    return rating

        return 3.0  # 默认值