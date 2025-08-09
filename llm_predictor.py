from llm_interface import UnifiedLLMInterface
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import time
from project_paths import paths


class InferenceTimeEstimator:
    def __init__(self):
        # 基于2×16GB GPU的预估速度
        self.speed_benchmarks = {
            "TinyLlama": 3.0,  # 样本/秒
            "Phi-3-mini": 1.5,
            "Llama-2": 0.3
        }

    def estimate_time(self, model_name, test_size):
        speed = self.speed_benchmarks.get(model_name, 1.0)
        hours = test_size / speed / 3600
        print(f"{model_name} 预估时间：{hours:.1f}小时（{test_size}样本）")
        return hours


class LLMRatingPredictor:
    def __init__(self):
        self.llm_interface = UnifiedLLMInterface()
        self.model_configs = {
            "TinyLlama": paths.get_model_path("tinyllama-1.1b"),
            "Phi-3-mini": paths.get_model_path("phi3-mini-3.8b"),
            "Llama-2": paths.get_model_path("llama2-7b")
        }

    def create_rating_prompt(self, user_history, target_movie):
        """使用原论文完整prompt格式"""
        history_str = ""
        for idx, (movie_name, rating) in enumerate(user_history[:20], 1):
            history_str += f"{idx}. {movie_name}, {rating};\n"

        prompt = f"""Below is an instruction that describes a rating prediction task, and the information for the task. Write a response that appropriately completes the instruction. The format of the answer must be consistent with the example, but the information in the example is fake information and shouldn't be used.

### Instruction:
Based on the rating history below, please predict user's rating for the movie: "{target_movie}". The output must predict the user's rating (1 being lowest and 5 being highest), and then explain the reasons why the user will give the rating. While predicting the user's rating for the movie, the user's preference and the features of the movie should be considered. An example of the format of the answer has been given as below

### Information:
Here is user rating history:
{history_str}

### Format Example:
title: <Movie Title>
rating: x stars
reasons:
[1] The movie is an <Genre> movie, and the rating history shows that the user like <Genre> movies.
[2] The movie is stared by <Actor/Actress>. She is one of the most popular stars. And the user thought highly of the movies stared by her in the rating history.
[3] The director of the movie is <Director>. People think highly of the director, and the user's rating history shows that he like the director;
[4] The movie talked about <Plot>. But the user rate movies with a sad story low. So the user might dislike the bad ending.

### Answer:
title: {target_movie}
rating: """
        return prompt

    def predict_with_batching(self, model_name, test_data, user_histories, batch_size=200):
        """分批执行推理"""
        estimator = InferenceTimeEstimator()
        estimator.estimate_time(model_name, len(test_data))

        # 加载模型
        model_path = self.model_configs[model_name]
        success = self.llm_interface.load_model(model_path, model_name)
        if not success:
            print(f"{model_name} 加载失败，返回默认值")
            return np.array([3.0] * len(test_data))

        all_predictions = []
        total_batches = (len(test_data) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test_data))
            batch_data = test_data.iloc[start_idx:end_idx]

            print(f"处理批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")

            batch_predictions = []
            start_time = time.time()

            for _, row in batch_data.iterrows():
                user_id = row['user_id']
                movie_name = row['Name']
                user_history = user_histories.get(user_id, [])

                prompt = self.create_rating_prompt(user_history, movie_name)
                try:
                    response = self.llm_interface.generate_rating_prediction(prompt)
                    rating = self.llm_interface.extract_rating_from_output(response)
                    batch_predictions.append(rating)
                except Exception as e:
                    batch_predictions.append(3.0)

            batch_time = time.time() - start_time
            print(f"批次耗时: {batch_time:.1f}s, 平均: {batch_time / len(batch_data):.2f}s/样本")

            all_predictions.extend(batch_predictions)

            # 每5批保存一次中间结果
            if batch_idx % 5 == 0:
                np.save(paths.get_result_file(f'{model_name}_partial_{batch_idx}.npy'),
                        np.array(all_predictions))

        return np.array(all_predictions)

    def build_user_histories(self, train_data):
        """构建用户历史"""
        user_histories = {}
        for _, row in train_data.iterrows():
            user_id = row['user_id']
            movie_name = row['Name']
            rating = row['rating']

            if user_id not in user_histories:
                user_histories[user_id] = []
            user_histories[user_id].append((movie_name, rating))

        # 保留最近20个交互
        for user_id in user_histories:
            user_histories[user_id] = user_histories[user_id][-20:]

        return user_histories

    def run_all_models(self, test_data, user_histories):
        """运行所有模型预测"""
        results = {}

        for model_name in ["TinyLlama", "Phi-3-mini", "Llama-2"]:
            print(f"\n{'=' * 50}")
            print(f"开始{model_name}预测")
            print('=' * 50)

            predictions = self.predict_with_batching(model_name, test_data, user_histories)
            results[model_name] = predictions

            # 保存结果
            np.save(paths.get_result_file(f'{model_name}_predictions.npy'), predictions)

            # 清理GPU内存
            if hasattr(self.llm_interface, 'current_model') and self.llm_interface.current_model:
                del self.llm_interface.current_model
                self.llm_interface.current_model = None
            torch.cuda.empty_cache()

        return results


if __name__ == "__main__":
    # 加载数据
    train_data = pd.read_csv(paths.get_train_data())
    test_data = pd.read_csv(paths.get_test_data())

    predictor = LLMRatingPredictor()
    user_histories = predictor.build_user_histories(train_data)

    print(f"构建了{len(user_histories)}个用户的历史记录")
    results = predictor.run_all_models(test_data, user_histories)
    print("LLM预测全部完成")