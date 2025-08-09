import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from project_paths import paths


class LAMARFramework:
    def __init__(self, alpha1=0.1, alpha2=0.3, threshold=80):
        """
        LAMAR自适应融合框架
        alpha1: 高交互用户的LLM权重（更信任传统模型）
        alpha2: 低交互用户的LLM权重（更信任LLM）
        threshold: 交互数量阈值
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.threshold = threshold

    def calculate_user_interactions(self, train_data, test_data):
        """计算用户交互次数"""
        all_data = pd.concat([train_data, test_data])
        user_interactions = all_data['user_id'].value_counts().to_dict()
        return user_interactions

    def adaptive_merge(self, llm_predictions, deepfm_predictions, user_interactions, test_users):
        """自适应融合预测结果"""
        merged_predictions = []
        stats = {"high_interaction": 0, "low_interaction": 0}

        for i, user_id in enumerate(test_users):
            interaction_count = user_interactions.get(user_id, 0)

            if interaction_count > self.threshold:
                alpha = self.alpha1  # 高交互用户，更信任传统模型
                stats["high_interaction"] += 1
            else:
                alpha = self.alpha2  # 低交互用户，更信任LLM
                stats["low_interaction"] += 1

            merged_rating = alpha * llm_predictions[i] + (1 - alpha) * deepfm_predictions[i]
            merged_predictions.append(merged_rating)

        print(f"融合统计: 高交互{stats['high_interaction']}, 低交互{stats['low_interaction']}")
        return np.array(merged_predictions)

    def evaluate_performance(self, true_ratings, pred_ratings, model_name="Model"):
        """评估性能"""
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae = mean_absolute_error(true_ratings, pred_ratings)

        print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    def run_complete_experiment(self, llm_results, deepfm_predictions, train_data, test_data, true_ratings):
        """运行完整LAMAR实验"""
        user_interactions = self.calculate_user_interactions(train_data, test_data)
        test_users = test_data['user_id'].values

        results = {}

        # 评估DeepFM基线
        deepfm_performance = self.evaluate_performance(true_ratings, deepfm_predictions, "DeepFM基线")
        results["DeepFM"] = {"traditional_only": deepfm_performance}

        # 评估每个LLM模型
        for model_name, llm_preds in llm_results.items():
            print(f"\n{'=' * 30} 评估 {model_name} {'=' * 30}")

            # 纯LLM性能
            llm_performance = self.evaluate_performance(true_ratings, llm_preds, f"{model_name}纯LLM")

            # LAMAR融合性能
            lamar_preds = self.adaptive_merge(llm_preds, deepfm_predictions, user_interactions, test_users)
            lamar_performance = self.evaluate_performance(true_ratings, lamar_preds, f"{model_name}+LAMAR")

            results[model_name] = {
                "llm_only": llm_performance,
                "lamar": lamar_performance,
                "predictions": lamar_preds.tolist()
            }

        return results


if __name__ == "__main__":
    # 加载数据和预测结果
    train_data = pd.read_csv(paths.get_train_data())
    test_data = pd.read_csv(paths.get_test_data())
    true_ratings = np.load(paths.get_result_file('true_ratings.npy'))
    deepfm_preds = np.load(paths.get_result_file('deepfm_predictions.npy'))

    # 加载LLM预测结果
    llm_results = {}
    for model_name in ["TinyLlama", "Phi-3-mini", "Llama-2"]:
        try:
            llm_results[model_name] = np.load(paths.get_result_file(f'{model_name}_predictions.npy'))
        except FileNotFoundError:
            print(f"{model_name}预测结果不存在，跳过")

    if llm_results:
        lamar = LAMARFramework()
        results = lamar.run_complete_experiment(llm_results, deepfm_preds, train_data, test_data, true_ratings)

        # 保存结果
        with open(paths.get_result_file('experiment_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print("\nLAMAR实验完成，结果已保存")
    else:
        print("没有可用的LLM预测结果")