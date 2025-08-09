import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from project_paths import paths


class ResultAnalyzer:
    def __init__(self):
        results_file = paths.get_result_file('experiment_results.json')
        with open(results_file, 'r') as f:
            self.results = json.load(f)

    def create_performance_summary(self):
        """创建性能汇总表"""
        summary_data = []

        # DeepFM基线
        if "DeepFM" in self.results:
            deepfm_result = self.results["DeepFM"]["traditional_only"]
            summary_data.append({
                "Model": "DeepFM",
                "Type": "Traditional",
                "RMSE": deepfm_result["rmse"],
                "MAE": deepfm_result["mae"]
            })

        # LLM模型结果
        for model_name in ["TinyLlama", "Phi-3-mini", "Llama-2"]:
            if model_name in self.results:
                # 纯LLM
                llm_result = self.results[model_name]["llm_only"]
                summary_data.append({
                    "Model": model_name,
                    "Type": "LLM-Only",
                    "RMSE": llm_result["rmse"],
                    "MAE": llm_result["mae"]
                })

                # LAMAR融合
                lamar_result = self.results[model_name]["lamar"]
                summary_data.append({
                    "Model": f"{model_name}+LAMAR",
                    "Type": "Hybrid",
                    "RMSE": lamar_result["rmse"],
                    "MAE": lamar_result["mae"]
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(paths.get_result_file('performance_summary.csv'), index=False)

        print("性能汇总表:")
        print(summary_df.to_string(index=False))
        return summary_df

    def plot_model_comparison(self):
        """绘制性能对比图"""
        summary_df = pd.read_csv(paths.get_result_file('performance_summary.csv'))

        plt.rcParams['font.size'] = 12
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # RMSE对比
        colors = ['skyblue' if t == 'Traditional' else 'lightcoral' if t == 'LLM-Only' else 'lightgreen'
                  for t in summary_df['Type']]

        bars1 = ax1.bar(range(len(summary_df)), summary_df['RMSE'], color=colors)
        ax1.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')

        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=10)

        # MAE对比
        bars2 = ax2.bar(range(len(summary_df)), summary_df['MAE'], color=colors)
        ax2.set_title('MAE Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.set_xticks(range(len(summary_df)))
        ax2.set_xticklabels(summary_df['Model'], rotation=45, ha='right')

        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(paths.get_result_file('model_comparison.png'), dpi=300, bbox_inches='tight')
        print("对比图已保存: model_comparison.png")
        plt.show()

    def analyze_scaling_effects(self):
        """分析模型规模效应"""
        model_sizes = {"TinyLlama": 1.1, "Phi-3-mini": 3.8, "Llama-2": 7.0}

        scaling_data = []
        for model_name in ["TinyLlama", "Phi-3-mini", "Llama-2"]:
            if model_name in self.results:
                lamar_result = self.results[model_name]["lamar"]
                scaling_data.append({
                    "Model": model_name,
                    "Size_B": model_sizes[model_name],
                    "RMSE": lamar_result["rmse"],
                    "MAE": lamar_result["mae"]
                })

        if len(scaling_data) >= 2:
            scaling_df = pd.DataFrame(scaling_data)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(scaling_df['Size_B'], scaling_df['RMSE'], 'bo-', linewidth=2, markersize=8)
            for _, row in scaling_df.iterrows():
                plt.annotate(row['Model'], (row['Size_B'], row['RMSE']),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            plt.xlabel('Model Size (Billion Parameters)')
            plt.ylabel('RMSE')
            plt.title('Model Size vs RMSE')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(scaling_df['Size_B'], scaling_df['MAE'], 'ro-', linewidth=2, markersize=8)
            for _, row in scaling_df.iterrows():
                plt.annotate(row['Model'], (row['Size_B'], row['MAE']),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            plt.xlabel('Model Size (Billion Parameters)')
            plt.ylabel('MAE')
            plt.title('Model Size vs MAE')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(paths.get_result_file('scaling_effects.png'), dpi=300, bbox_inches='tight')
            print("规模效应图已保存: scaling_effects.png")
            plt.show()

            print("\n模型规模效应:")
            print(scaling_df.to_string(index=False))
            return scaling_df
        else:
            print("数据不足，无法分析规模效应")
            return None

    def generate_paper_comparison(self):
        """与原论文结果对比"""
        paper_results = {
            "DeepFM (Paper)": {"rmse": 1.1204, "mae": 0.8729},
            "DeepFM+LAMAR (Paper)": {"rmse": 1.0778, "mae": 0.8399}
        }

        comparison_data = []

        # 论文结果
        for model, metrics in paper_results.items():
            comparison_data.append({
                "Model": model,
                "Source": "Paper",
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"]
            })

        # 我们的结果
        if "DeepFM" in self.results:
            our_deepfm = self.results["DeepFM"]["traditional_only"]
            comparison_data.append({
                "Model": "DeepFM (Ours)",
                "Source": "Ours",
                "RMSE": our_deepfm["rmse"],
                "MAE": our_deepfm["mae"]
            })

        # 最佳LAMAR结果
        best_lamar_rmse = float('inf')
        best_model = None
        for model_name in ["TinyLlama", "Phi-3-mini", "Llama-2"]:
            if model_name in self.results:
                rmse = self.results[model_name]["lamar"]["rmse"]
                if rmse < best_lamar_rmse:
                    best_lamar_rmse = rmse
                    best_model = model_name

        if best_model:
            our_lamar = self.results[best_model]["lamar"]
            comparison_data.append({
                "Model": f"{best_model}+LAMAR (Ours)",
                "Source": "Ours",
                "RMSE": our_lamar["rmse"],
                "MAE": our_lamar["mae"]
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(paths.get_result_file('paper_comparison.csv'), index=False)

        print("\n与原论文对比:")
        print(comparison_df.to_string(index=False))

        if best_model:
            paper_rmse = 1.0778
            our_rmse = our_lamar["rmse"]
            diff_percent = abs(our_rmse - paper_rmse) / paper_rmse * 100
            print(f"\n复现质量评估:")
            print(f"论文最佳RMSE: {paper_rmse:.4f}")
            print(f"我们最佳RMSE: {our_rmse:.4f}")
            print(f"相对差异: {diff_percent:.2f}%")

        return comparison_df


if __name__ == "__main__":
    analyzer = ResultAnalyzer()

    print("开始结果分析...\n")

    # 1. 性能汇总
    summary = analyzer.create_performance_summary()

    # 2. 绘制对比图
    analyzer.plot_model_comparison()

    # 3. 规模效应分析
    analyzer.analyze_scaling_effects()

    # 4. 与论文对比
    analyzer.generate_paper_comparison()

    print("\n结果分析完成，所有图表已保存到results目录")