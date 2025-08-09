import os
import sys
import subprocess
import time
import psutil
import torch
from datetime import datetime, timedelta
from project_paths import paths


class ExperimentRunner:
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.experiment_mode = None

    def select_experiment_mode(self):
        """选择实验模式"""
        print("请选择实验模式：")
        print("1. 快速实验（50样本，约10分钟，验证流程）")
        print("2. 完整实验（全量数据，4-8小时）")

        while True:
            choice = input("请输入选择 (1或2): ").strip()
            if choice == "1":
                self.experiment_mode = "quick"
                print("已选择快速实验模式")
                return "quick"
            elif choice == "2":
                self.experiment_mode = "full"
                print("已选择完整实验模式")
                return "full"
            else:
                print("无效选择，请输入1或2")

    def get_steps(self, mode):
        """根据模式返回执行步骤"""
        if mode == "quick":
            return [
                ("python data_preprocessor.py quick", "数据预处理（快速模式）", True, 1),
                ("python deepfm_model.py", "DeepFM模型训练", True, 5),
                ("python llm_predictor.py", "LLM评分预测", True, 3),
                ("python lamar_framework.py", "LAMAR融合实验", True, 1),
                ("python result_analyzer.py", "结果分析", False, 1)
            ]
        else:
            return [
                ("python data_preprocessor.py", "数据预处理（完整模式）", True, 3),
                ("python deepfm_model.py", "DeepFM模型训练", True, 15),
                ("python llm_predictor.py", "LLM评分预测", True, 240),  # 4小时
                ("python lamar_framework.py", "LAMAR融合实验", True, 5),
                ("python result_analyzer.py", "结果分析", False, 2)
            ]

    def monitor_system(self):
        """系统资源监控"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        gpu_info = ""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_percent = (gpu_memory / gpu_total) * 100
            gpu_info = f", GPU: {gpu_memory:.1f}/{gpu_total:.1f}GB ({gpu_percent:.1f}%)"

        # 磁盘使用情况
        disk_usage = psutil.disk_usage('.')
        disk_free = disk_usage.free / 1024 ** 3

        print(f"系统状态: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%{gpu_info}, 磁盘剩余 {disk_free:.1f}GB")

        # 温度警告（如果CPU使用率过高）
        if cpu_percent > 80:
            print("警告: CPU使用率较高，可能影响性能")
        if memory.percent > 85:
            print("警告: 内存使用率较高")

    def estimate_remaining_time(self, current_step, total_steps, elapsed_time, step_estimates):
        """估算剩余时间"""
        if current_step == 0:
            return "计算中..."

        # 基于已完成步骤的实际耗时调整估算
        completed_estimate = sum(step_estimates[:current_step])
        remaining_estimate = sum(step_estimates[current_step:])

        if completed_estimate > 0:
            # 根据实际耗时调整系数
            adjustment_factor = elapsed_time / (completed_estimate * 60)  # 转换为分钟
            remaining_minutes = remaining_estimate * adjustment_factor
        else:
            remaining_minutes = remaining_estimate

        if remaining_minutes < 60:
            return f"约{remaining_minutes:.0f}分钟"
        else:
            hours = remaining_minutes / 60
            return f"约{hours:.1f}小时"

    def run_command(self, command, description, critical=True, estimated_minutes=1):
        """执行命令并处理错误"""
        print(f"\n{'=' * 80}")
        print(f"步骤: {description}")
        print(f"命令: {command}")
        print(f"预估耗时: {estimated_minutes}分钟")
        print('=' * 80)

        step_start = time.time()
        self.monitor_system()

        # 显示开始时间和预计完成时间
        start_time_str = datetime.now().strftime("%H:%M:%S")
        estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
        print(f"开始时间: {start_time_str}, 预计完成: {estimated_completion.strftime('%H:%M:%S')}")

        try:
            # 检查脚本文件存在性
            if command.startswith('python '):
                script_name = command.split(' ')[1]
                if not os.path.exists(script_name):
                    print(f"错误: 脚本 {script_name} 不存在!")
                    return not critical

            # 实时监控长时间运行的任务
            if estimated_minutes > 10:  # 超过10分钟的任务
                print("开始长时间任务，将每5分钟显示一次进度...")
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)

                last_monitor = time.time()
                while process.poll() is None:
                    current_time = time.time()
                    if current_time - last_monitor > 300:  # 每5分钟
                        elapsed = (current_time - step_start) / 60
                        progress = min(elapsed / estimated_minutes * 100, 95)
                        print(f"进度更新: 已运行{elapsed:.1f}分钟 ({progress:.1f}%), 继续执行中...")
                        self.monitor_system()
                        last_monitor = current_time
                    time.sleep(30)  # 每30秒检查一次

                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command, stderr)
                result_stdout = stdout

            else:
                # 短时间任务直接执行
                result = subprocess.run(command, shell=True, check=True,
                                        capture_output=True, text=True, timeout=28800)
                result_stdout = result.stdout

            step_time = time.time() - step_start
            self.step_times[description] = step_time
            actual_minutes = step_time / 60

            completion_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n执行成功! 完成时间: {completion_time}")
            print(f"实际耗时: {actual_minutes:.1f}分钟 (预估: {estimated_minutes}分钟)")

            if actual_minutes > estimated_minutes * 1.5:
                print("注意: 实际耗时超出预估较多，后续时间估算将调整")

            # 显示输出预览
            if result_stdout and len(result_stdout.strip()) > 0:
                print("输出预览:")
                lines = result_stdout.strip().split('\n')
                # 显示前3行和后3行
                if len(lines) <= 6:
                    print(result_stdout[-1000:])
                else:
                    print('\n'.join(lines[:3]))
                    if len(lines) > 6:
                        print("...")
                    print('\n'.join(lines[-3:]))

            return True

        except subprocess.TimeoutExpired:
            print(f"执行超时（8小时限制）")
            return False
        except subprocess.CalledProcessError as e:
            print(f"执行失败 (退出码: {e.returncode})")
            if e.stderr:
                print(f"错误信息:\n{e.stderr}")
            return not critical
        except Exception as e:
            print(f"未知错误: {e}")
            return not critical

    def check_prerequisites(self):
        """检查运行前置条件"""
        print("检查前置条件...")

        checks = []

        # Python库检查
        try:
            import torch, transformers, deepctr_torch, pandas, matplotlib, seaborn
            checks.append(("Python库", True))
        except ImportError as e:
            checks.append(("Python库", False, str(e)))

        # CUDA检查
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            checks.append(("CUDA", True, f"{gpu_count}个GPU, {gpu_name}"))
        else:
            checks.append(("CUDA", False, "未检测到CUDA"))

        # 数据文件检查
        data_file = paths.data_dir / 'ml-100k' / 'u.data'
        data_exists = data_file.exists()
        if data_exists:
            import pandas as pd
            try:
                df = pd.read_csv(data_file, sep='\t', nrows=5)
                data_size = len(pd.read_csv(data_file, sep='\t'))
                checks.append(("数据文件", True, f"{data_size}条记录"))
            except:
                checks.append(("数据文件", False, "文件损坏"))
        else:
            checks.append(("数据文件", False, "文件不存在"))

        # 模型文件检查
        model_status = []
        for model_name, model_dir in [("TinyLlama", "tinyllama-1.1b"),
                                      ("Phi-3-mini", "phi3-mini-3.8b"),
                                      ("Llama-2", "llama2-7b")]:
            model_path = paths.get_model_path(model_dir)
            if os.path.exists(model_path):
                # 检查模型文件完整性
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    model_status.append(f"{model_name}✓")
                else:
                    model_status.append(f"{model_name}?")
            else:
                model_status.append(f"{model_name}✗")

        model_count = len([m for m in model_status if '✓' in m])
        checks.append(("预训练模型", model_count > 0, f"{model_count}/3个: {', '.join(model_status)}"))

        # 磁盘空间检查
        free_space = psutil.disk_usage('.').free / (1024 ** 3)
        space_ok = free_space > 10  # 至少10GB
        checks.append(("磁盘空间", space_ok, f"{free_space:.1f}GB剩余"))

        # 内存检查
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        memory_ok = memory_gb > 8  # 至少8GB
        checks.append(("系统内存", memory_ok, f"{memory_gb:.1f}GB总量"))

        # 打印检查结果
        print("\n前置条件检查结果:")
        print("-" * 60)
        all_ok = True
        for check in checks:
            name = check[0]
            status = check[1]
            info = check[2] if len(check) > 2 else ""

            status_symbol = "✓" if status else "✗"
            print(f"{status_symbol} {name:<15}: {info}")
            if not status:
                all_ok = False

        print("-" * 60)
        return all_ok

    def run_experiment(self):
        """运行完整实验"""
        print("=" * 80)
        print("LAMAR项目完整实验")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 选择实验模式
        mode = self.select_experiment_mode()

        # 前置条件检查
        if not self.check_prerequisites():
            print("\n前置条件不满足，请先解决上述问题")
            return False

        # 获取执行步骤
        steps = self.get_steps(mode)

        if mode == "quick":
            print("\n快速实验模式：")
            print("- 测试集：50样本")
            print("- 预计总时间：约10分钟")
            print("- 目的：验证完整流程")
        else:
            print("\n完整实验模式：")
            print("- 测试集：全量数据")
            print("- 预计总时间：4-8小时")
            print("- 目的：获得论文级别完整结果")

        total_estimated = sum(step[3] for step in steps)
        print(f"- 预估总耗时：{total_estimated}分钟")

        input("\n按回车键开始实验...")

        success_count = 0
        experiment_start = time.time()
        step_estimates = [step[3] for step in steps]

        for i, (command, description, critical, estimated_time) in enumerate(steps):
            elapsed_total = (time.time() - experiment_start) / 60
            remaining_time = self.estimate_remaining_time(i, len(steps), elapsed_total, step_estimates)

            print(f"\n[{i + 1}/{len(steps)}] 总进度: {(i / len(steps) * 100):.1f}%, 剩余时间: {remaining_time}")

            success = self.run_command(command, description, critical, estimated_time)
            if success:
                success_count += 1
            elif critical:
                print(f"\n关键步骤失败，实验终止")
                self.print_summary(success_count, len(steps), failed=True)
                return False

        self.print_summary(success_count, len(steps), failed=False)
        return True

    def print_summary(self, success_count, total_steps, failed=False):
        """打印实验总结"""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        if failed:
            print("实验执行失败")
        else:
            print("实验执行完成!")
        print("=" * 80)

        print(f"执行统计:")
        print(f"   成功步骤: {success_count}/{total_steps}")
        print(f"   总耗时: {total_time / 3600:.2f}小时 ({total_time / 60:.1f}分钟)")
        print(f"   完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.step_times:
            print(f"\n各步骤详细耗时:")
            total_step_time = 0
            for step, duration in self.step_times.items():
                total_step_time += duration
                if duration < 60:
                    print(f"   {step:<25}: {duration:.1f}秒")
                elif duration < 3600:
                    print(f"   {step:<25}: {duration / 60:.1f}分钟")
                else:
                    print(f"   {step:<25}: {duration / 3600:.2f}小时")

        if not failed:
            print(f"\n结果文件:")
            result_files = [
                ("experiment_results.json", "实验原始数据"),
                ("performance_summary.csv", "性能汇总表"),
                ("model_comparison.png", "模型对比图"),
                ("scaling_effects.png", "规模效应图"),
                ("paper_comparison.csv", "与论文对比")
            ]

            for filename, description in result_files:
                filepath = paths.get_result_file(filename)
                if os.path.exists(filepath):
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"   ✓ {filename:<25} ({description}, {size_kb:.1f}KB)")
                else:
                    print(f"   ✗ {filename:<25} (未生成)")

            print(f"\n建议下一步:")
            if self.experiment_mode == "quick":
                print("1. 查看results目录验证结果正确性")
                print("2. 如果流程正常，重新运行选择完整实验模式")
                print("3. 完整实验将获得论文级别的完整结果")
            else:
                print("1. 查看results目录中的所有图表和数据")
                print("2. 分析performance_summary.csv了解各模型性能")
                print("3. 查看model_comparison.png可视化结果")
                print("4. 根据结果撰写实验报告")


if __name__ == "__main__":
    runner = ExperimentRunner()
    success = runner.run_experiment()

    if not success:
        print("\n故障排除建议:")
        print("1. 检查错误信息，确保依赖已正确安装")
        print("2. 验证数据文件和模型文件是否完整")
        print("3. 单独运行失败的步骤进行调试")
        print("4. 检查系统资源是否充足")
        print("5. 如果是长时间任务，可以考虑分步骤执行")