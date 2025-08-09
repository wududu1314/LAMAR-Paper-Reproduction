from pathlib import Path


class ProjectPaths:
    def __init__(self, project_root='.'):
        self.root = Path(project_root)
        self.data_dir = self.root / 'data'
        self.models_dir = self.root / 'models'
        self.results_dir = self.root / 'results'

        # 创建必要目录
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)

    def get_data_file(self, filename):
        return str(self.data_dir / 'ml-100k' / filename)

    def get_model_path(self, model_name):
        return str(self.models_dir / model_name)

    def get_result_file(self, filename):
        return str(self.results_dir / filename)

    def get_train_data(self):
        return str(self.data_dir / 'train_set.csv')

    def get_valid_data(self):
        return str(self.data_dir / 'valid_set.csv')

    def get_test_data(self):
        return str(self.data_dir / 'test_set.csv')


# 全局路径管理器
paths = ProjectPaths()