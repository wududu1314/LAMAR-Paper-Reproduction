import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import random
from project_paths import paths


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepFMTrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.feature_names = None

    def prepare_features(self, data):
        sparse_features = [
                              "movie_id", "user_id", "gender", "age", "occupation", "zipcode",
                              'Name', 'Year'
                          ] + [f'genre{i + 1}' for i in range(19)] + ['director', 'writers', 'stars']

        self.feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
        self.feature_names = get_feature_names(self.feature_columns)
        return sparse_features

    def train_model(self, train_data, valid_data, device='cuda'):
        set_seed()

        train_input = {name: train_data[name].values for name in self.feature_names}
        valid_input = {name: valid_data[name].values for name in self.feature_names}

        self.model = DeepFM(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            task='regression', device=device
        )

        self.model.compile("adam", "mse", metrics=['mse'])

        print("开始训练DeepFM...")
        history = self.model.fit(
            train_input, train_data['rating'].values,
            batch_size=256, epochs=50, verbose=2,
            validation_data=(valid_input, valid_data['rating'].values)
        )

        return history

    def predict(self, test_data):
        test_input = {name: test_data[name].values for name in self.feature_names}
        predictions = self.model.predict(test_input, batch_size=256)
        return predictions.flatten()

    def evaluate(self, true_ratings, pred_ratings):
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae = mean_absolute_error(true_ratings, pred_ratings)

        print(f"DeepFM性能: RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {"rmse": rmse, "mae": mae}


if __name__ == "__main__":
    # 加载数据
    train_data = pd.read_csv(paths.get_train_data())
    valid_data = pd.read_csv(paths.get_valid_data())
    test_data = pd.read_csv(paths.get_test_data())

    all_data = pd.concat([train_data, valid_data, test_data], axis=0)

    trainer = DeepFMTrainer()
    trainer.prepare_features(all_data)

    train_len, valid_len = len(train_data), len(valid_data)
    train_encoded = all_data[:train_len]
    valid_encoded = all_data[train_len:train_len + valid_len]
    test_encoded = all_data[train_len + valid_len:]

    trainer.train_model(train_encoded, valid_encoded)
    predictions = trainer.predict(test_encoded)
    trainer.evaluate(test_encoded['rating'].values, predictions)

    # 保存结果
    np.save(paths.get_result_file('deepfm_predictions.npy'), predictions)
    np.save(paths.get_result_file('true_ratings.npy'), test_encoded['rating'].values)
    print("DeepFM训练完成")