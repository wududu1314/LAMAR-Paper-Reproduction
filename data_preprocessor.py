import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from project_paths import paths


class MovieLensPreprocessor:
    def __init__(self):
        self.label_encoders = {}

    def load_raw_data(self):
        """加载原始MovieLens数据"""
        self.ratings = pd.read_csv(paths.get_data_file('u.data'), sep='\t',
                                   names=['user_id', 'movie_id', 'rating', 'timestamp'])
        self.users = pd.read_csv(paths.get_data_file('u.user'), sep='|',
                                 names=['user_id', 'age', 'gender', 'occupation', 'zipcode'])
        self.movies = pd.read_csv(paths.get_data_file('u.item'), sep='|', encoding='latin-1',
                                  names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] +
                                        [f'genre{i + 1}' for i in range(19)])

        print(f"数据加载完成：评分{len(self.ratings)}, 用户{len(self.users)}, 电影{len(self.movies)}")

    def extract_movie_features(self):
        """提取电影特征"""
        self.movies['Year'] = self.movies['movie_title'].str.extract(r'\((\d{4})\)')
        self.movies['Year'] = pd.to_numeric(self.movies['Year'], errors='coerce').fillna(1995)
        self.movies['Name'] = self.movies['movie_title']

    def generate_side_info(self):
        """基于现有电影信息生成侧信息"""
        genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                       'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                       'Thriller', 'War', 'Western', 'Unknown']

        directors = []
        writers = []
        stars = []

        for _, row in self.movies.iterrows():
            # 找到主要类型作为导演风格
            main_genre = 'Unknown'
            for i, genre in enumerate(genre_names):
                if i < 19 and row[f'genre{i + 1}'] == 1:
                    main_genre = genre
                    break

            directors.append(f"{main_genre}_Director")
            writers.append(f"Writer_{int(row['Year'])}")
            stars.append(f"Star_{row['movie_id']}")

        self.movies['director'] = directors
        self.movies['writers'] = writers
        self.movies['stars'] = stars

        print(f"侧信息生成完成：基于类型和年份构造导演、编剧、主演信息")

    def merge_all_features(self):
        """合并所有特征，包含改进的侧信息"""
        data = self.ratings.merge(self.users, on='user_id', how='left')

        # 生成侧信息
        self.generate_side_info()

        # 合并电影特征（包含真实的侧信息）
        movie_features = self.movies[['movie_id', 'Name', 'Year'] +
                                     [f'genre{i + 1}' for i in range(19)] +
                                     ['director', 'writers', 'stars']]
        data = data.merge(movie_features, on='movie_id', how='left')

        self.merged_data = data
        print(f"特征合并完成：{data.shape}")

    def temporal_split(self, quick_mode=False):
        """时间序列leave-one-out划分，支持快速模式"""
        train_data, valid_data, test_data = [], [], []

        for user_id in self.merged_data['user_id'].unique():
            user_data = self.merged_data[self.merged_data['user_id'] == user_id].sort_values('timestamp')

            if len(user_data) >= 3:
                train_data.extend(user_data[:-2].to_dict('records'))
                valid_data.append(user_data.iloc[-2].to_dict())
                test_data.append(user_data.iloc[-1].to_dict())
            elif len(user_data) == 2:
                train_data.append(user_data.iloc[0].to_dict())
                test_data.append(user_data.iloc[1].to_dict())
            else:
                train_data.append(user_data.iloc[0].to_dict())

        self.train_df = pd.DataFrame(train_data)
        self.valid_df = pd.DataFrame(valid_data)
        self.test_df = pd.DataFrame(test_data)

        # 快速模式：缩减测试集
        if quick_mode:
            if len(self.test_df) > 50:
                self.test_df = self.test_df.sample(n=50, random_state=42).reset_index(drop=True)
            print(f"快速模式：测试集缩减为 {len(self.test_df)} 样本")

        print(f"数据划分：训练{len(self.train_df)}, 验证{len(self.valid_df)}, 测试{len(self.test_df)}")

    def encode_features(self):
        """特征编码"""
        sparse_features = [
                              "movie_id", "user_id", "gender", "age", "occupation", "zipcode",
                              'Name', 'Year'
                          ] + [f'genre{i + 1}' for i in range(19)] + ['director', 'writers', 'stars']

        all_data = pd.concat([self.train_df, self.valid_df, self.test_df], axis=0, ignore_index=True)

        for feat in sparse_features:
            if feat in all_data.columns:
                lbe = LabelEncoder()
                all_data[feat] = lbe.fit_transform(all_data[feat].astype(str))
                self.label_encoders[feat] = lbe

        train_len, valid_len = len(self.train_df), len(self.valid_df)
        self.train_encoded = all_data[:train_len].copy()
        self.valid_encoded = all_data[train_len:train_len + valid_len].copy()
        self.test_encoded = all_data[train_len + valid_len:].copy()

    def save_processed_data(self):
        """保存处理后的数据"""
        self.train_encoded.to_csv(paths.get_train_data(), index=False)
        self.valid_encoded.to_csv(paths.get_valid_data(), index=False)
        self.test_encoded.to_csv(paths.get_test_data(), index=False)
        print("数据预处理完成")


if __name__ == "__main__":
    # 检查是否为快速模式
    quick_mode = len(sys.argv) > 1 and sys.argv[1] == "quick"

    processor = MovieLensPreprocessor()
    processor.load_raw_data()
    processor.extract_movie_features()
    processor.merge_all_features()
    processor.temporal_split(quick_mode=quick_mode)
    processor.encode_features()
    processor.save_processed_data()