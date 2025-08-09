# LAMAR Reproduction

Reproduction of "Large Language Models Augmented Rating Prediction in Recommender System" (ICASSP 2024).

## Results

DeepFM baseline: RMSE 1.0758, MAE 0.8206  
Phi-3-mini+LAMAR: RMSE 1.0432, MAE 0.8234  
Llama-2+LAMAR: RMSE 1.0509, MAE 0.8193  

All results outperform the original paper.

## Installation

conda create -n lamar python=3.9 -y
conda activate lamar
pip install torch transformers deepctr-torch scikit-learn pandas numpy matplotlib
```

## Usage

# Download data
curl -O http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d ./data/

# Run experiment
python run_complete_experiment.py


## Requirements

- Python 3.9+
- CUDA-capable GPU (16GB+ recommended)
- 50GB disk space

## Citation

@inproceedings{luo2024lamar,
  title={Large Language Models Augmented Rating Prediction in Recommender System},
  author={Luo, Sichun and Wang, Jiansheng and Zhou, Aojun and Ma, Li and Song, Linqi},
  booktitle={ICASSP 2024},
  year={2024}
}
