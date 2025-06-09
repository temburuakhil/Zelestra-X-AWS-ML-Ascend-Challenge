# Solar Panel Performance Optimization

This project implements a machine learning solution for predicting solar panel efficiency degradation and enabling predictive maintenance using sensor data and environmental factors.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── approach.txt
├── src/
│   ├── data_exploration.py
│   ├── data_exploration.ipynb
│   ├── feature_engineering.py
│   ├── feature_engineering.ipynb
│   ├── model_training.py
│   └── model_training.ipynb
├── plots/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Jupyter (if not already installed):
```bash
pip install jupyter
```

## Project Components

### 1. Data Exploration
- **Python Script** (`data_exploration.py`): Automated EDA pipeline
- **Jupyter Notebook** (`data_exploration.ipynb`): Interactive EDA
- Features:
  - Comprehensive EDA of the dataset
  - Analysis of target variable distribution
  - Missing value analysis
  - Correlation analysis
  - Categorical variable analysis
  - Generates visualization plots in the `plots/` directory

### 2. Feature Engineering
- **Python Script** (`feature_engineering.py`): Automated feature engineering pipeline
- **Jupyter Notebook** (`feature_engineering.ipynb`): Interactive feature engineering
- Features:
  - Advanced feature creation based on domain knowledge
  - Power-related features
  - Temperature efficiency relationships
  - Environmental impact features
  - Degradation indicators
  - Categorical feature encoding
  - Missing value handling
  - Outlier detection and treatment

### 3. Model Training
- **Python Script** (`model_training.py`): Automated model training pipeline
- **Jupyter Notebook** (`model_training.ipynb`): Interactive model training
- Features:
  - Multiple model implementations:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Ridge Regression
  - Hyperparameter optimization using Optuna
  - Model evaluation and comparison
  - Ensemble creation using stacking
  - Feature importance analysis
  - SHAP analysis for model interpretation

## Usage

### Using Python Scripts
1. Run the data exploration:
```bash
python src/data_exploration.py
```

2. Train the models and generate predictions:
```bash
python src/model_training.py
```

### Using Jupyter Notebooks
1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `src` directory and open any of the notebooks:
   - `data_exploration.ipynb`
   - `feature_engineering.ipynb`
   - `model_training.ipynb`

## Model Performance

The project uses a custom scoring metric:
```
Score = 100 * (1 - √(MSE))
```

The ensemble model combines the best performing individual models to achieve optimal performance.

## Output Files

- `plots/`: Directory containing all generated visualizations
- `submission.csv`: Final predictions for the test set

## Key Features

1. **Advanced Feature Engineering**
   - Physics-based features
   - Interaction features
   - Ratio features
   - Polynomial features

2. **Robust Data Preprocessing**
   - Missing value handling
   - Outlier detection and treatment
   - Categorical feature encoding

3. **Multiple Model Approach**
   - Various algorithms tested
   - Hyperparameter optimization
   - Ensemble methods

4. **Model Interpretation**
   - Feature importance analysis
   - SHAP analysis
   - Performance metrics

5. **Interactive Development**
   - Jupyter notebooks for interactive development
   - Python scripts for automated pipelines
   - Comprehensive documentation

## Contributing

Feel free to submit issues and enhancement requests! 