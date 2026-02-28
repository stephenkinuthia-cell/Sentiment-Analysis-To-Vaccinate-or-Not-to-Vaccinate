# Sentiment Analysis Challenge: To Vaccinate or Not to Vaccinate

## Project Overview

This project is a text classification challenge focused on sentiment analysis of tweets regarding vaccination. The goal is to classify tweets into sentiment categories to understand public opinion on vaccination topics.

## Dataset Description

- **Train Data**: Contains labeled tweets with sentiment labels and agreement scores
- **Test Data**: Contains unlabeled tweets for which we need to predict sentiment
- **Features**:
  - `tweet_id`: Unique identifier for each tweet
  - `safe_text`: The tweet text content
  - `label`: Sentiment label (target variable) - Binary classification
  - `agreement`: Agreement score among annotators

## Data Preprocessing Pipeline

### 1. **Handling Missing Values**

- Dropped rows with missing sentiment labels
- Converted `label` column to integer type
- Filled missing `agreement` values with the median of the training set

### 2. **Data Splitting & Balancing**

- Performed an 80/20 stratified train/validation split to preserve class ratios
- Addressed class imbalance using **SMOTE** when training KNN, and used `class_weight='balanced'` for tree‑based models

### 3. **Text Cleaning**

- Converted text to lowercase
- Removed special characters and non-alphabetic characters
- Removed English stopwords using NLTK
- Applied tokenization and simple normalization

### 4. **Feature Extraction**

- Used **TF-IDF Vectorizer** to convert text to numerical features
- Limited to 5,000 features and included unigrams and bigrams (`ngram_range=(1,2)`)
- Transformed both training/validation and test datasets

### 5. **Regression Extension (Zindi Requirement)**

- Converted sentiment labels to continuous targets in [-1, 1]
- Trained regression versions of several models (Linear Regression, Random Forest Regressor, XGBoost) and evaluated using RMSE
- Selected the best regressor for final test predictions when required by the competition format

## Models Evaluated

The following machine learning models were trained and evaluated in the classification setting:

1. **Logistic Regression**
   - Fast and interpretable baseline model
   - Applied with `class_weight='balanced'` to account for imbalance
2. **Random Forest Classifier**
   - Ensemble method with 200 estimators (used in notebook)
   - Captures non-linear relationships
   - **Selected as the final classifier based on weighted F1-score**
3. **K-Nearest Neighbors (KNN)**
   - Instance-based learning with k=5
   - Oversampled training data with SMOTE to mitigate imbalance

In addition, when the competition required a regression-style submission (continuous target), the following regressors were compared:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

The best regressor was chosen in this case XGBoost Regressor since it had the lowest RMSE on the validation set, therefore it was used to generate the final `submission.csv` .

## Evaluation Metric

For the classification task the primary evaluation metric is the **weighted F1-score**. It balances precision and recall, making it ideal for imbalanced class distributions. F1-score provides a harmonic mean of precision and recall, offering a more informative assessment than accuracy alone.

When the problem was reformulated as a regression (continuous target between -1 and 1), we used **Root Mean Squared Error (RMSE)** on the validation set to compare regressors and select the best model for submission.

**Why F1-score was used?**

- Balances precision and recall, useful for imbalanced classes
- Provides a harmonic mean of precision and recall
- More informative than accuracy when class ratios are skewed

**Why we used RMSE?**

- Penalizes large prediction errors more heavily
- Standard metric for continuous target comparison
- Aligns with Zindi’s regression-style leaderboard when required

## Approach Summary

### Methodology:

1. **Data Exploration**: Inspected the dataset, checked for missing values, and visualized class distribution with Plotly
2. **Data Cleaning & Preprocessing**: Removed null labels, standardized text (lowercasing, special character removal, stopword filtering), and added a `clean_text` column
3. **Train/Validation Split**: Performed an 80/20 stratified split to hold out data for unbiased model evaluation
4. **Feature Engineering**: Converted cleaned text into TF-IDF features with unigrams and bigrams (max 5,000 features)
5. **Class Balancing**: Applied SMOTE oversampling when training KNN and set `class_weight='balanced'` for other classifiers
6. **Model Experimentation**: Trained and validated Logistic Regression, Random Forest, and KNN classifiers; recorded F1-scores
7. **Regression Extension**: When the challenge required continuous targets, also trained Linear Regression, Random Forest Regressor, and XGBoost, selecting the best by lowest RMSE
8. **Model Selection & Final Training**: Chose Random Forest (classifier or regressor depending on submission type), retrained on full data, and generated predictions
9. **Prediction & Submission**: Produced `submission.csv` with either discrete labels or clipped continuous targets

### Key Implementation Details:

- **Vectorization Method**: TF-IDF with max 5,000 features, ngram_range=(1,2)
- **Stopword Removal**: NLTK English stopwords
- **Sampling Strategy**: Stratified train/validation split and SMOTE for imbalance
- **Hyperparameters**: Random Forest with 200 estimators and `random_state=42` for reproducibility
- **Regression Models**: Added support for Linear, RandomForest, and XGBoost regressors when required by the contest format

## Model Performance

### Best Performing Model

For the **classification task**, the Random Forest Classifier achieved the highest weighted F1-score on the validation set and was chosen for the final submission. The ensemble approach reduced overfitting, captured complex text patterns, and provided the best precision‑recall balance.

For the **regression variant**, a Random Forest Regressor (and in some runs XGBoost) produced the lowest RMSE; whichever regressor achieved the best validation RMSE was used to generate predictions and produce the required continuous output.

### Model Details (classification example):

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators=200
  - random_state=42
  - class_weight='balanced'

### Model Details; regression,

- **Algorithms**: Random Forest Regressor / XGBoost Regressor
- **Parameters** (RF): n_estimators=100, max_depth=10, random_state=42
- **Parameters** (XGB): n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42

## Lessons Learned During the Process

### 1. **Text Preprocessing is Critical**

- Proper text cleaning significantly impacts model performance
- Stopword removal helps reduce noise and focuses on meaningful words
- Lowercase conversion and special character removal standardize the data

### 2. **Feature Extraction Technique Matters**

- TF-IDF vectorization effectively captures the importance of terms
- Limiting to 5,000 features balances information richness with computational efficiency
- Term frequency weighting helps distinguish between common and rare meaningful terms

### 3. **Ensemble Methods Excel in Text Classification**

- Random Forest outperformed individual algorithms like Logistic Regression and KNN
- Ensemble methods are particularly effective for high-dimensional text data
- The combination of multiple decision trees reduces variance and improves generalization

### 4. **Class Balance Considerations**

- F1-score selection indicates potential class imbalance in the dataset
- Models need to balance precision and recall rather than optimizing for accuracy alone
- Agreement scores in the data suggest annotation challenges that reflect real-world uncertainty

### 5. **Importance of Model Reproducibility**

- Setting random_state=42 ensures consistency across runs
- Important for debugging and team collaboration
- Allows for fair comparison between different models and parameters

### 6. **Hyperparameter Optimization Opportunities**

- The models trained used default or modest hyperparameters
- Future improvements could include:
  - Grid search or random search for optimal hyperparameters
  - Cross-validation for more robust performance estimation
  - Different vectorization parameters (different max_features values)
  - Advanced models like Gradient Boosting or Neural Networks

## File Structure

```
├── data/
│   ├── Train.csv           # Training data with labels
│   ├── Test.csv            # Test data for predictions
│   ├── SampleSubmission.csv # Sample submission format
│   └── NLP_Primer_twitter_challenge.ipynb
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis and Modeling
│   ├── data.py             # Data utilities
│   └── submission.csv      # Final predictions
└── README.md               # This file
```

## Usage Instructions

### Running the Analysis:

1. Ensure all required libraries are installed
2. Run the cells in `notebooks/eda.ipynb` sequentially
3. The script will generate a `submission.csv` file with predictions

### Required Libraries:

- pandas
- numpy
- nltk
- scikit-learn
- regex (re)

## Short Summary

- **Approach:** Cleaned and preprocessed tweet text, split the data into stratified train/validation sets, vectorized with TF-IDF (unigrams and bigrams), balanced classes using SMOTE and class weights, and experimented with multiple classifiers and regressors. Models were compared using weighted F1-score for classification and RMSE for regression before retraining the best model on all available data for final predictions.
- **Best Model:** Random Forest Classifier delivered the top F1-score in the classification setting; for regression submissions the Random Forest Regressor (or XGBoost when slightly better) was chosen based on validation RMSE.
- **Lessons Learned:** Effective text preprocessing is critical, feature extraction choices influence results significantly, ensemble methods outperform simpler baselines on high‑dimensional text data, handling class imbalance is necessary, and adapting the pipeline for regression tasks made the solution flexible.

## Conclusion

This project demonstrates a complete machine learning pipeline for text classification, from raw tweet data to final predictions. The Random Forest Classifier proved to be an effective choice for this sentiment analysis task, balancing interpretability with predictive performance. The F1-score metric's focus on precision-recall trade-off aligns well with the goal of accurately detecting sentiment in potentially imbalanced tweet distributions.

Future work could explore more advanced NLP techniques such as word embeddings, deep learning models, or ensemble methods combining multiple specialized algorithms to further improve classification performance.
