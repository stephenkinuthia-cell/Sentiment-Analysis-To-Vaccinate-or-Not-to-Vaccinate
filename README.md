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

- Removed rows with missing sentiment labels
- Filled missing agreement values with the median value
- Converted labels to integer format

### 2. **Text Cleaning**

- Converted text to lowercase
- Removed special characters and non-alphabetic characters
- Removed English stopwords using NLTK
- Performed tokenization and normalization

### 3. **Feature Extraction**

- Used **TF-IDF Vectorizer** to convert text to numerical features
- Maximum of 5,000 features extracted
- Applied on both training and test datasets

## Models Evaluated

The following machine learning models were trained and evaluated:

1. **Logistic Regression**

   - Fast and interpretable baseline model
   - Effective for text classification tasks
2. **Random Forest Classifier**

   - Ensemble method with 100 estimators
   - Captures non-linear relationships
   - **Selected as the final model for predictions**
3. **K-Nearest Neighbors (KNN)**

   - Instance-based learning with k=5
   - Provides local decision boundaries

## Evaluation Metric

The challenge uses **F1-score** as the primary evaluation metric.

**Why F1-score?**

- F1-score balances precision and recall, making it ideal for datasets with imbalanced classes
- Provides a harmonic mean of precision and recall
- More informative than accuracy alone when dealing with imbalanced datasets
- Better representation of model performance in real-world scenarios

## Approach Summary

### Methodology:

1. **Data Exploration**: Analyzed the distribution of sentiment labels and text features
2. **Text Preprocessing**: Applied comprehensive text cleaning to prepare data for modeling
3. **Feature Engineering**: Used TF-IDF vectorization to convert text into numerical features suitable for machine learning
4. **Model Experimentation**: Tested multiple algorithms (Logistic Regression, Random Forest, KNN) to identify the best performer
5. **Model Selection**: Selected Random Forest Classifier based on performance metrics
6. **Prediction & Submission**: Generated predictions on test set and created submission file

### Key Implementation Details:

- **Vectorization Method**: TF-IDF with max 5,000 features
- **Stopword Removal**: Used NLTK English stopwords
- **Train-Test Split**: Maintained original data split provided by challenge
- **Hyperparameters**: Random Forest with 100 trees and random_state=42 for reproducibility

## Model Performance

### Best Performing Model: Random Forest Classifier

The Random Forest Classifier was selected as the final model because:

- Ensemble approach reduces overfitting
- Captures complex patterns in text features
- Provides robust predictions across different feature combinations
- Showed the best balance between precision and recall

### Model Details:

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators=100
  - random_state=42

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

## Conclusion

This project demonstrates a complete machine learning pipeline for text classification, from raw tweet data to final predictions. The Random Forest Classifier proved to be an effective choice for this sentiment analysis task, balancing interpretability with predictive performance. The F1-score metric's focus on precision-recall trade-off aligns well with the goal of accurately detecting sentiment in potentially imbalanced tweet distributions.

Future work could explore more advanced NLP techniques such as word embeddings, deep learning models, or ensemble methods combining multiple specialized algorithms to further improve classification performance.
