# Autism_prediction_using_ML
# Project Overview
Detect ASD traits using an ML pipeline in Google Colab. This project features SMOTE for data balancing and compares XGBoost, Random Forest, and Decision Trees. Optimized via RandomizedSearchCV, it includes full EDA with Seaborn and model serialization using Pickle for deployment. High-accuracy screening for clinical support.
Here is a high-level, detailed README designed for a GitHub repository. It is structured to highlight your technical skills (like handling data imbalance and hyperparameter tuning) which are highly valued by recruiters.
Early detection of Autism Spectrum Disorder (ASD) is crucial for providing timely intervention and support. This project utilizes Machine Learning to predict the likelihood of ASD based on behavioral and demographic data, specifically utilizing screening datasets such as the **AQ-10 tool**.
The goal is to provide a fast, data-driven screening framework that can assist clinical professionals in identifying traits earlier and more objectively than traditional manual assessments.
##  Technical Tech Stack
  * **Platform:** Google Colaboratory (Cloud-based GPU/CPU)
  * **Data Handling:** `Numpy`, `Pandas`
  * **Visualization:** `Matplotlib`, `Seaborn`
  * **Preprocessing:** `Scikit-Learn` (LabelEncoder, Train-Test-Split)
  * **Sampling:** `Imbalanced-Learn` (**SMOTE**)
  * **Models:** `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`
  * **Optimization:** `RandomizedSearchCV`
  * **Persistence:** `Pickle`
##  Machine Learning Pipeline
### 1\. Exploratory Data Analysis (EDA)

Comprehensive analysis to identify patterns in demographic data (age, gender, ethnicity) and their correlation with ASD traits. We use Heatmaps and Countplots to visualize feature importance.

### 2\. Handling Data Imbalance (**SMOTE**)

Medical datasets are often imbalanced (fewer ASD+ cases than ASD-). I implemented **Synthetic Minority Over-sampling Technique (SMOTE)** to generate synthetic samples for the minority class, ensuring the model doesn't become biased toward "No ASD" predictions.

### 3\. Advanced Modeling & Tuning

I implemented and compared three robust algorithms:

  * **Decision Trees:** For baseline interpretability.
  * **Random Forest:** An ensemble bagging method to reduce variance.
  * **XGBoost:** An optimized gradient boosting framework for maximum predictive power.

**Hyperparameter Tuning:** Instead of manual testing, I used `RandomizedSearchCV` to find the optimal parameters for these models, significantly improving the F1-Score and Recall.

### 4\. Evaluation Metrics

The models are evaluated using:

  * **Confusion Matrix:** To track False Negatives (critical in healthcare).
  * **Classification Report:** Precision, Recall, and F1-Score.
  * **Cross-Validation:** Using `cross_val_score` to ensure the model generalizes to new data.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/himanshi0814/autism-prediction-ml.git
    ```
2.  **Open the Notebook:**
    Upload the `.ipynb` file to your [Google Colab](https://colab.research.google.com/).
3.  **Run the Cells:**
    The notebook will automatically handle the library imports and data processing. Ensure your dataset is uploaded to the Colab environment.

## Results & Conclusion

The final model is serialized into a `.pkl` file using **Pickle**, making it ready for deployment in a web or mobile application. By utilizing ensemble methods and SMOTE, we achieved a significant reduction in False Negatives compared to basic classification models.
##  Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/himanshi0814/autism-prediction-ml/issues).

