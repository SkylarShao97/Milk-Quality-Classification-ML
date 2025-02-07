# Milk-Quality-Classification-ML

#### Project Overview
This project applies **supervised machine learning** to classify milk quality based on seven chemical and sensory features. We implement and compare **Logistic Regression, Decision Trees, and Random Forests**, evaluating their performance through **cross-validation, hyperparameter tuning, and normalization**. Additionally, we explore **TabR (2023)**, a hybrid deep learning model that integrates k-Nearest Neighbors (KNN) for tabular data classification.

#### Dataset
- **Source**: Kaggle  
- **Size**: 1,059 entries
- **Missing data**: 0
- **Duplicated data**: 976
- **Features**:
  - pH  
  - Temperature  
  - Taste  
  - Odor  
  - Fat  
  - Turbidity  
  - Color  
- **Target Variable**: Milk Quality (Low, Medium, High)  

#### Methodology
1. **Data Preprocessing**
   - Data cleaning and handling duplicates  
   - Exploratory Data Analysis (EDA)  
   - Train-test split (80-20)  
   - Normalization  

2. **Model Selection & Training**
   - **Logistic Regression** (Regularization tuning)  
   - **Decision Tree** (Depth optimization)  
   - **Random Forest** (Number of trees tuning)  
   - **TabR** (Hybrid KNN-Deep Learning model)  

3. **Hyperparameter Tuning**
   - Manual tuning with **5-fold cross-validation**  
   - Evaluated **C (Logistic Regression)**, **max_depth (Decision Tree)**, and **n_estimators (Random Forest)**  

4. **Evaluation Metrics**
   - Accuracy  
   - Feature importance analysis  
   - Impact of normalization  

#### Key Findings
- **Random Forest** achieved the highest accuracy (**99.88%**), outperforming other models due to ensemble learning.  
- **Normalization** significantly improved **Logistic Regression accuracy by 17.4%**.  
- **Tree-based models (Decision Tree, Random Forest)** performed best, prioritizing **pH** as the most important predictor.  
- **TabR** provided insights into retrieval-augmented deep learning for tabular data.  

#### Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)  
- **Machine Learning Models** (Logistic Regression, Decision Tree, Random Forest)  
- **Deep Learning Approach** (TabR, KNN-based feature augmentation)  

#### References
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.  
2. Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., & Babenko, A. (2023). *TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023*. arXiv preprint arXiv:2307.14338.  

#### Repository Link
[GitHub Repository](https://github.com/skywalker007-cpu/DATA572_Project.git)
