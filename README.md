# CodeAlpha_IrisClassification

**CodeAlpha Data Science Internship - Task 1**

Classifies Iris flowers into three species (*setosa*, *versicolor*, *virginica*) from four measurements: sepal length, sepal width, petal length and petal width.

## What this project does
1. Loads the Iris dataset (built-in from scikit-learn, or the CSV from CodeAlpha)
2. Explores the data (statistics, missing values, class balance, pairplot, correlation heatmap)
3. Splits the data 80/20 with stratification
4. Trains and compares 5 models: Logistic Regression, KNN, Decision Tree, Random Forest, SVM
5. Evaluates the best model with accuracy, precision/recall/F1 and a confusion matrix
6. Shows feature importance and predicts a new flower

## Results
| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| SVM (RBF) | 0.967 | 0.967 |
| Logistic Regression | 0.958 | 0.933 |
| K-Nearest Neighbors | 0.967 | 0.933 |
| Decision Tree | 0.942 | 0.933 |
| Random Forest | 0.950 | 0.900 |

Best model: **SVM (RBF)** at about 97% test accuracy. Setosa is classified perfectly; the only confusion is between versicolor and virginica, which overlap slightly. Petal width and petal length are the most informative features.

## How to run
```bash
pip install -r requirements.txt
python iris_classification.py                  # built-in dataset
python iris_classification.py --csv Iris.csv   # CodeAlpha CSV
```
Plots and results are saved to the `outputs/` folder.

## Tech stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

*Built for the CodeAlpha internship program.*