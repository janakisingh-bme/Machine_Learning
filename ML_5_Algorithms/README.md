\# Classical Machine Learning — 5 Algorithm Comparison



A hands-on study and implementation of five fundamental classical machine learning classification algorithms, combining mathematical foundations with practical experimentation.



\## 📌 Algorithms Covered



1\. Logistic Regression

2\. K-Nearest Neighbors (KNN)

3\. Naive Bayes

4\. Decision Tree

5\. Support Vector Machine (SVM)



The project focuses on understanding \*\*how these algorithms work mathematically\*\*, implementing them using Python and scikit-learn, and comparing their classification performance using the same experimental setup.



\---



\## 🎯 Objectives



\* Understand the mathematical foundations of classical ML algorithms

\* Implement five major classification algorithms

\* Understand the role of feature scaling

\* Compare model performance using standard classification metrics

\* Analyze confusion matrices

\* Understand the strengths and limitations of different algorithms

\* Build a foundation for more advanced machine learning and medical AI applications



\---



\## 📂 Project Structure



```text

ML\_5\_Algorithms/

│

├── study\_notes.md

│

├── classical\_ml\_comparison.py

│

├── outputs/

│   ├── logistic\_regression\_confusion\_matrix.png

│   ├── knn\_confusion\_matrix.png

│   ├── naive\_bayes\_confusion\_matrix.png

│   ├── decision\_tree\_confusion\_matrix.png

│   ├── svm\_confusion\_matrix.png

│   ├── model\_comparison.csv

│   └── model\_comparison.png

│

└── README.md

```



\---



\## 🧠 Mathematical Foundations



\### 1. Logistic Regression



Logistic Regression models the probability of a binary outcome using the sigmoid function.



```text

z = wᵀx + b



σ(z) = 1 / (1 + e⁻ᶻ)

```



The predicted probability is:



```text

P(y=1|x) = σ(wᵀx+b)

```



Binary cross-entropy is used as the loss function.



\---



\### 2. K-Nearest Neighbors



KNN classifies a sample based on the labels of its nearest neighbors.



Euclidean distance:



```text

d(x,y) = √Σ(xᵢ-yᵢ)²

```



The predicted class is determined through majority voting among the selected neighbors.



KNN is sensitive to feature scale, so standardization is important.



\---



\### 3. Naive Bayes



Naive Bayes is based on Bayes' theorem:



```text

P(C|X) = P(X|C)P(C) / P(X)

```



For multiple features:



```text

P(C|X) ∝ P(C) ∏ P(xᵢ|C)

```



The algorithm assumes conditional independence between features given the class.



\---



\### 4. Decision Tree



Decision Trees recursively split the dataset to improve class separation.



Entropy:



```text

H(S) = -Σ pᵢ log₂(pᵢ)

```



Information Gain:



```text

IG(S,A) = H(S) - Σ (|Sᵥ|/|S|) H(Sᵥ)

```



Gini impurity is another commonly used splitting criterion.



\---



\### 5. Support Vector Machine



SVM searches for a separating hyperplane:



```text

wᵀx + b = 0

```



The margin between the supporting hyperplanes is:



```text

Margin = 2 / ||w||

```



SVM attempts to maximize this margin while controlling classification errors.



For nonlinear problems, kernel functions such as the RBF kernel can be used.



\---



\## 🧪 Dataset



The project uses the \*\*Breast Cancer Wisconsin Diagnostic dataset\*\* provided through scikit-learn.



Dataset characteristics:



\* 569 samples

\* 30 numerical features

\* Binary classification

\* 2 target classes

\* 80/20 stratified train-test split

\* Random state: `42`



The dataset is used for educational experimentation and should not be interpreted as a clinical diagnostic system.



\---



\## ⚙️ Experimental Setup



All five algorithms were trained and evaluated using the same train-test split.



\### Preprocessing



Feature standardization was applied where appropriate:



```text

z = (x - μ) / σ

```



Scaling was used for:



\* Logistic Regression

\* KNN

\* SVM



It was not required for Decision Trees and was not essential for Gaussian Naive Bayes.



\### Models



| Algorithm           | Main Configuration |

| ------------------- | ------------------ |

| Logistic Regression | `max\_iter=5000`    |

| KNN                 | `n\_neighbors=5`    |

| Naive Bayes         | GaussianNB         |

| Decision Tree       | `max\_depth=5`      |

| SVM                 | RBF kernel         |



\---



\## 📊 Evaluation Metrics



The models were evaluated using:



\### Accuracy



```text

Accuracy = (TP + TN) / (TP + TN + FP + FN)

```



\### Precision



```text

Precision = TP / (TP + FP)

```



\### Recall / Sensitivity



```text

Recall = TP / (TP + FN)

```



\### F1 Score



```text

F1 = 2 × (Precision × Recall) / (Precision + Recall)

```



Confusion matrices were also generated for each model.



\---



\## 📈 Results



| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |

| ----------------------- | ---------: | ---------: | ---------: | ---------: |

| \*\*Logistic Regression\*\* | \*\*98.25%\*\* | \*\*98.61%\*\* | \*\*98.61%\*\* | \*\*98.61%\*\* |

| \*\*SVM\*\*                 | \*\*98.25%\*\* | \*\*98.61%\*\* | \*\*98.61%\*\* | \*\*98.61%\*\* |

| KNN                     |     95.61% |     95.89% |     97.22% |     96.55% |

| Naive Bayes             |     93.86% |     94.52% |     95.83% |     95.17% |

| Decision Tree           |     92.11% |     95.65% |     91.67% |     93.62% |



\---



\## 🏆 Key Findings



\### Logistic Regression and SVM



Both models achieved the highest performance on this particular train-test split:



```text

Accuracy : 98.25%

Precision: 98.61%

Recall   : 98.61%

F1 Score : 98.61%

```



This demonstrates strong classification performance on the selected dataset.



\### KNN



KNN also performed strongly, achieving:



```text

F1 Score: 96.55%

```



Its recall of 97.22% indicates that it correctly identified most positive samples.



\### Naive Bayes



Naive Bayes achieved:



```text

F1 Score: 95.17%

```



Its performance was lower than Logistic Regression, SVM, and KNN in this experiment.



\### Decision Tree



The Decision Tree produced the lowest F1 score among the five models:



```text

F1 Score: 93.62%

Recall  : 91.67%

```



This highlights how tree performance can depend strongly on tree depth and other hyperparameters.



\---



\## 🔎 Important Interpretation



The results should \*\*not\*\* be interpreted as meaning that Logistic Regression or SVM is always better than the other algorithms.



The correct conclusion is:



> Logistic Regression and SVM achieved the best performance on this dataset using the selected preprocessing, hyperparameters, and train-test split.



Model performance can change with:



\* Dataset

\* Train-test split

\* Feature engineering

\* Hyperparameter tuning

\* Class imbalance

\* Cross-validation strategy



\---



\## 📁 Generated Outputs



The Python implementation automatically generates:



\### Confusion Matrices



One confusion matrix is generated for each algorithm.



\### Model Comparison



`model\_comparison.csv` contains the evaluation metrics for all five algorithms.



\### Performance Plot



`model\_comparison.png` provides a visual comparison of the models.



\---



\## 🚀 Future Improvements



The next stage of this project can include:



\* K-Fold Cross-Validation

\* Hyperparameter tuning using GridSearchCV

\* ROC-AUC comparison

\* Precision-Recall curves

\* Specificity calculation

\* Learning curves

\* Feature importance analysis

\* Decision Tree visualization

\* SVM kernel comparison

\* KNN optimization for different values of K

\* Logistic Regression regularization comparison

\* Statistical comparison across cross-validation folds



\---



\## 🧰 Technologies Used



\* Python

\* NumPy

\* Pandas

\* Matplotlib

\* Scikit-learn



\---



\## 📚 Learning Outcome



This project provided practical understanding of five important classical machine learning approaches:



| Algorithm           | Core Concept          |

| ------------------- | --------------------- |

| Logistic Regression | Probability + sigmoid |

| KNN                 | Distance + voting     |

| Naive Bayes         | Bayes theorem         |

| Decision Tree       | Impurity reduction    |

| SVM                 | Maximum margin        |



The project connects mathematical concepts with practical machine learning implementation and evaluation.



\---



\## 👩‍💻 Author



\*\*Janaki Singh\*\*



Electronics, Communication \& Automation | Machine Learning | Medical AI | Embedded Systems



\---



\## ⭐ Project Purpose



This project is part of my ongoing journey of learning machine learning through \*\*mathematics, implementation, experimentation, and practical evaluation\*\*.



