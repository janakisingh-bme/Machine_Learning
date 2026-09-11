\# Machine Learning — Mathematical Study Notes



\## Five Fundamental Classification Algorithms



\*\*Algorithms covered\*\*



1\. Logistic Regression

2\. K-Nearest Neighbors (KNN)

3\. Naive Bayes

4\. Decision Tree

5\. Support Vector Machine (SVM)



\---



\# 1. Logistic Regression



\## 1.1 What is Logistic Regression?



Logistic Regression is a supervised learning algorithm used primarily for classification.



Although its name contains "regression", it is commonly used to predict the probability of a categorical outcome.



For binary classification:



`y ∈ {0, 1}`



The model estimates:



`P(y = 1 | x)`



\---



\## 1.2 Linear Combination



The first step is calculating a linear score:



`z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ`



In vector notation:



`z = wᵀx + b`



where:



\* `x` = feature vector

\* `w` = weight vector

\* `b` = bias/intercept

\* `z` = linear score



The problem is that `z` can have any value from `-∞` to `+∞`.



We need a value between 0 and 1 to represent probability.



\---



\## 1.3 Sigmoid Function



Logistic Regression uses the sigmoid function:



`σ(z) = 1 / (1 + e⁻ᶻ)`



Therefore:



`P(y = 1 | x) = σ(wᵀx + b)`



The sigmoid function maps:



`(-∞, +∞) → (0, 1)`



\### Important properties



If:



`z → +∞`



then:



`σ(z) → 1`



If:



`z → -∞`



then:



`σ(z) → 0`



If:



`z = 0`



then:



`σ(0) = 0.5`



\---



\## 1.4 Decision Rule



A common classification threshold is 0.5:



`ŷ = 1  if  P(y=1|x) ≥ 0.5`



`ŷ = 0  if  P(y=1|x) < 0.5`



Since:



`σ(z) ≥ 0.5`



when:



`z ≥ 0`



the decision boundary is:



`wᵀx + b = 0`



\---



\## 1.5 Log-Odds



Logistic Regression can also be expressed using odds.



Odds:



`odds = p / (1-p)`



Taking the natural logarithm:



`log(p / (1-p)) = wᵀx + b`



This is called the \*\*logit\*\* or \*\*log-odds\*\*.



Therefore:



`logit(p) = log(p/(1-p))`



and:



`logit(p) = wᵀx + b`



\---



\## 1.6 Loss Function



Logistic Regression commonly uses \*\*Binary Cross-Entropy (Log Loss)\*\*.



For one observation:



`L(y,p) = -\[y log(p) + (1-y)log(1-p)]`



For `m` observations:



`J(w,b) = -(1/m) Σ \[yi log(pi) + (1-yi)log(1-pi)]`



The model learns weights that minimize this loss.



\---



\## 1.7 Regularization



Regularization helps prevent overfitting.



\### L2 Regularization



A common objective is:



`J = Loss + λ Σ wj²`



where:



\* `λ` = regularization strength

\* larger `λ` → stronger penalty



In scikit-learn Logistic Regression, this behavior is controlled through the `C` parameter, where approximately:



`C ∝ 1/λ`



Therefore:



\* Small `C` → stronger regularization

\* Large `C` → weaker regularization



\---



\## Key Concepts



\* Linear combination

\* Sigmoid function

\* Probability

\* Decision boundary

\* Log-odds

\* Cross-entropy loss

\* Regularization



\---



\# 2. K-Nearest Neighbors (KNN)



\## 2.1 What is KNN?



K-Nearest Neighbors is a \*\*distance-based\*\*, instance-based supervised learning algorithm.



Unlike many models, KNN does not learn a traditional parametric equation during training.



Instead, it stores the training data and makes a decision when a new observation arrives.



\---



\## 2.2 Distance Calculation



For two points:



`x = (x₁,x₂,...,xₙ)`



and:



`y = (y₁,y₂,...,yₙ)`



the Euclidean distance is:



`d(x,y) = √\[Σ(xᵢ-yᵢ)²]`



For two-dimensional data:



`d = √\[(x₁-y₁)² + (x₂-y₂)²]`



\---



\## 2.3 Classification



Suppose:



`K = 5`



For a new observation:



1\. Calculate distances to training observations.

2\. Select the 5 closest observations.

3\. Count their classes.

4\. Assign the majority class.



Mathematically:



`ŷ = mode{yᵢ : xᵢ ∈ Nₖ(x)}`



where:



`Nₖ(x)` = K nearest neighbors of `x`.



\---



\## 2.4 Choosing K



\### Small K



Example:



`K = 1`



The model is highly sensitive to individual observations.



This can lead to \*\*overfitting\*\*.



\### Large K



A large K produces a smoother decision boundary but may ignore local patterns.



This can lead to \*\*underfitting\*\*.



Therefore, K controls the bias-variance trade-off.



\---



\## 2.5 Feature Scaling



KNN is highly affected by feature magnitude.



Suppose:



\* Age ranges from 20–80

\* Income ranges from 20,000–200,000



Income would dominate the distance calculation.



Standardization is:



`z = (x - μ) / σ`



where:



\* `μ` = mean

\* `σ` = standard deviation



After standardization:



`mean ≈ 0`



`standard deviation ≈ 1`



\---



\## 2.6 Weighted KNN



Instead of giving every neighbor equal importance, closer neighbors can receive greater weight.



A common weighting scheme is:



`wᵢ = 1 / (dᵢ + ε)`



where:



\* `dᵢ` = distance to neighbor

\* `ε` = small value preventing division by zero



Closer observations therefore have greater influence.



\---



\## Key Concepts



\* Distance

\* Nearest neighbors

\* K value

\* Majority voting

\* Euclidean distance

\* Feature scaling

\* Bias-variance trade-off



\---



\# 3. Naive Bayes



\## 3.1 What is Naive Bayes?



Naive Bayes is a probabilistic classification algorithm based on \*\*Bayes' theorem\*\*.



It calculates:



`P(C | X)`



the probability of class `C` given the observed features `X`.



\---



\## 3.2 Bayes' Theorem



The fundamental equation is:



`P(C|X) = \[P(X|C)P(C)] / P(X)`



where:



\* `P(C|X)` = posterior probability

\* `P(X|C)` = likelihood

\* `P(C)` = prior probability

\* `P(X)` = evidence



\---



\## 3.3 Multiple Features



Suppose:



`X = (x₁,x₂,...,xₙ)`



Then:



`P(C|x₁,x₂,...,xₙ)`



Using the Naive Bayes conditional independence assumption:



`P(C|X) ∝ P(C) ∏ P(xᵢ|C)`



Therefore:



`P(C|X) ∝ P(C)P(x₁|C)P(x₂|C)...P(xₙ|C)`



The predicted class is:



`Ĉ = argmax\_C P(C) ∏ P(xᵢ|C)`



\---



\## 3.4 Why "Naive"?



The algorithm assumes:



`P(x₁,x₂,...,xₙ | C)`



can be decomposed into:



`P(x₁|C)P(x₂|C)...P(xₙ|C)`



This means the features are assumed to be conditionally independent given the class.



This assumption is often unrealistic, but Naive Bayes can still perform surprisingly well.



\---



\## 3.5 Prior Probability



The prior probability of a class is:



`P(C) = Number of samples in class C / Total number of samples`



For example, if 60 out of 100 observations belong to Class A:



`P(A) = 60/100 = 0.6`



\---



\## 3.6 Likelihood



Likelihood represents:



`P(X|C)`



It answers:



\*\*How likely are these features if the observation belongs to class C?\*\*



For continuous features, Gaussian Naive Bayes commonly assumes a normal distribution.



The Gaussian probability density function is:



`P(x|C) = 1/√(2πσ²) × exp\[-(x-μ)²/(2σ²)]`



where:



\* `μ` = class-specific mean

\* `σ²` = class-specific variance



\---



\## 3.7 Posterior Probability



The posterior combines prior knowledge with observed evidence:



`Posterior ∝ Likelihood × Prior`



Therefore:



`P(C|X) ∝ P(X|C)P(C)`



The class with the highest posterior probability is selected.



\---



\## 3.8 Log Probability



Because multiplying many small probabilities can cause numerical underflow, implementations often work with logarithms.



Instead of:



`P(C) × ∏P(xᵢ|C)`



we can calculate:



`log P(C) + Σ log P(xᵢ|C)`



The predicted class becomes:



`Ĉ = argmax\_C \[log P(C) + Σ log P(xᵢ|C)]`



\---



\## Key Concepts



\* Bayes' theorem

\* Prior

\* Likelihood

\* Posterior

\* Evidence

\* Conditional independence

\* Gaussian probability density

\* Log probabilities



\---



\# 4. Decision Tree



\## 4.1 What is a Decision Tree?



A Decision Tree recursively divides data into smaller groups using feature-based conditions.



The objective is to create child nodes that are \*\*more pure\*\* than the parent node.



\---



\# 4.2 Entropy



Entropy measures uncertainty or impurity.



The formula is:



`H(S) = -Σ pᵢ log₂(pᵢ)`



where:



\* `S` = dataset

\* `pᵢ` = proportion of samples belonging to class `i`



\---



\## Binary Classification Example



Suppose:



`p₁ = 0.5`



`p₂ = 0.5`



Then:



`H(S) = -(0.5log₂0.5 + 0.5log₂0.5)`



Since:



`log₂(0.5) = -1`



we get:



`H(S) = 1`



So entropy is maximum for a balanced binary dataset.



\---



\### Pure Node



Suppose:



`p₁ = 1`



`p₂ = 0`



Then:



`H(S) = -(1log₂1 + 0log₂0)`



`H(S) = 0`



Therefore:



\*\*Entropy = 0 → completely pure node\*\*



\---



\# 4.3 Information Gain



Information Gain measures how much uncertainty is reduced after a split.



The formula is:



`IG(S,A) = H(S) - Σ \[|Sᵥ|/|S| × H(Sᵥ)]`



where:



\* `S` = parent dataset

\* `A` = feature used for splitting

\* `Sᵥ` = child subset

\* `|Sᵥ|` = number of observations in child

\* `|S|` = number of observations in parent



The Decision Tree prefers the split with \*\*higher Information Gain\*\* when using entropy.



\### In simple words:



`Information Gain = Parent Entropy - Weighted Child Entropy`



Therefore:



\*\*Higher Information Gain → better split\*\*



\---



\# 4.4 Gini Impurity



Another popular splitting criterion is Gini impurity.



Formula:



`Gini(S) = 1 - Σpᵢ²`



For a pure node:



`Gini = 0`



For a balanced binary node:



`Gini = 1 - (0.5² + 0.5²)`



`Gini = 0.5`



\---



\# 4.5 Entropy vs Gini



| Criterion        | Formula                   | Objective                  |

| ---------------- | ------------------------- | -------------------------- |

| Entropy          | `-Σpi log₂(pi)`           | Measure uncertainty        |

| Information Gain | `H(parent) - H(children)` | Maximize entropy reduction |

| Gini             | `1 - Σpi²`                | Minimize impurity          |



In scikit-learn:



```python

DecisionTreeClassifier(criterion="gini")

```



uses Gini by default.



Entropy can be selected using:



```python

DecisionTreeClassifier(criterion="entropy")

```



\---



\# 4.6 Recursive Splitting



A Decision Tree repeatedly performs:



```text

Choose feature

&#x20;     ↓

Choose threshold

&#x20;     ↓

Calculate impurity

&#x20;     ↓

Evaluate split

&#x20;     ↓

Choose best split

&#x20;     ↓

Create child nodes

&#x20;     ↓

Repeat

```



The process continues until a stopping condition is reached.



\---



\# 4.7 Overfitting



A very deep tree can memorize the training data.



Important parameters include:



\* `max\_depth`

\* `min\_samples\_split`

\* `min\_samples\_leaf`

\* `max\_leaf\_nodes`



Controlling tree complexity helps improve generalization.



\---



\## Key Concepts



\* Root node

\* Internal node

\* Leaf node

\* Entropy

\* Information Gain

\* Gini impurity

\* Splitting

\* Threshold

\* Overfitting

\* Pruning



\---



\# 5. Support Vector Machine (SVM)



\## 5.1 What is SVM?



Support Vector Machine attempts to find a decision boundary that separates classes while maximizing the margin.



For a linear classifier:



`wᵀx + b = 0`



This equation represents the separating hyperplane.



\---



\# 5.2 Hyperplane



For binary classification:



`f(x) = wᵀx + b`



The predicted class can be represented as:



`ŷ = sign(wᵀx + b)`



Therefore:



\* `wᵀx + b > 0` → one class

\* `wᵀx + b < 0` → other class



\---



\# 5.3 Margin



For a hard-margin SVM, the supporting hyperplanes are:



`wᵀx + b = +1`



and:



`wᵀx + b = -1`



The distance between these two boundaries is:



`Margin = 2 / ||w||`



Therefore, maximizing the margin is equivalent to minimizing:



`1/2 ||w||²`



subject to:



`yᵢ(wᵀxᵢ + b) ≥ 1`



\---



\# 5.4 Support Vectors



The observations closest to the decision boundary are called:



\*\*Support Vectors\*\*



They satisfy:



`yᵢ(wᵀxᵢ + b) = 1`



These points are especially important because they determine the optimal separating hyperplane.



\---



\# 5.5 Soft-Margin SVM



Real-world data is often not perfectly separable.



SVM introduces slack variables:



`ξᵢ ≥ 0`



The optimization becomes:



`minimize 1/2 ||w||² + CΣξᵢ`



subject to:



`yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ`



where:



\* `ξᵢ` = amount of constraint violation

\* `C` = penalty parameter



\---



\## Meaning of C



\### Large C



The model strongly penalizes classification errors.



This may produce a smaller margin.



\### Small C



The model allows more violations.



This encourages a wider margin.



\---



\# 5.6 Kernel Trick



A linear boundary may not be sufficient for nonlinear data.



SVM can use a \*\*kernel function\*\*.



Instead of explicitly transforming:



`x → φ(x)`



the kernel computes:



`K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ)`



This is known as the \*\*kernel trick\*\*.



It allows SVM to operate as though the data had been mapped into a higher-dimensional feature space.



\---



\# 5.7 Linear Kernel



The linear kernel is:



`K(xᵢ,xⱼ) = xᵢᵀxⱼ`



It is useful when the data is approximately linearly separable.



\---



\# 5.8 Polynomial Kernel



The polynomial kernel can be written as:



`K(xᵢ,xⱼ) = (γxᵢᵀxⱼ + r)^d`



where:



\* `γ` = scaling parameter

\* `r` = coefficient

\* `d` = polynomial degree



It allows polynomial decision boundaries.



\---



\# 5.9 RBF Kernel



RBF stands for \*\*Radial Basis Function\*\*.



The RBF kernel is:



`K(xᵢ,xⱼ) = exp(-γ||xᵢ-xⱼ||²)`



This kernel is widely used for nonlinear classification.



\---



\## 5.10 Gamma



Gamma controls the influence of individual training observations in an RBF kernel.



\### Low gamma



A point has a broader influence.



Decision boundaries tend to be smoother.



\### High gamma



A point has a more localized influence.



The decision boundary can become highly complex.



Very high gamma can cause overfitting.



\---



\# 5.11 Feature Scaling



SVM is sensitive to feature magnitudes.



A common transformation is standardization:



`z = (x - μ) / σ`



Scaling prevents features with large numerical ranges from dominating the model.



\---



\# Key Concepts



\* Hyperplane

\* Margin

\* Support vectors

\* Hard margin

\* Soft margin

\* Slack variables

\* C parameter

\* Kernel trick

\* Linear kernel

\* Polynomial kernel

\* RBF kernel

\* Gamma

\* Feature scaling



\---



\# 6. Classification Metrics



After training the models, we need to evaluate their performance.



\---



\## 6.1 Confusion Matrix



|                 | Predicted Negative | Predicted Positive |

| --------------- | -----------------: | -----------------: |

| Actual Negative |                 TN |                 FP |

| Actual Positive |                 FN |                 TP |



Where:



\* \*\*TP\*\* = True Positive

\* \*\*TN\*\* = True Negative

\* \*\*FP\*\* = False Positive

\* \*\*FN\*\* = False Negative



\---



\# 6.2 Accuracy



Accuracy measures the proportion of all predictions that are correct.



`Accuracy = (TP + TN) / (TP + TN + FP + FN)`



\---



\# 6.3 Precision



Precision measures how many predicted positives were actually positive.



`Precision = TP / (TP + FP)`



High precision means relatively few false positives.



\---



\# 6.4 Recall / Sensitivity



Recall measures how many actual positive cases were correctly identified.



`Recall = TP / (TP + FN)`



High recall means relatively few false negatives.



\---



\# 6.5 Specificity



Specificity measures how many actual negative cases were correctly identified.



`Specificity = TN / (TN + FP)`



\---



\# 6.6 F1 Score



F1 score combines precision and recall.



`F1 = 2 × (Precision × Recall) / (Precision + Recall)`



It is especially useful when you need a balance between precision and recall.



\---



\# 7. Overall Mathematical Comparison



| Algorithm           | Mathematical Foundation                             |

| ------------------- | --------------------------------------------------- |

| Logistic Regression | Sigmoid + Maximum Likelihood / Cross-Entropy        |

| KNN                 | Distance + Neighbor Voting                          |

| Naive Bayes         | Bayes' Theorem + Conditional Independence           |

| Decision Tree       | Entropy / Information Gain / Gini                   |

| SVM                 | Margin Maximization + Convex Optimization + Kernels |



\---



\# 8. Scaling Requirements



| Algorithm           | Usually Scale Features? | Reason                                  |

| ------------------- | ----------------------- | --------------------------------------- |

| Logistic Regression | Yes                     | Coefficient optimization                |

| KNN                 | Yes                     | Distance calculation                    |

| Naive Bayes         | Usually not required    | Probability-based                       |

| Decision Tree       | No                      | Split thresholds are scale-invariant    |

| SVM                 | Yes                     | Distance/margin and kernel calculations |



\---



\# 9. Important Hyperparameters



\## Logistic Regression



```python

C

penalty

solver

max\_iter

```



\## KNN



```python

n\_neighbors

weights

metric

```



\## Naive Bayes



For Gaussian Naive Bayes:



```python

var\_smoothing

```



\## Decision Tree



```python

criterion

max\_depth

min\_samples\_split

min\_samples\_leaf

```



\## SVM



```python

C

kernel

gamma

degree

```



\---



\# 10. The Big Picture



These five algorithms approach classification in completely different mathematical ways.



\### Logistic Regression



Asks:



\*\*What probability does a linear combination of features produce?\*\*



\---



\### KNN



Asks:



\*\*Which training observations are closest to this new observation?\*\*



\---



\### Naive Bayes



Asks:



\*\*Given these features, which class has the highest posterior probability?\*\*



\---



\### Decision Tree



Asks:



\*\*Which feature split reduces impurity the most?\*\*



\---



\### SVM



Asks:



\*\*What decision boundary maximizes the separation margin between classes?\*\*



\---



\# 11. Final Learning Summary



The most important formulas to remember are:



\### Logistic Regression



`σ(z) = 1 / (1 + e⁻ᶻ)`



`P(y=1|x) = σ(wᵀx+b)`



\---



\### KNN



`d(x,y) = √Σ(xᵢ-yᵢ)²`



`ŷ = mode(nearest K labels)`



\---



\### Naive Bayes



`P(C|X) = P(X|C)P(C) / P(X)`



`P(C|X) ∝ P(C)∏P(xᵢ|C)`



\---



\### Decision Tree



`H(S) = -Σpᵢlog₂(pᵢ)`



`IG = H(parent) - H(weighted children)`



`Gini = 1 - Σpᵢ²`



\---



\### SVM



`wᵀx+b=0`



`Margin = 2/||w||`



`K(xᵢ,xⱼ)=φ(xᵢ)ᵀφ(xⱼ)`



`K\_RBF(xᵢ,xⱼ)=exp(-γ||xᵢ-xⱼ||²)`



\---



\# Final Takeaway



Do not memorize these algorithms as five unrelated techniques.



Understand the mathematical question each one is answering:



\*\*Logistic Regression → Probability\*\*



\*\*KNN → Distance\*\*



\*\*Naive Bayes → Probability + Bayes\*\*



\*\*Decision Tree → Impurity Reduction\*\*



\*\*SVM → Maximum Margin\*\*



Once these foundations are clear, more advanced algorithms such as \*\*Random Forest, Gradient Boosting, XGBoost, neural networks, and deep learning\*\* become much easier to understand.






