📊 Bank Marketing Decision Tree Classifier
📌 Overview
This project builds a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.
We use the Bank Marketing Dataset from the UCI Machine Learning Repository.

The classifier is trained to predict the target variable y:

yes → Customer subscribed to the product/service

no → Customer did not subscribe

📂 Files in This Project
bank.csv → Bank Marketing dataset (semicolon-separated file)

bank_marketing_decision_tree.py → Python script containing preprocessing, training, and evaluation steps

decision_tree_plot.png → Visual representation of the trained Decision Tree

📜 Requirements
Install dependencies before running:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib

🖥️ How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/<your-username>/<your-repo-name>.git
Navigate to the project folder:

bash
Copy
Edit
cd <PRODIGY_DS_03>
Place the bank.csv file in the same directory as the script.

Run the script:

bash
Copy
Edit
python bank_marketing_decision_tree.py
The script will:

Display dataset info & missing values

Encode categorical variables using one-hot encoding

Train a Decision Tree Classifier

Evaluate model performance (accuracy, classification report, confusion matrix)

Save the decision tree visualization as decision_tree_plot.png

🔍 Code Workflow
1️⃣ Data Loading & Inspection
Load dataset (; separated) with Pandas

Display shape, sample rows, and missing values

2️⃣ Data Preprocessing
Convert categorical variables to numeric using pd.get_dummies()

Split data into features (X) and target (y)

3️⃣ Model Training
Use DecisionTreeClassifier from scikit-learn

Train/test split with an 80-20 ratio

4️⃣ Model Evaluation
Accuracy score

Classification report (precision, recall, F1-score)

Confusion matrix

5️⃣ Visualization
Plot the trained decision tree and save it as decision_tree_plot.png

📈 Output Example
Decision Tree Plot


✨ Insights
Decision Trees provide an interpretable model for classification.

Important features such as age, job type, contact method, and previous campaign outcomes influence the prediction.

This model can help in targeting potential customers more effectively.

👨‍💻 Author
Tanmay Gupta
Data Science Intern
