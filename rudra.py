import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score


urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-test.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-thyroid.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick-euthyroid",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick-euthyroid.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid.theory",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.names"
]
dataframes = [pd.read_csv(url, names=["T3", "T4", "TSH", "Goiter", "Label"]) for url in urls]
data = pd.concat(dataframes, ignore_index=True)


data['Label'] = data['Label'].map({'hypothyroid': 0, 'hyperthyroid': 1, 'normal': 2})
data = data.dropna()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


model = BayesianNetwork([('T3', 'Label'), ('T4', 'Label'), ('TSH', 'Label'), ('Goiter', 'Label')])

model.fit(train_data, estimator=MaximumLikelihoodEstimator)


inference = VariableElimination(model)

y_pred = []
for _, row in test_data.iterrows():
    q = inference.map_query(variables=['Label'], evidence=row.to_dict())
    y_pred.append(q['Label'])

accuracy = accuracy_score(test_data['Label'], y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")