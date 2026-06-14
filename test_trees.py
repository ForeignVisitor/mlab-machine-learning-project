import numpy as np

from mlab.tree_ensemble._decision_tree import DecisionTree
from mlab.tree_ensemble._random_forest import RandomForest

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [4.0, 4.0],
    [4.0, 5.0],
    [5.0, 4.0],
    [5.0, 5.0]
], dtype=float)

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

tree = DecisionTree(max_depth=3, min_samples_split=2, random_state=42)
tree.fit(X, y)
print("DecisionTree predict:", tree.predict(X))
print("DecisionTree depth:", tree.get_depth())
print("DecisionTree feature_importances_:", tree.feature_importances_)

forest = RandomForest(n_estimators=10, max_depth=3, random_state=42)
forest.fit(X, y)
print("RandomForest predict:", forest.predict(X))
print("RandomForest feature_importances_:", forest.feature_importances_)
print("RandomForest oob_score_:", forest.oob_score_)
print("Number of trees:", len(forest.trees_))