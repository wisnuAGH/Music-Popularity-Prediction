from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso


def select_top_features_rfe(X, y, n_features=5):
    estimator = Lasso(alpha=0.1, max_iter=10000)
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    print("\n=== Feature Selection (RFE) ===")
    print("Available features:", list(X.columns))
    print("Selected features:", list(selected_features))
    print("===============================\n")

    return X[selected_features]
