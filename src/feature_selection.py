from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
import numpy as np


def select_top_features_rfe(X, y, n_features=5):
    lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
    selector = RFE(estimator=lasso_cv, n_features_to_select=n_features)
    selector.fit(X, y)

    selected_mask = selector.support_
    selected_features = X.columns[selected_mask]

    print("\n=== Feature Selection (RFE + LassoCV) ===")
    print("Wybrane cechy:", list(selected_features))
    print("Alpha wybrane przez LassoCV:", lasso_cv.alphas)
    print("Współczynniki cech:", dict(zip(X.columns, np.round(selector.estimator_.coef_, 4))))
    print("=========================================\n")

    return X[selected_features]
