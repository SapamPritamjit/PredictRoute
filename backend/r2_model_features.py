"""R2.2 support - fresh-process reload of catboost_model.pkl.

Prints feature_names_ (exact serving order), pipeline structure and the
expected input dtypes. No predictions, no modification.
"""
import joblib

pipe = joblib.load("catboost_model.pkl")
print("pipeline steps:", [(n, type(t).__name__) for n, t in pipe.steps])
pre = pipe.steps[0][1]
print("preprocessor:", pre)
est = pipe.steps[-1][1]
names = list(est.feature_names_)
print(f"feature_names_ ({len(names)}):")
for i, nm in enumerate(names):
    print(f"  {i:2d} {nm}")
cat_idx = getattr(est, "get_cat_feature_indices", lambda: [])()
print("categorical feature indices:", cat_idx)
print("tree_count_:", est.tree_count_)
