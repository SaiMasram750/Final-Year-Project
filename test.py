import joblib
model = joblib.load("isolation_forest_model.pkl")
if hasattr(model, 'feature_names_in_'):
    print(model.feature_names_in_)
else:
    print("No names found. Did you train the model with a DataFrame or a Numpy array?")