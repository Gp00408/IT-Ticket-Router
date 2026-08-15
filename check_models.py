import os

import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

category_model = joblib.load(os.path.join(MODELS_DIR, "category_model.joblib"))
urgency_model = joblib.load(os.path.join(MODELS_DIR, "urgency_model.joblib"))

print("Category model classes:", list(category_model.classes_))
print("Urgency model classes:", list(urgency_model.classes_))
