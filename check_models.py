import joblib

category_model = joblib.load("models/category_model.joblib")
urgency_model = joblib.load("models/urgency_model.joblib")

print("Category model classes:", list(category_model.classes_))
print("Urgency model classes:", list(urgency_model.classes_))
