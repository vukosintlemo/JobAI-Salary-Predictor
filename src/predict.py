import pandas as pd
import joblib

def make_prediction():
    #1. Load the "Brain" we trained
    model = joblib.load('../models/salary_model_v1.pkl')
    
    #2. Create a "New Job" - matches the columns the model expects
    new_job = pd.DataFrame({
        'experience_level': ['Senior'],
        'company_size': ['Medium'],
        'remote_type': ['Remote'],
        'industry': ['Tech'],
        'company_type': ['Startup'],
        'min_experience_years': [8],
        'skill_count': [12],
        'employment_type': ['Full-time'],
        'posted_year': [2024]
        })
    
    #3. Predict! (The pipeline handles all the scaling/encoding automatically)
    prediction = model.predict(new_job)
    
    print(f"--- Prediction for New Job ---")
    print(f"Estimated salary: ${prediction[0]:,.2f}")
    
if __name__ == "__main__":
    make_prediction()