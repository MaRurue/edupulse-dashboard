import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def generate_synthetic_data(n_students=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    gpa = rng.uniform(1.0, 4.0, n_students)
    attendance = rng.integers(50, 101, n_students)
    missed_assignments = rng.integers(0, 11, n_students)
    study_hours = rng.uniform(1.0, 40.0, n_students)
    financial_aid = rng.integers(0, 2, n_students)
    # New features
    engagement_score = rng.uniform(0.0, 10.0, n_students)   # 0 = completely disengaged, 10 = highly engaged
    clubs_sports = rng.integers(0, 2, n_students)             # 0 = not involved, 1 = involved

    dropout_risk = []
    for i in range(n_students):
        risk_score = 0
        if gpa[i] < 2.5:                risk_score += 1
        if attendance[i] < 70:          risk_score += 1
        if missed_assignments[i] > 5:   risk_score += 1
        if study_hours[i] < 10:         risk_score += 1
        if financial_aid[i] == 0:       risk_score += 0.5
        # Low engagement increases risk; involvement in clubs/sports is protective
        if engagement_score[i] < 4.0:  risk_score += 0.75
        if clubs_sports[i] == 0:        risk_score += 0.25
        dropout_risk.append(1 if risk_score >= 2.5 else 0)

    df = pd.DataFrame({
        'GPA': gpa,
        'Attendance_Percentage': attendance,
        'Missed_Assignments': missed_assignments,
        'Study_Hours_Per_Week': study_hours,
        'Financial_Aid': financial_aid,
        'Engagement_Score': engagement_score,
        'Clubs_Sports': clubs_sports,
        'Dropout_Risk': dropout_risk
    })
    return df

def train_and_save_model(output_path=None):
    df = generate_synthetic_data()
    feature_cols = [
        'GPA', 'Attendance_Percentage', 'Missed_Assignments',
        'Study_Hours_Per_Week', 'Financial_Aid',
        'Engagement_Score', 'Clubs_Sports'
    ]
    X = df[feature_cols]
    y = df['Dropout_Risk']

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X, y)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'dropout_model.joblib')

    joblib.dump(clf, output_path)
    print(f"Model trained on {len(df)} students with features: {feature_cols}")
    print(f"Dropout rate in training data: {y.mean():.2%}")
    print(f"Saved dropout model to {output_path}")

if __name__ == '__main__':
    train_and_save_model()
