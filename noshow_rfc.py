import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import hotel_noshow_pipeline as pipeline

# randomforestclassifier as the first example


# creating a main function in hotel_noshow_ml.py that uses the pipeline
# and implements the ml algorithm

def main():
    final_df = pipeline.main()
    
    X = final_df.drop('no_show', axis=1)
    y = final_df['no_show']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # making predictions
    y_pred = model.predict(X_test)
    
    # evaluations
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    