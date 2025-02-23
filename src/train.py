import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import argparse

def train_model(data_path='data/music.csv', model_save_path='models/music-recommender.joblib'):
    # Load data
    music_data = pd.read_csv(data_path)
    
    # Prepare features/target
    X = music_data.drop(columns='genre')
    y = music_data['genre']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/music.csv')
    parser.add_argument('--model_save_path', default='models/music-recommender.joblib')
    args = parser.parse_args()
    
    train_model(args.data_path, args.model_save_path)