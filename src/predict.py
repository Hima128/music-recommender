import joblib
import argparse

def predict_genre(age, gender, model_path='models/music-recommender.joblib'):
    model = joblib.load(model_path)
    prediction = model.predict([[age, gender]])
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Music Genre Predictor')
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--gender', type=int, required=True)
    parser.add_argument('--model_path', default='models/music-recommender.joblib')
    
    args = parser.parse_args()
    
    result = predict_genre(args.age, args.gender, args.model_path)
    print(f"Recommended genre: {result}")