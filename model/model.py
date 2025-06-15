from tensorflow.keras.models import load_model
import joblib
import pandas as pd

# 1. Wczytaj model i scaler
model = load_model("popularity_model.keras")
scaler = joblib.load("scaler.pkl")

# 2. Dane nowego utworu
new_song = pd.DataFrame([{
    "danceability": 0.65,
    "energy": 0.80,
    "acousticness": 0.05,
    "tempo": 120,
    "valence": 0.9
}])

# 3. Skalowanie (tak samo jak podczas treningu)
new_song_scaled = scaler.transform(new_song)

# 4. Predykcja
predicted_popularity = model.predict(new_song_scaled)[0][0]
print(f"Przewidywana popularność: {predicted_popularity*100:.2f}")
