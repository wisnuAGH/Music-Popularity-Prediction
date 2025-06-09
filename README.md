# 🎵 Music Popularity Prediction

Projekt wykorzystujący uczenie maszynowe do przewidywania **popularności piosenki** na podstawie jej cech akustycznych.

## 🎯 Cel projektu
Celem projektu jest stworzenie modelu ML, który przewiduje **popularność utworu muzycznego** (wartość numeryczna od 0 do 100) na podstawie zestawu cech dostępnych np. na:
- [tunebat.com](https://tunebat.com/)
- [Spotify API](https://developer.spotify.com/documentation/web-api/)

Przykładowe cechy wykorzystywane w predykcji:
- `danceability` – rytmiczność
- `energy` – intensywność
- `tempo` – tempo utworu
- `acousticness` – akustyczność
- `valence` – pozytywność utworu
- `speechiness` – ilość mowy w utworze
- i wiele innych

## 🧠 Zastosowane modele ML
Projekt testuje i porównuje kilka modeli regresyjnych:
- Regresja liniowa (Lasso, Ridge)
- Sieci neuronowe (Keras MLP)
- Random Forest
- Gradient Boosting
- XGBoost
- LGBM

## 📈 Dane
Dane wejściowe to plik `.csv` z cechami utworów oraz ich popularnością. Dane zostały poddane:
- Czyszczeniu
- Skalowaniu
- Usunięciu wartości odstających
- Selekcji cech

## 📦 Struktura projektu
