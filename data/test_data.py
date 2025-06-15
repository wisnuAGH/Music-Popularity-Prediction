import pandas as pd

# Wczytanie danych
df = pd.read_csv("data.csv")

# Filtrowanie tylko dla gatunku 'rap'
df_rap = df[df['playlist_genre'].str.lower() == 'rap']

# Wyświetlenie unikatowych artystów
unique_artists = df_rap['track_artist'].unique()

print("data reduced size: ", len(unique_artists))

print("Unique RAP artists:")
for artist in unique_artists:
    print(artist)
