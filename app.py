import pickle
import streamlit as st
import numpy as np

st.header("Games Recommender System using Machine Learning")

try:
    model = pickle.load(open('game_pickle/model.pkl', 'rb'))
    game_name = pickle.load(open('game_pickle/game_name.pkl', 'rb'))
    final_rating = pickle.load(open('game_pickle/final_rating.pkl', 'rb'))
    game_pivot = pickle.load(open('game_pickle/game_pivot.pkl', 'rb'))

    def fecth_poster(suggestion):
      game_name = []
      ids_index = []
      poster_url = []

      for game_id in suggestion:
        game_name.append(game_pivot.index[game_id])

      for name in game_name[0]:
        ids = np.where(final_rating['game_name'] == name)[0][0]
        ids_index.append(ids)

      for idx in ids_index:
        url = final_rating.iloc[idx]['img']
        poster_url.append(url)

      return poster_url

    def recommend_games(game_name):
      game_list = []
      game_id = np.where(game_pivot.index == game_name)[0][0]
      distance, suggestion = model.kneighbors(game_pivot.iloc[game_id,:].values.reshape(1,-1), n_neighbors=4)

      poster_url = fecth_poster(suggestion)

      for i in range(len(suggestion)):
        games = game_pivot.index[suggestion[i]]
        for j in games:
          game_list.append(j)

      return game_list, poster_url

    selected_game = st.selectbox(
        "Type or select a game",
        game_name
    )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

if st.button('Show Recommendation'):
  recommendation_games, poster_url = recommend_games(selected_game)
  col1, col2, col3 = st.columns(3)

  with col1:
    st.text(recommendation_games[1])
    st.image(poster_url[1])
  
  with col2:
    st.text(recommendation_games[2])
    st.image(poster_url[2])

  with col3:
    st.text(recommendation_games[3])
    st.image(poster_url[3])
