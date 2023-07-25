"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import requests

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

from utils.faq import faq

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview",
                    "Movie Search", "About Us", "FAQ"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.image('resources/imgs/name_image.png',use_column_width=True)
        # st.write('# Movie Recommender Engine')
        st.write('## ----------- Movie Recommender Engine -----------')
        st.image('resources/imgs/movie_page.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[1000:1500])
        movie_2 = st.selectbox('Second Option',title_list[2000:2500])
        movie_3 = st.selectbox('Third Option',title_list[3001:3500])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    elif page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    elif page_selection == "About Us":
        st.title("About Us")
        st.markdown(
            """
            __Welcome to our Movie Recommender App!__
            
            At RECORDmender, we believe that every movie night should be unforgettable.
            We understand the excitement of discovering new films that captivate your heart
            and transport you to different worlds. That's why we've built a powerful movie
            recommendation engine at your fingertips.

            Our app is designed to take the hassle out of choosing what to watch next.
            With a vast database of movies spanning various genres, we ensure that there's
            something for everyone. Whether you're in the mood for an adrenaline-pumping action
            flick, a heartwarming romantic comedy, a mind-bending sci-fi adventure, or an
            Oscar-worthy drama, we've got you covered.

            But what sets us apart from the rest? Our recommendation algorithm is powered by
            advanced machine learning and artificial intelligence techniques, constantly learning
            and adapting to your unique movie preferences. As you rate movies, browse different
            genres, and add films to your watchlist, our app intelligently tailors recommendations
            specifically to your taste. Say goodbye to endless scrolling and indecisiveness
            we're here to make your movie choices easier than ever before.

            But that's not all. We also provide detailed movie information, including cast 
            and crew, synopsis, ratings, and reviews, giving you all the insights you need to 
            make an informed decision. Plus, our community of movie enthusiasts contributes their 
            own ratings and reviews, making it a hub for lively discussions and recommendations. 

            Whether you're a film buff, a casual viewer, or just looking for a great movie night 
            with friends or family, our Movie Recommender App is your ultimate companion.
            So sit back, relax, and let us take you on a cinematic journey like no other. 
            Join our community today and let the movie magic unfold!"""
        )
        st.header("Meet The Team")
        st.image('resources/imgs/team_bm2.png',use_column_width=True)
    
    elif page_selection == "Movie Search":
        api_key = "87d991ea"
        title = st.text_input("Enter a movie title")
        if title:
            try:
                url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
                re = requests.get(url)
                re = re.json()
                col1, col2= st.columns([1, 2])
                with col1:
                    st.image(re["Poster"])
                with col2:
                    st.subheader(re["Title"])
                    st.caption(f"GENRE: {re['Genre']}, YEAR: {re['Year']} ")
                    st.caption(re["Plot"])
                    st.progress(float(re['imdbRating']) / 10)
                    st.text(f"Rating: {re['imdbRating']} / 10")
            except:
                st.error(f"No movie with title '{title}'")
    elif page_selection == "FAQ":
        st.title("Frequently Asked Questions")
        faq_select = st.selectbox("Select from the dropdown below", list(faq.keys()))
        st.markdown(faq[faq_select])
        # st.selectbox("1. What is the purpose of this movie recommender system app?", faq_list.keys())
        # st.selectbox("1. What is the purpose of this movie recommender system app?", faq_list.keys())
    


        
if __name__ == '__main__':
    main()
