"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import scipy as sp

import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import operator # <-- Convienient item retrieval during iteration

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def preprocessing(movie, rating):
    # merge movie and rating to get the title column
    merged_rating = pd.merge(rating, movie[["movieId", "title"]], on='movieId')

    util_matrix = merged_rating.pivot_table(index=['userId'],
                                            columns=['title'],
                                            values='rating') 

    # Normalize each row (a given user's ratings) of the utility matrix
    util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    # Fill Nan values with 0's, transpose matrix, and drop users with no ratings
    util_matrix_norm.fillna(0, inplace=True)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
    # Save the utility matrix in scipy's sparse matrix format
    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

    # Compute the similarity matrix using the cosine similarity metric
    user_similarity = cosine_similarity(util_matrix_sparse.T)
    # Save the matrix as a dataframe to allow for easier indexing  
    user_sim_df = pd.DataFrame(user_similarity, 
                            index = util_matrix_norm.columns, 
                            columns = util_matrix_norm.columns)
    return (merged_rating, util_matrix_norm, user_sim_df)


def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list, top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    
    # Initialize an empty list to store the recommendations for each user

    merged_rating, util_matrix_norm, user_sim_df = preprocessing(movies_df, ratings_df)

    all_recommendations = []
    
    movie_ids = pred_movies(movie_list)
    k = 20

    for user in movie_ids:
        # Cold-start problem - no ratings given by the reference user. 
        # With no further user data, we solve this by simply recommending
        # the top-N most popular books in the item catalog. 
        if user not in user_sim_df.columns:
            recommendations = merged_rating.groupby('title').mean().sort_values(by='rating',
                                                ascending=False).index[:top_n].to_list()
        else:
            # Gather the k users which are most similar to the reference user 
            sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:k+1]
            favorite_user_items = [] # <-- List of highest rated items gathered from the k users  
            most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users

            for i in sim_users:
                # Maximum rating given by the current user to an item 
                max_score = util_matrix_norm.loc[:, i].max()
                # Save the names of items maximally rated by the current user   
                favorite_user_items.append(util_matrix_norm[util_matrix_norm.loc[:, i]==max_score].index.tolist())

            # Loop over each user's favorite items and tally which ones are 
            # most popular overall.
            for item_collection in range(len(favorite_user_items)):
                for item in favorite_user_items[item_collection]: 
                    if item in most_common_favorites:
                        most_common_favorites[item] += 1
                    else:
                        most_common_favorites[item] = 1
            # Sort the overall most popular items and return the top-N instances
            sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
            recommendations = [x[0] for x in sorted_list]
        all_recommendations.append(recommendations)
    
    # Flatten the list of recommendations for all users into a single list
    flattened_recommendations = [book for user_recommendations in all_recommendations for book in user_recommendations]
    
    # Tally the occurrences of each book in the list of recommendations
    book_counts = {}
    for book in flattened_recommendations:
        if book in book_counts:
            book_counts[book] += 1
        else:
            book_counts[book] = 1
    
    # Sort the dictionary of book counts by the occurrence counts and return the top-10 books
    sorted_books = sorted(book_counts.items(), key=operator.itemgetter(1), reverse=True)[:10]
    all_recommendations = [x[0] for x in sorted_books]
        
    return all_recommendations

