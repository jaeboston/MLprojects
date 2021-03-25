import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

class Recommender():
    '''
    This Recommender class uses FunkSVD to make predictions of exact ratings.  
    And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) 
    to make recommendations for users.  
    Finally, if given a movie, the recommender will provide movies that are most 
    similar as a Content Based Recommender.

    '''
    def __init__(self, ):
        '''
        I didn't have any required attributes needed when creating my class.
        '''


    def fit(self, ):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        '''

    def predict_rating(self, ):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''

    def make_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''


if __name__ == '__main__':
    # test different parts to make sure it works
