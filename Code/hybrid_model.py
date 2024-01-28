import pandas as pd

algo = "Hybrid"
dataframe = pd.read_csv("Downloads/SVD.csv")
dataframe.rename(columns= {'predictedRating':'SVD_predicted_rating'}, inplace= True)

predicted_test_data = pd.read_csv("Downloads/KNN Basic.csv")[['userId', 'movieId', 'predictedRating']]
dataframe['KNN_predicted_rating'] = predicted_test_data['predictedRating']

predicted_test_data = pd.read_csv("Downloads/Coclustering.csv")[['userId', 'movieId', 'predictedRating']]
dataframe['CoClustering_predicted_rating'] = predicted_test_data['predictedRating']


dataframe['predictedRating'] = (0.1*dataframe['SVD_predicted_rating']+0.8*dataframe['KNN_predicted_rating']+ 0.1* dataframe['CoClustering_predicted_rating'])
hybrid_dataframe = dataframe[['userId', 'movieId', 'rating', 'predictedRating']]
hybrid_dataframe.to_csv('Downloads/' +algo+'_1_8_1.csv', index=False)

