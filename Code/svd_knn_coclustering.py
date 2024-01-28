from surprise import Reader, Dataset
from surprise import SVD, KNNBasic, CoClustering
import pandas as pd

# Using only userId, movieId and rating, excluding timestamp feature
ratings_data = pd.read_csv("Downloads/ratings.csv")[['userId', 'movieId', 'rating']]
       
# Dividing training data and testing data into 80:20 ratio
train_rating_data = ratings_data.sample(frac = 0.8)
test_rating_data = ratings_data.drop(train_rating_data.index)

training_rating_data = Dataset.load_from_df(train_rating_data, reader = Reader(rating_scale=(1,5))).build_full_trainset()

algorithms = {"SVD": SVD(), "KNN Basic": KNNBasic(), "Coclustering": CoClustering()}

def predict_testdata(testdata):
    prediction = algorithms[algo].predict(testdata[0], testdata[1])
    
    # Appending data into a list with predicted rating
    indi_pred_list = []
    indi_pred_list.append(int(prediction[0]))
    indi_pred_list.append(int(prediction[1]))
#     indi_pred_list.append(testdata[0])
#     indi_pred_list.append(testdata[1])
    indi_pred_list.append(testdata[2])
    indi_pred_list.append(prediction[3])
    test_predictions.append(indi_pred_list)

for algo in algorithms:
    algorithms[algo].fit(training_rating_data)
    
    test_predictions = []
    # Get predictions for all the test data using predict_testdata function (defined above)
    test_rating_data.apply(predict_testdata, axis=1)
    
    predicted_test_data = pd.DataFrame.from_records(test_predictions, columns= ['userId', 'movieId', 'rating', 'predictedRating'])
    
    # Write predicted rating with userId, movieId, original rating into csv file
    predicted_test_data.to_csv('Downloads/' +algo+'.csv', index=False)
