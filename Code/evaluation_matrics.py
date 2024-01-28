import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

# Root mean square error (RMSE)
def rmse(algo):
     rmse = sqrt(mean_squared_error(predicted_test_data.iloc[:, [2]], predicted_test_data.iloc[:, [3]]))
#     rmse = sqrt(mean_squared_error(predicted_test_data['rating'], predicted_test_data["predictedRating"]))
     print("Root Mean Square Error of {} is {}".format(algo, rmse))
    
# Precision and recall    
def precision_recall(algo, test_users):
    precision_recall =[]
    precision_recall_fscore = []
    
    for userId in test_users:
        precision_recall_user = [userId]
        top_recommendations = top_recommend(userId, 10, algo)
        top_ratings = top_actual(userId, 10, algo)
        true_positive = list(set(top_recommendations) & set(top_ratings))
        false_positive =list(set(top_recommendations) - set(top_ratings))
        false_negative =list(set(top_ratings) - set(top_recommendations))
        precision_recall_user.append(float(len(true_positive))/ (len(true_positive) + len(false_positive)))
        precision_recall_user.append(float(len(true_positive))/ (len(true_positive) + len(false_negative)))
        precision_recall.append(precision_recall_user)
    df_precision_recall = pd.DataFrame(precision_recall, columns =['userId', 'precision', 'recall'])
    precision_recall = [df_precision_recall['precision'].mean(), df_precision_recall['recall'].mean()]
    f_score = 2 * precision_recall[0] * precision_recall[1] / (precision_recall[0] + precision_recall[1])
    precision_recall_fscore.append(precision_recall)
    precision_recall_fscore.append(f_score)
    return precision_recall_fscore


# Top recommendation function 
def top_recommend(userId, num, algo): 
    predicted_test_data = pd.read_csv("Downloads/{}.csv".format(algo))[['userId', 'movieId', 'predictedRating']]
    predicted_test_data['movieId_predictedRating'] = predicted_test_data.apply(lambda x: (x['movieId'], x['predictedRating']), axis=1)
    predicted_test_data = predicted_test_data[['userId', 'movieId_predictedRating']]
    predicted_test_data = predicted_test_data.groupby('userId')['movieId_predictedRating'].apply(list).reset_index(name= "recommendations")
    predicted_test_data['recommendations'] = predicted_test_data['recommendations'].apply(lambda x: sorted(x, key=lambda tuple_r: tuple_r[1], reverse=True))
    sortd_userid_recom = pd.Series(predicted_test_data['recommendations'].values, index= predicted_test_data['userId']).to_dict()
    movies_with_ratings = sortd_userid_recom[userId][:num]
    movies = list(zip(*movies_with_ratings))[0]   
    return movies

# Top original rating function
def top_actual(userId, num, algo):
    predicted_test_data = pd.read_csv("Downloads/{}.csv".format(algo))[['userId', 'movieId', 'rating']]
    predicted_test_data['movieId_rating'] = predicted_test_data.apply(lambda x: (x['movieId'], x['rating']), axis=1)
    predicted_test_data = predicted_test_data[['userId', 'movieId_rating']]
    predicted_test_data = predicted_test_data.groupby('userId')['movieId_rating'].apply(list).reset_index(name= "ratings")
    predicted_test_data['ratings'] = predicted_test_data['ratings'].apply(lambda x: sorted(x, key=lambda tuple_r: tuple_r[1], reverse=True))
    sortd_userid_actual = pd.Series(predicted_test_data['ratings'].values, index= predicted_test_data['userId']).to_dict()
    movies_with_ratings = sortd_userid_actual[userId][:num]
    movies = list(zip(*movies_with_ratings))[0]   
    return movies

algo = "Hybrid2_6_2"
predicted_test_data = pd.read_csv("Downloads/{}.csv".format(algo))
test_users = list(predicted_test_data['userId'].unique())
rmse(algo)

precision_recall(algo, test_users)
# top_ten_recommendations_dict[10]
# top_actual(10, 10, algo)
# top_recommend(10, 10, algo)
