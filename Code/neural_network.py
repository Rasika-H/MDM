import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
 
rating_dataset=pd.read_csv('Downloads/ratings.csv',sep = ',',names = ['userId','movieId','rating', 'timestamp'],skiprows=1)

n_users=rating_dataset['userId'].max()
n_movies=rating_dataset['movieId'].max()

training_data = rating_dataset.sample(frac = 0.8)
test_data = rating_dataset.drop(training_data.index)

user_id_list=list(test_data['userId'])
movie_id_list=list(test_data['movieId'])
rating_list=list(test_data['rating'])

# creating movie embedding vec
movie_input = Input(shape=[1])
movie_embedding = Embedding(n_movies+1, 6)(movie_input)
movie_vec = Flatten()(movie_embedding)

# creating user embedding vec
user_input = Input(shape=[1])
user_embedding = Embedding(n_users+1, 6)(user_input)
user_vec = Flatten()(user_embedding)

# concatenate movie_vec and user_vec 
concat_features = Concatenate()([movie_vec, user_vec])

# Add two fully-connected-layers
fc_layer_1 = Dense(256, activation='relu')(concat_features)
fc_layer_2 = Dense(64, activation='relu')(fc_layer_1)
final = Dense(1)(fc_layer_2)

# Create model and compile it using Augmented Dynamic Adaptive Model (ADAM)
model_final = Model([user_input, movie_input], final)
model_final.compile('adam', 'mean_squared_error')
model_final.fit([training_data.userId, training_data.movieId], training_data.rating, epochs=3, verbose=1)
predictions = model_final.predict([test_data.userId, test_data.movieId])

predicted_rating=list(predictions.T)[0]

cols={'userId':user_id_list,'movieId':movie_id_list, 'rating':rating_list, 'predictedRating':predicted_rating}

predicted_test_data = pd.DataFrame(cols)

predicted_test_data.to_csv('Downloads/nn.csv', index=False)
