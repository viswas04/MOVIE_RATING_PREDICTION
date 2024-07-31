import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('cartoonmovies.csv')

# preprocessing of dataset
df = df.dropna()
df = df.drop('Title', axis=1)
df = df.drop('Director', axis=1)
df = df.drop('cast', axis=1)

# For example, label encode the "TYPE" column
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

# Use one-hot encoding for nominal categorical features like "COUNTRY" and "LISTEDIN"
df = pd.get_dummies(df, columns=['country', 'ListedIn'], drop_first=True)

# Split the data into train and test sets
X = df.drop('Ratings', axis=1)  
y = df['Ratings']              

# Splitting the dataset into Test and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Accuracy: {mape:.2f}%")

# User input for prediction
def predict_rating(user_country, user_duration, user_listedin):
    user_input = {
        'country': user_country,
        'Duration': float(user_duration),
        'ListedIn': user_listedin
    }

    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_encoded = pd.get_dummies(user_input_df, columns=['country', 'ListedIn'], drop_first=True)
    user_input_encoded = user_input_encoded.reindex(columns=X_train.columns, fill_value=0)
    
    predicted_rating = model.predict(user_input_encoded)[0]
    predicted_rating = round(predicted_rating, 2)
    return predicted_rating

