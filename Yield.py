import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Read the dataset
data = pd.read_csv("C:\\Users\\sarav\\OneDrive\\Desktop\\agri\\src\\views\\examples\\crop_yield.csv").drop(
    ['Production', 'Crop_Year', 'Pesticide', 'Annual_Rainfall', 'Fertilizer'], axis=1)

# Strip any leading or trailing whitespace from the 'Season' column
data['Season'] = data['Season'].str.strip()

# Get unique values for each categorical column
unique_crops = data['Crop'].unique().tolist()
unique_seasons = data['Season'].unique().tolist()
unique_states = data['State'].unique().tolist()

# Initialize OneHotEncoder with actual categories from your dataset
encoder = OneHotEncoder(sparse_output=False, categories=[unique_crops, unique_seasons, unique_states], handle_unknown='ignore')

# Encode the features
X_encoded = encoder.fit_transform(data[['Crop', 'Season', 'State']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Crop', 'Season', 'State']))

# Combine with 'Area' column
X_final = pd.concat([X_encoded_df, data[['Area']].reset_index(drop=True)], axis=1)

# Define target variable
Y = data['Yield']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.3)

# Initialize and train the model
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)

# Save the model and encoder to separate files
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('encoder.pkl', 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)
