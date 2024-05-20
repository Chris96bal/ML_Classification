train_file = "CS98XClassificationTrain.csv"
test_file = "CS98XClassificationTest.csv"

import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

def prep_data(train_file, test_file):
    start_time = time.time()

    # Load train and test data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    # Store 'top genre' column and drop 'Id' & 'top genre' & 'title' column
    song_genre = train['top genre']
    train.drop(columns=['Id', 'title', 'top genre'], inplace=True)
    
    # Drop any null values and reset index. Then do the same process for the test file
    train.dropna(inplace=True)
    train.reset_index(drop=True, inplace=True)
    test.drop(columns=['Id', 'title'], inplace=True)
    test.dropna(inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    # Perform one-hot encoding
    combined = pd.concat([train, test])
    combined_encoded = pd.get_dummies(combined, columns=['artist'])
    
    # Split the DataFrame vertically at original row size
    train_encoded = combined_encoded.iloc[:len(train)]
    test_encoded = combined_encoded.iloc[len(train):].reset_index(drop=True)
    
    end_time = time.time()
    
    return train_encoded, test_encoded, end_time - start_time

train_encoded, test_encoded, exec_time = prep_data(train_file, test_file)
print("Data preprocessing completed in {:.2f} seconds.".format(exec_time))

