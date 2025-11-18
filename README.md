# Spotify Classification Problem 2023
This project is about a multiclass classification problem resolved with Machine Learning techniques.

551 records of different songs were split into 2 separate files called "CS98XClassificationTrain" and "CS98XClassificationTest". The first file contains the training set containing the first 438 records of songs, while the second file contains the testing set of the remaining 113 records respectively (Finally, there is a 3rd file called "CS98XRegressionTrain" which contains all 113 labels of unknown data).

On both files, each row contains a categorical column of the "song genre" and 11 numerical columns containing general information about the specific song, such as "val", "dur", "acous".

Objective of this project is to build a multiclass classification problem, which will be able to predict the "song genre" using as many of the remaining numerical columns as necessary. The performance metric used, will be accuracy, and the goal will be to build a ML model with the highest accuracy.

This is a supervised learning problem, meaning that we will use already labeled data to train our model and predict results.

To review this project, use the URL: https://www.kaggle.com/competitions/cs9856-spotify-classification-problem-2023 which relocates you to the Kaggle website.

Acknowledgements
Thanks to Nicolas Carbone for providing the original dataset.

https://www.kaggle.com/cnic92/spotify-past-decades-songs-50s10s

Finally, it is important to mention that no modifications will be performed to the "CS98XRegressionTest" file, as in real life scenarios it will have been impossible to have access to it.
