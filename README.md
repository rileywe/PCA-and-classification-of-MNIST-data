# PCA-and-classification-of-MNIST-data
Principal Component Analysis by means of Singular Value Decomposition, V-Mode projection and visualization, and classification through Linear Discriminant Analysis, Support Vector Machine, and Decision tree algorithms.

##### Author: 
Riley Estes

### Abstract
70000 handwritten digit images in the MNIST data set are analysed in order to train a machine learning model to identify handwritten numbers. The dimensions or the data are reduced to identifying the most important singular vectors in classifying the images and to plot the distribution of the digits in 3 dimensions. Each pair of digits is classified using LCA analysis, and one combination of 3 digits is classified with the same method. All ten digits are also classified using an SVM and a Decision Tree. Finally, using the outcome from the 2 digit LCA analysis, each of the methods LCA, SVM, and Decision Tree are used to classify the easiest to differentiate and hardest to differentiate pair of digits. For each model fit, the testing and training errors are calculated and recorded to observe any overtraining. 

### Introduction and Overview
The MNIST data set is a easy to visualize dataset that can still prove a challenge for more rudimentary machine learning algorithms. There is a lot of opportunity to analyse the MNIST data, and this program explores some of the different ways this dataset can be dimensionally reduced, analysed, and classified. Firstly, the data is broken into its SVD components so that only the most important aspects of the images in terms of variance can be extracted from the dataset. Then, with the new dataset drastically reduced in size, the images are reconstructed, and the amount of data stored in the small number of dimensions is shown to be able to almost fully recreate the original image. Then, using three of these dimensions or V-modes, each image is projected onto and plotted in 3D space and visualized. This makes the data distribution very clear, and gives an insight into how a machine learning model might distinguish and draw lines between the data. 
Next, each possible pair of two digits is is put through an LDA algorithm, and the test and training classification error is recorded for each pair. This identifies which digits may be harder to tell apart, and which are easier. It also gives some insight into potential overtraining between some digits if the training and testing errors vary by much. The LDA algorithm is also tested on one combination of three of the digits and both errors are measured. 
Finally, an SVM and decision tree algorithm are tasked to classify all ten of the digits from the dataset, and the errors are observed in much the same way as before. All three methods: LDA, SVM, and decision tree are then trained to differentiate the easiest pair of digits to discern (based on the solo LDA errors from before) as well as to differentiate between the hardest digit pair. This identifies the range of accuracies and difficulty in identifying the digits each algorithm yields, and can help the developer make a decision as to which method is best to classify the MNIST dataset. 

### Theoretical Background
#### Singular Value Decomposition (SVD)
Singular Value Decomposition gives a method to split a matrix into three parts: a left singular matrix, a diagonal matrix of singular values, and a right singular matrix. Given an m x n input matrix A, the SVD of A can be represented as A = UΣV^T, where U is an m x m matrix with columns being the left singular vectors of A, and V is an n x n matrix with columns being The right singular vectors of A. Σ is an m x n diagonal matrix where each value is the standard deviation of the corresponding singular vectors. The left singular vectors represent the basis vectors for the row space in A and are the eigenvectors of AA^T, and the right singular vectors represent the basis vectors for the column vectors in A and are the eigenvectors of A^TA. The left and right singular vectors are of unit length and are orthogonal to each other. Using the eigenvalue properties, the singular vectors corresponding to the greatest standard deviations/variances can be extracted, and then because A = UΣV^T, only the most important vectors in U and V^T can be used to reconstruct the original matrix A. This effectively reduces the amount of raw data and processing power required to classify the original data, and can greatly increase the efficiency of machine learning algorithms as well as show the developer which parts of the input data are most important in classifying the data. 

#### V-mode Projection
V-mode projecting uses the V matrix from the SVD analysis and the singular vectors/modes in it. By taking the dot product of the original matrix with the selected modes in the V matrix, the original matrix can be projected into that space. This takes data in the original matrix from many different dimensions, and condenses them into a few dimensions represented as modes in V.

#### Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis is a supervised classification technique used to identify linear combinations of features to separate classes. LDA finds a set of new variables that are a linear combination of the original features by maximizing the ratio of the between-class variance to the within-class variance. The LDA algorithm seeks to find the direction that maximizes the separation between the class means while minimizing the variation within each class. This direction is called the discriminant function, and it is used to project the data onto a lower-dimensional space. The goal is to find a lower-dimensional space where the classes are as separable as possible. By maximising the separability of each class, LDA can effectively classify points in a dataset. 

#### Support Vector Machine (SVM)
Support Vector Machines attempt to find the best possible boundary between different classes in the data by identifying a hyperplane in a high-dimensional feature space that maximizes the margin (distance from the hyperplane to the nearest data point) between the classes instead of directly minimizing the error. SVMs are particularly useful when the number of dimensions in the feature space is large, and the data is not linearly separable. In such cases, SVMs can use a kernel function to transform the data into a higher-dimensional space, where it may be possible to separate the classes by a hyperplane. This model can very effectively classify data with supervised machine learning. However, it does so at great computational cost with a complexity of at least O(N^2).

#### Decision Tree
Decision trees are another type of classification algorithm that works by splitting the input data into smaller subsets based on the input features until the data cannot be further split, or a certain number of splits have occured. In each new split, the amount of data to consider has been reduced, but there are more data splits that need to be computed as the tree splits and separates further. This method produces an easy to visualize tree-like model where each internal node corresponds to a decision rule based on one or more input features, and each leaf node corresponds to a prediction for the target variable. 

#### Overtraining
Overtraining/overfitting occurs when a machine learning model becomes too complex for the given data and begins to sacrifice smoothness in order to fit the training data more accurately. This means that the model becomes too specialized for the training data and does not generalize well to new, unseen validation data. Overtraining often results from training a model on too small of a dataset, or from making the model too complex and prone to over-adjusting to the training data. To prevent this from occuring, the data should be split into training, testing, and validation data. The training and testing data will give the model a way to fit to the data and test its performance along the way, and the validation data will show the developer how well the model works with entirely new data. A model will generally have a test error close to the training error until a point where the training error continues to drop while the test error starts to rise. To stop the model from overtraining, go to the point where the errors diverge and back up a little bit. 


### Algorithm Implementation and Development
After importing the MNIST data as X and the digit labels as Y, the SVD was calculated using the numpy package:
```
u, s, vt = np.linalg.svd(X, full_matrices=False)
```
Then, an image from the original dataset was displayed with matplotlib and compared to the same image reconstructed with different amounts of singular vectors. The reconstruction code is shown here:
```
u_k = u[:, :k]
s_k = s[:k]
vt_k = vt[:k, :]
X_reconstructed = np.dot(u_k, np.dot(np.diag(s_k), vt_k))
```
Where k = the number of singular vectors in the reconstruction. By comparing the reconstructed image to the original, a value for k could be determined as the point where there is enough data in the singular vectors to accurately reconstruct and classify the data. 

The MNIST data was projected onto three of the top V-modes (right singular vectors) and the digits colored by their label were plotted in 3D with each axis being a V-mode. 

The program then cycles through each possible pair of digits and performs and LCA fit on each pair from the original (unprojected) dataset. Notably, the data was split into training and testing data with:
```
X_train, X_test, y_train, y_test = train_test_split(X_pair, y_pair, test_size=0.2, random_state=0)
```
and the error (training and testing) for the model was calculated as the rate at which the algorithm would misidentify an image. This was computed by summing the successful identifications along the diagonal of an identification matrix and dividing by the total sum of the confusion matrix representing the total number of trials. The training error calculation is shown here:
```
cmTrain = confusion_matrix(y_train, y_pred_train)
error_rate_train = 1 - (1 - np.sum(np.diag(cmTrain)) / np.sum(cmTrain))
```
The results are subsequently sorted to show the easiest and hardest digit pair and the two particular digits in each case are plotted in 3D using the same V-mode dimensional reduction technique as before. 

Using the same LDA method as before, digits 4, 6, and 7 were arbitrarily selected to be classified together and the LDA model was fit and the errors were calculated using LinearDiscriminantAnalysis().score() from sklearn. 

All ten digits in the original MNIST data werte classified next using an SVM and a decision tree. Much the same, calculations were made using the sklearn package, and both errors were recorded. The rbf kernel was used for the SVM.
```
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
```
```
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
```

Finally, the SVM and decision tree methods were tested against with the LDA method in classifying the easiest and the hardest pair of digits to discern. The previously calculated LDA errors were used to compare here, and the SVM and decision tree performances were calculated again using the same sklearn approach. 


### Computational Results
The s matrix from the SVD calculation is graphed and shown here:
###################
And here is a zoomed in version showing the elbow point: the point where the following singular values are not worth keeping.
####################
The elbow point could be between rank 10 and rank 30 based on these images, but the reconstructed values will show how well some of these ranks perform:
Original from dataset:
##########
Reconstructed with rank = 50:

Reconstructed with rank = 20:

Reconstructed with rank = 30:

I've decided rank 30 to be the best as a good, crisp reconstruction, but lower ranks would work if quicker or cheaper calculations are needed. 

After projecting each image to 3 V-modes, the dataset was plotted and the results are shown below. Some images use the same axis but rearranged in order to provide a different perspective on the same distibution.
###############################################################################################################
Notice how data of one type is clearly clustered. Some clusters in the middle are presumably overlapping in some places, while others are on the outside and would be easier to separate.

The LDA errors tested on each pair of digits ranged between 94.977% to 99.753% test accuracy and from 98.386% to 99.850% training accuracy. The most accurate pair was [6, 7], and the least was [5, 8]. Overall there is not much room here to overfit to the training data, but on some of the lower test accuracies the training accuracy was around 1% higher up to a maximum of 1.75%. The LDA model performed very well. 

The [6, 7] combination graph is shown here:
#############
Notice the clear gap between the clusters, with only a little potential overlap on the left side.

The [5, 8] combination graph is shown here:
#############
Notice how both clusters are very close to each other, and while still discerably 2 different clusters, have a lot more overlap than the [6, 7] pair did. 

After calculating the LDA accuracies on the digits [4, 6, 7] all together, the training error is 98.2672%, and the test error is 97.8804%. These values are very close, but the model overtrained by a small amount. The difference falls within the range of train - test accuracy for the 2 digit combinations, so it's hard to tell how 3 digits affected the overtraining. 

For the SVM, 10 digit classification, the model yielded 98.99% training accuracy and 97.64% test accuracy. This is compared to the decision tree's 100% training accuracy and 87.11% testing accuracy. The SVM performed well with a 1.35% train/test difference and 97.64% test accuracy, indicating moderate/low amounts of overfitting and a great overall accuracy. The decision tree on the other hand performed well but subpar compared to the SVM, with a lame 87.11% overall (test) accuracy and egregious 12.89% train/test difference. This model was very much overfit to the training data, and limitations such as number of tree splits should be imposed to stop the training before the test and train accuracies diverge. 

For all three models tested on the [6, 7] pair, they all performed above 99% with the results as follows:
LDA training accuracy: 99.85%
LDA testing accuracy: 99.75%
SVM training accuracy: 99.99%
SVM testing accuracy: 100.00%
Decision Tree training accuracy: 100.00%
Decision Tree testing accuracy: 99.44%
Very little overfitting occured, and the SVM surprisingly got lucky and scored a better testing accuracy than training accuracy. 

for the [5, 8] pair, the results are a little less clean:
LDA training accuracy: 96.73%
LDA testing accuracy: 94.98%
SVM training accuracy: 99.77%
SVM testing accuracy: 99.20%
Decision Tree training accuracy: 100.00%
Decision Tree testing accuracy: 96.00%
The LDA and Decision Tree both overfit by a little bit, the LDA considerably less so than the 4% difference in the decision tree accuracies. The SVM didn't get as lucky this time but still outperformed the other methods by a good margin with above 99% accuracy and a 0.57% accuracy difference.


### Summary and Conclusions
Overall the SVM performed the best but took the most resources to calculate, and the decision tree performed the worst and overtrained a lot, with the LDA falling in the middle closer to the SVM performance. Some of the digit clusters when projected into 3D are pretty well separated, which makes classification much easier when compared to other digit clusters in the MNIST data that were much closer and overlapped a lot more. This was reflected in comparing the errors of each algorithm when tested against the pairs [6, 7] and [5, 8]. The SVD analysis presented a method by which the dimensional data of each image in the MNIST dataset could be drastically reduced without giving up very much important information useful to classify the digits. The elbow point for this data was determined be at most rank = 30, but could be less if more reduction is neccesary, the images would just be less crisp and clear. 
