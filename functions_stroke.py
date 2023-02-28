# Functions for KMeans project

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import tree


############################################################################
## 1) Function to run kmeans and get score on different datasets and/or features
# also runs PCA and creates confusion matrix
# make sure data points aren't listed in any particular order

""" 
Runs KMeans modelling and checking for given dataset

feature_df: feature dataset - includes columns of chosen features
y_data: target variable (string)
kmeans_clusters: number of clusters to use for KMeans model
target_df: stroke target (y) dataframe including id column

"""

def run_kmeans(X_train, y_train, kmeans_clusters = 2):

   
    # Put the features on the same scale

    scaler = StandardScaler()

    X_train_ss = scaler.fit_transform(X_train)

    # run KMeans model

    model = KMeans(n_clusters = kmeans_clusters, random_state = 0)
    kmeans = model.fit(X_train_ss)

    # get the labels for the model
    predicted_labels_k = kmeans.labels_
    print(f'labels: \n\n {predicted_labels_k} \n\n')

        # print score
    print(f"Inertia score for k_means model: \n\n {kmeans.inertia_} \n\n")

    #ref 15

    # print accuracy score
    print(f"Accuracy score for k_means model: \n\n {accuracy_score(y_train, predicted_labels_k)} \n\n")

    # What are the centroids?
    centroids = kmeans.cluster_centers_

    print(f'Centroids \n\n {centroids} \n\n')

# Plot confusion matrix

    conf_matrix=confusion_matrix(y_train, predicted_labels_k) 

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', 
                ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix for K-means model', fontsize=18)
    plt.show()

    stroke_labels = X_train.join(y_train)
    stroke_labels['labels'] = predicted_labels_k

    sns.lmplot(x='avg_glucose_level', y='bmi', data=stroke_labels, hue='stroke', 
        fit_reg=False)
    plt.title('Actual Classification')

    sns.lmplot(x='avg_glucose_level', y='bmi', data=stroke_labels, hue='labels', fit_reg=False)
    plt.title('Predicted Cluster')
    plt.show();

    # dataset with both original stoke column and labels (stoke estimate) column
    stroke_labels.head()

    # part 3 - compare true labels with predicted labels
    print(f"TASK 3: \n Adjusted rand score for k_means: \n\n \
          {adjusted_rand_score(stroke_labels['stroke'], stroke_labels['labels'])} \n\n")

    # run a decision tree classifier

    clf = tree.DecisionTreeClassifier(max_depth = 3, random_state = 0)
    clf = clf.fit(X_train, y_train)

    dec_tree_labels = clf.predict(X_train)

    print(f"Decision tree labels: \n\n {dec_tree_labels} \n\n")

    # Calculate adjusted rand score to compare labels from decision tree with true labels

    print(f"adjusted rand score for decision tree: \n\n {adjusted_rand_score(stroke_labels['stroke'], dec_tree_labels)} \n\n")

# ref 10

    print(f"predict proba: \n\n {clf.predict_proba(X_train)} \n\n")


    # plot decision tree
    plt.figure(figsize=(12,12))
    tree.plot_tree(clf, fontsize = 8)
    plt.show()

    # ref 11

    # print decision tree accuracy score
    print(f"Accuracy score for decision tree model: \n\n {accuracy_score(y_train, dec_tree_labels)} \n\n")

    # Plot confusion matrix

    conf_matrix=confusion_matrix(y_train, dec_tree_labels) 

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', 
                ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix for decision tree model', fontsize=18)
    plt.show()

    

##################################################################
# 2) Create an instance of PCA, function

""" 
Runs principal component analysis

X_train: feature dataset
y_data: target variable (string)
pca_components: number of principal components to use when applying dimensionality reduction

"""

    # Fit Xs
def run_pca(feature_df, y_data, pca_components = 2):

    pca = PCA()

    pca.fit(feature_df)

    # Plot explained_variance_
    # Most of the variance is explained by the first component

    print(f'Explained variance ratios: \n\n {pca.explained_variance_ratio_} \n\n')

    plt.plot(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Explained Variance Ratio')
    plt.show()
    
    # Apply dimensionality reduction to Xs using transform

    pca = PCA(n_components = pca_components)

    pca.fit(feature_df)

    print(f'Explained variance ratio: \n\n {pca.explained_variance_ratio_} \n\n')

    variance = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100

    print(f'Variance explained by the first 2 components: \n\n \
    {"%.4f" % variance}% \n\n')

    # Add the ys back into df and project the data into this 2D space and convert it back to a tidy dataframe
    X_train_y_orig = pd.merge(feature_df, y_data, on = "id", how = 'left')
    nmes = list(feature_df.columns)
    df_2D = pd.DataFrame(pca.transform(feature_df[nmes]),
                     columns=['PCA1', 'PCA2'])
    df_2D['stroke'] = X_train_y_orig['stroke']
    print(df_2D.head())

    # Create PairPlot of PCA

    for key, group in df_2D.groupby(['stroke']):
        plt.plot(group.PCA1, group.PCA2, 'o', alpha=0.7, label=key)

    # Tidy up plot
    plt.legend(loc=0, fontsize=15)
    plt.margins(0.05)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('2D plot of the first and second principal components')
    plt.show()


    