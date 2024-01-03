Objective:

The primary goal of this analysis is to segment products based on their sales patterns using two clustering techniques: K-means and Spectral Clustering. The dataset comprises weekly sales data for various products, with a focus on columns starting with 'Normalized,' indicating normalized weekly sales figures.
Data Overview:
The dataset, 'Sales_Transactions_Dataset_Weekly.csv,' was loaded, and feature selection was performed to isolate columns related to normalized weekly sales.
Data Preprocessing:
Feature Selection:
Columns starting with 'Normalized' were selected for clustering analysis.
Data Normalization:
StandardScaler was applied to normalize the selected data, ensuring uniformity for clustering algorithms.
Dimensionality Reduction:
Principal Component Analysis (PCA) was employed to reduce the dimensionality of the dataset to two components for effective visualization.
Clustering Analysis:
K-means Clustering:
K-means clustering was applied to partition the data into K clusters based on sales patterns.
Spectral Clustering:
Spectral Clustering, a graph-based approach, was utilized to form clusters by considering the similarity of data points.
Visualization and Comparison:
A line plot of weekly sales trends provides an overview of the overall sales patterns.
Scatter plots for K-means and Spectral Clustering visualize how products are grouped within clusters.



Note on Color Representation:
Colors are applied in the scatter plots to enhance the comparison between different clusters. Each color represents a distinct cluster. It is essential to understand that the choice of colors is arbitrary and is used for visual identification only. 
In the K-means Clustering plot:
Yellow: Cluster 0
Blue: Cluster 1
Green: Cluster 2
In the Spectral Clustering plot:
Yellow: Cluster 0
Blue: Cluster 1
Green: Cluster 2
Quantitative Evaluation:
Silhouette scores were calculated for both K-means and Spectral Clustering, offering a quantitative measure of the cohesion and separation of clusters.
Comparison and Contrast:
K-means:
Strengths:
Well-suited for spherical clusters.
Efficient and easy to implement.
Weaknesses:
Sensitive to initial cluster centers.
Assumes clusters with similar variances.
Spectral Clustering:
Strengths:
Effective in identifying non-linear structures.
Not sensitive to the shape of clusters.
Weaknesses:
Computationally more expensive, especially for large datasets.
Requires careful selection of parameters.


Conclusion:
This analysis provides a comprehensive view of the dataset, emphasizing clustering techniques, weekly sales trends, and a detailed comparison between different clusters. The addition of colors in the scatter plots enhances the visual representation, aiding in the interpretation of clustered results.
