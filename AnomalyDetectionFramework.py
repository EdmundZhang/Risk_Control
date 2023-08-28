import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import matthews_corrcoef
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
#from tensorflow.keras.layers import Input, Dense
#from tensorflow.keras.models import Model

# 硬编码数据集
data = pd.DataFrame({
    'A': [1, 5, 9, 13, 17],
    'B': [2, 6, 10, 14, 18],
    'C': [3, 7, 11, 15, 19],
    'D': [4, 8, 12, 16, 20]
})


#histplot:绘制每个数值列的分布图和密度曲线

#corr:计算每对数值列之间的相关系数并绘制热图
#pairplot:绘制每对数值列之间的散点图

#boxplot:类似quantile
#quantile:基于分位数的离群点检测


def explore_data(df,methods):
    print("function:df.head\n")
    print(df.head())
    print("\ndf.describe")
    print(df.describe())
    print("\ndf.isnull().sum")
    print(df.isnull().sum())
    print("\n")
    
    
    for method in methods:
        if method == 'histplot':
            for col in df.select_dtypes(include=np.number).columns:
                plt.figure()
                sns.histplot(df[col], kde=True)
                plt.title(f'{col} distribution')
                plt.show()
        elif method == 'pairplot':
            print("pairplot\n")
            sns.pairplot(df.select_dtypes(include=np.number))
            plt.title('pairplot')
            plt.show()
        elif method == 'boxplot':
            print("boxplot\n")
            sns.boxplot(data=df.select_dtypes(include=np.number))
            plt.title('boxplot')
            plt.show()
        elif method == 'quantile':
            print("quantile\n")
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            print(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum())
        elif method == 'corr':
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        else:
            raise ValueError(f"Invalid method '{method}' specified.")

methods = ['histplot','pairplot','boxplot']
#methods = ['histplot','corr','quantile']
#test
explore_data(data,methods)


# isolation_forest,local_outlier_factor,elliptic_envelope,dbscan,pca,autoencoder

def detect_anomalies(df, methods,contamination=0.1, vote_threshold=2):
    """
    Detects anomalies in a dataset using various anomaly detection methods and a voting mechanism.

    Parameters:
        df (pandas.DataFrame): The dataset to be analyzed.
        methods (list): A list of anomaly detection methods to be used. Can include 'isolation_forest',
                        'local_outlier_factor', 'elliptic_envelope', 'dbscan', 'pca', and 'autoencoder'.
        contamination (float): The proportion of outliers in the dataset. Only used for some methods.
        vote_threshold (int): The minimum number of methods that must identify an observation as an outlier for it to be
                              considered an outlier in the final result.

    Returns:
        pandas.Series: A boolean series indicating which observations are outliers.
    """
    results = pd.DataFrame()

    for method in methods:
        if method == 'isolation_forest':
            # Fit an isolation forest to the data
            model = IsolationForest(contamination=contamination)
            model.fit(df.values)

            # Identify outliers based on the isolation forest
            is_outlier = pd.Series(model.predict(df.values) == -1, index=df.index)

        elif method == 'local_outlier_factor':
            # Fit a local outlier factor model to the data
            model = LocalOutlierFactor(contamination=contamination)
            is_outlier = pd.Series(model.fit_predict(df.values) == -1, index=df.index)

        elif method == 'elliptic_envelope':
            # Fit an elliptic envelope to the data
            model = EllipticEnvelope(contamination=contamination)
            model.fit(df.values)

            # Identify outliers based on the envelope
            is_outlier = pd.Series(model.predict(df.values) == -1, index=df.index)

        elif method == 'dbscan':
            # Cluster the data using DBSCAN
            model = DBSCAN()
            clusters = model.fit_predict(df.values)

            # Identify outliers as observations not assigned to any cluster
            is_outlier = pd.Series(clusters == -1, index=df.index)

        #elif method == 'pca':
        #    # Reduce the data to two dimensions using PCA
        #    pca = PCA(n_components=2)
        #    X = pca.fit_transform(df.values)

        #    # Fit a k-means clustering model to the reduced data
        #    model = DBSCAN()
        #    clusters = model.fit_predict(X)

        #    # Identify outliers as observations not assigned to any cluster
        #    is_outlier = pd.Series(clusters == -1, index=df.index)
            
        #elif method == 'autoencoder':
        #    # Define an autoencoder model with a single hidden layer
        #    input_layer = Input(shape=(df.shape[1],))
        #    hidden_layer = Dense(2, activation='relu')(input_layer)
        #    output_layer = Dense(df.shape[1], activation='linear')(hidden_layer)
        #    model = Model(inputs=input_layer, outputs=output_layer)

        #    # Compile and fit the model to the data
        #    model.compile(optimizer='adam', loss='mse')
        #    model.fit(df.values, df.values, epochs=10, batch_size=32, verbose=0)

        #    # Calculate the mean squared error for each observation
        #    mse = np.mean(np.square(df.values - model.predict(df.values)), axis=1)

        #    # Identify outliers based on the mean squared error
        #    is_outlier = pd.Series(mse > np.percentile(mse, 100 * (1 - contamination)), index=df.index)
        
        else:
            raise ValueError(f"Invalid method '{method}' specified.")

        results[method] = is_outlier

    return results

def evaluate_anomaly_detection(df, results):
    """
    Evaluates the results of multiple anomaly detection algorithms.

    Parameters:
        df (pandas.DataFrame): The original dataset.
        results (dict): A dictionary containing the results of each anomaly detection algorithm.

    Returns:
        dict: A dictionary containing the correlation matrix and average correlation of the results.
    """
    # Convert the results to a DataFrame
    result_df = pd.DataFrame(results)

    # Calculate the correlation matrix
    corr_matrix = result_df.corr(method=matthews_corrcoef)

    # Calculate the average correlation
    avg_corr = np.mean(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)])

    return {'corr_matrix': corr_matrix, 'avg_corr': avg_corr}


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

def plot_anomaly_detection(df, results, dimension=2):
    # 使用 PCA 或 KernelPCA 对数据进行降维
    if dimension == 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        df_reduced = pca.fit_transform(df)
    elif dimension == 3:
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components=3, kernel='rbf')
        df_reduced = kpca.fit_transform(df)
    else:
        raise ValueError("Invalid dimension value. Must be 2 or 3.")
    
    # 针对每一列结果，生成一个散点图
    for col in results.columns:
        # 创建一个新的图形
        fig = plt.figure()
        if dimension == 2:
            ax = fig.add_subplot(111)
        elif dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
        
        # 获取异常值和非异常值的索引
        outliers_idx = results[results[col] == True].index
        inliers_idx = results[results[col] == False].index
        
        # 绘制异常值和非异常值的散点图
        if dimension == 2:
            ax.scatter(df_reduced[inliers_idx, 0], df_reduced[inliers_idx, 1], marker='*', label='Inliers')
            ax.scatter(df_reduced[outliers_idx, 0], df_reduced[outliers_idx, 1], marker='o', label='Outliers')
        elif dimension == 3:
            ax.scatter(df_reduced[inliers_idx, 0], df_reduced[inliers_idx, 1], df_reduced[inliers_idx, 2], marker='*', label='Inliers')
            ax.scatter(df_reduced[outliers_idx, 0], df_reduced[outliers_idx, 1], df_reduced[outliers_idx, 2], marker='o', label='Outliers')
        
        # 添加图例和标题
        ax.legend()
        ax.set_title(f"Anomaly detection result for {col}")
        
    # 显示所有图形
    plt.show()

#test
#plot_anomaly_detection(df=data, results=results, dimension=3)

#anomaly_models = ['isolation_forest','local_outlier_factor','elliptic_envelope','dbscan','autoencoder']
anomaly_models = ['isolation_forest','local_outlier_factor','elliptic_envelope','dbscan']
# 检测异常值
results = detect_anomalies(df=data, methods=anomaly_models,vote_threshold=len(anomaly_models)/2)

evaluate_anomaly_detection(df=data,results=results)
plot_anomaly_detection(df=data, results=results, dimension=2)
