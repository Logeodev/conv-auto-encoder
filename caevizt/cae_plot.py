import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import choice
from pandas import Series
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from cae import CAEVizT


def plot_history(history, figsize=(10, 6)):
    """
    Plot the training history.
    
    Parameters
    ----------
    history : keras.callbacks.History
        Training history object

    figsize : tuple, default=(10, 6)
        Figure size
    """
    
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], label="Train Loss")
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label="Test Loss")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
def plot_latent_space(cae:CAEVizT, X, y=None, figsize=(10, 8)):
    """
    Plot the first 2 dimensions of the latent space with cluster assignments.
    
    Parameters
    ----------
    cae : CAEVizT
        Fitted CAEVizT model

    X : array-like of shape (n_samples, seq_length) or (n_samples, n_features)
        Input data
    
    y : array-like of shape (n_samples,), optional
        True labels for color-coding, if available
    
    figsize : tuple, default=(10, 8)
        Figure size
    """
    if cae.cluster_model_ is None:
        raise ValueError("Model has not been fit yet. Call 'fit' before using 'plot_latent_space'.")
    
    # Get latent representations
    latent_features = cae.transform(X)
    
    # Get cluster assignments
    cluster_labels = cae.predict_clusters(X)
    
    # Use only the first 2 dimensions for plotting
    if latent_features.shape[1] < 2:
        raise ValueError("Latent space dimension must be at least 2 for visualization.")
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    tsnes = [TSNE(n_components=2, perplexity=p) for p in [10, 30, 60]]
    latent_2d_pca = pca.fit_transform(latent_features)
    latent_2d_tsnes = [tsne.fit_transform(latent_features) for tsne in tsnes]

    
    plt.figure(figsize=figsize)
    
    # Plot with true labels if provided
    if y is not None:
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='True Label')
        plt.title('Latent Space (True Labels)')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True)
            
    # Plot with cluster assignments
    n_plots = 1 + len(latent_2d_tsnes)  # PCA + all TSNE plots
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0]*n_plots/2, figsize[1]))
    
    # PCA plot
    scatter = axes[0].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=cluster_labels, cmap='rainbow', alpha=0.7)
    axes[0].set_title('PCA')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].grid(True)
    
    # TSNE plots with different perplexities
    for i, (latent_2d_tsne, perplexity) in enumerate(zip(latent_2d_tsnes, range(10, 51, 10))):
        axes[i+1].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=cluster_labels, cmap='rainbow', alpha=0.7)
        axes[i+1].set_title(f'TSNE (perp={perplexity})')
        axes[i+1].set_xlabel('Dimension 1')
        axes[i+1].grid(True)
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.6, label='Cluster')
    
    plt.tight_layout()
    plt.show()
    
def plot_reconstruction(cae:CAEVizT, X, figsize=(12, 8)):
    """
    Plot original vs reconstructed time series.
    
    Parameters
    ----------
    cae : CAEVizT
        Fitted CAEVizT model

    X : array-like of shape (n_samples, 1)
        Input data
    
    figsize : tuple, default=(12, 8)
        Figure size
    """    
    X_recon = cae.predict(X)
    if X_recon.shape != X.shape:
        X_recon = X_recon[:X.shape[0]]

    plt.figure(figsize=figsize)
    plt.plot(X[:, 0], label='Original', color='darkblue', alpha=0.7)
    plt.plot(X_recon[:, 0], label='Reconstructed', color='blue', alpha=0.7, linestyle='--')
    plt.title('Original vs Reconstructed Time Series')
    plt.legend()
    plt.show()

def plot_silhouette(cae:CAEVizT, X, figsize=(10, 6)):
    """
    Plot the silhouette scores for each cluster.
    
    Parameters
    ----------
    cae : CAEVizT
        Fitted CAEVizT model
    """
    if cae.cluster_model_ is None:
        raise ValueError("Model has not been fit yet. Call 'fit' before using 'plot_silhouette'.")
    
    # Get silhouette scores
    cluster_labels = cae.predict_clusters(X)
    silhouette_scores = silhouette_score(cae.flatten_data(X), cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(cae.n_clusters), silhouette_scores)
    plt.title('Silhouette Scores by Cluster')
    plt.xlabel('Cluster')
    plt.xticks(range(cae.n_clusters))
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()
    
def plot_clustered_data(cae:CAEVizT, X, figsize=(10,6)):
    """
    Plot the clustered data, by predicting cluster label in latent space
    and transform it back into the original.
    
    Parameters
    ----------
    cae : CAEVizT
        Fitted CAEVizT model

    X : array-like of shape (n_samples, seq_length) or (n_samples, n_features)
        Input data
    
    figsize : tuple, default=(10, 6)
        Figure size
    """
    if cae.cluster_model_ is None:
        raise ValueError("Model has not been fit yet. Call 'fit' before using 'plot_clustered_data'.")
    
    # Get cluster assignments
    cluster_labels = cae.predict_clusters(X)    
    
    plt.figure(figsize=figsize)
    # Plot original data
    plt.plot(X[:, 0], label='Original', color='darkblue', alpha=0.7)
    
    # Plot each cluster with different colors and add to legend
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        plt.scatter(np.where(mask)[0], X[mask, 0], 
                   label=f'Cluster {cluster_id}',
                   alpha=0.7)
    plt.title('Original vs Clustered Time Series')
    plt.legend()
    plt.show()

def plot_latent_contour_spectrum(cae, X, anomalia_clusters=None, smothing_window_size=10, tolerance=0.1, spectrum_width_coef=4.0, figsize=(16, 8), show_cluster_labels=False):
    """
    Plot spectral boundaries computed in latent space and projected to original space.
    
    Parameters
    ----------
    cae : CAEVizT
        Fitted CAEVizT model
    X : array-like
        Input data
    anomalia_clusters : list or None
        List of cluster IDs to consider as anomalies
    smothing_window_size : int
        Size of the smoothing window for the median and bounds
    tolerance : float
        Tolerance for points outside the normal spectrum
    spectrum_width_coef : float
        Coefficient to expand the spectrum width
    figsize : tuple
        Figure size
    show_cluster_labels : bool
        Whether to show cluster labels in the plot
    """
    from scipy.stats import gaussian_kde
    from skimage import measure
    from scipy.interpolate import interp1d
    
    # Get latent features and cluster assignments
    latent_features = cae.transform(X)
    cluster_labels = cae.predict_clusters(X)
    
    # If anomalia_clusters not provided
    if anomalia_clusters is None:
        min_anomalia_ratio = 1.0
        selected_cluster = np.max(cluster_labels)
        for cluster_id in np.unique(cluster_labels):
            cluster_size = np.sum(cluster_labels == cluster_id)
            if cluster_size < min_anomalia_ratio * len(X):
                min_anomalia_ratio = cluster_size / len(X)
                selected_cluster = cluster_id
        anomalia_clusters = [selected_cluster]
    
    # Create masks for normal and anomalous points
    anomalia_mask = np.isin(cluster_labels, anomalia_clusters)
    normal_mask = ~anomalia_mask
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Reduce latent space to 2D for visualization using PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_features)
    
    # Visualize points in latent space
    ax1.scatter(latent_2d[normal_mask, 0], latent_2d[normal_mask, 1], 
               label='Normal', alpha=0.7, c='blue')
    ax1.scatter(latent_2d[anomalia_mask, 0], latent_2d[anomalia_mask, 1], 
               label='Anomaly', alpha=0.7, c='red')
    
    # Create a mesh grid for density estimation
    x_min, x_max = latent_2d[:, 0].min() - 1, latent_2d[:, 0].max() + 1
    y_min, y_max = latent_2d[:, 1].min() - 1, latent_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    
    # Compute KDE for normal and anomalous points
    try:
        normal_points = latent_2d[normal_mask]
        if len(normal_points) > 1:
            normal_kde = gaussian_kde(normal_points.T, bw_method='scott')
            normal_density = normal_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            normal_density = normal_density / np.max(normal_density) if np.max(normal_density) > 0 else normal_density
        else:
            normal_density = np.zeros_like(xx)
        
        anomaly_points = latent_2d[anomalia_mask]
        if len(anomaly_points) > 1:
            anomaly_kde = gaussian_kde(anomaly_points.T, bw_method='scott')
            anomaly_density = anomaly_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            anomaly_density = anomaly_density / np.max(anomaly_density) if np.max(anomaly_density) > 0 else anomaly_density
        else:
            anomaly_density = np.zeros_like(xx)
        
        # Compute decision map: higher values indicate normal regions
        decision_map = normal_density - anomaly_density
        
        # Plot the decision heatmap
        im = ax1.pcolormesh(xx, yy, decision_map, cmap=plt.get_cmap('bwr').reversed(), alpha=0.3, vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax1, label='Normal vs Anomaly Density')
        
        # Find contours at decision boundary
        contours = measure.find_contours(decision_map, 0.0)
        
        # If no contours found at 0, try at median value
        if not contours:
            median_value = np.median(decision_map)
            contours = measure.find_contours(decision_map, median_value)
        
        # Plot the decision boundary contours in latent space
        for contour in contours:
            # Scale contour back to original coordinates
            y_contour = contour[:, 0] / 100 * (y_max - y_min) + y_min
            x_contour = contour[:, 1] / 100 * (x_max - x_min) + x_min
            ax1.plot(x_contour, y_contour, 'k-', linewidth=2)
    
    except Exception as e:
        print(f"Error computing latent space boundaries: {e}")
    
    ax1.set_title('Latent Space with Decision Boundaries')
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_xlim(x_min + 0.8, x_max - 0.8)
    ax1.set_ylim(y_min + 0.8, y_max - 0.8)
    ax1.legend()
    
    # Plot time series in original space
    ax2.plot(X[:, 0], label='Time Series', color='darkblue', alpha=0.5)
    
    # Plot normal and anomalous points
    if show_cluster_labels:
        for cluster_id in np.unique(cluster_labels):
            if cluster_id not in anomalia_clusters:
                mask = cluster_labels == cluster_id
                ax2.scatter(np.where(mask)[0], X[mask, 0], 
                        label=f'Cluster {cluster_id} (Normal)', alpha=0.7)
        
        for cluster_id in anomalia_clusters:
            if cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                ax2.scatter(np.where(mask)[0], X[mask, 0], 
                        label=f'Cluster {cluster_id} (Anomaly)',
                        alpha=0.7, marker='x', color='red', s=100)
    
    # Create time series boundary based on the distribution of normal points
    x_indices = np.arange(len(X))
    y_values = X[:, 0]
    
    # Calculate median and bounds for normal points at each time step
    time_steps = 50  # Divide the time series into segments
    time_bins = np.linspace(0, len(X), time_steps + 1)
    bin_centers = (time_bins[1:] + time_bins[:-1]) / 2
    
    medians = []
    upper_bounds = []
    lower_bounds = []
    
    for i in range(time_steps):
        bin_mask = (x_indices >= time_bins[i]) & (x_indices < time_bins[i+1]) & normal_mask
        if np.sum(bin_mask) > 0:
            bin_values = y_values[bin_mask]
            medians.append(np.median(bin_values))
            upper_bounds.append(np.percentile(bin_values, 90))
            lower_bounds.append(np.percentile(bin_values, 10))
        else:
            # If no normal points in this bin, use NaN
            medians.append(np.nan)
            upper_bounds.append(np.nan)
            lower_bounds.append(np.nan)
    
    # Remove NaNs
    valid_mask = ~np.isnan(medians)
    valid_bin_centers = bin_centers[valid_mask]
    valid_medians = np.array(medians)[valid_mask]
    valid_upper = np.array(upper_bounds)[valid_mask]
    valid_lower = np.array(lower_bounds)[valid_mask]
    
    if len(valid_bin_centers) > 1:
        # Create interpolation functions
        median_interp = interp1d(valid_bin_centers, valid_medians, 
                                bounds_error=False, fill_value='extrapolate')
        upper_interp = interp1d(valid_bin_centers, valid_upper, 
                               bounds_error=False, fill_value='extrapolate')
        lower_interp = interp1d(valid_bin_centers, valid_lower, 
                               bounds_error=False, fill_value='extrapolate')
        
        # Create smooth curves
        def smooth_curve(curve):
            """Smooth the curves (moving average)"""
            return Series(curve)\
                .rolling(
                    window=smothing_window_size, 
                    min_periods=max(1, smothing_window_size//10), 
                    center=True
                ).mean().values
        
        x_curve = np.linspace(0, len(X)-1, len(X))  # Ensure x_curve matches the length of X
        median_curve = smooth_curve(median_interp(x_curve))
        upper_curve = smooth_curve(upper_interp(x_curve))
        lower_curve = smooth_curve(lower_interp(x_curve))

        expand_coef = spectrum_width_coef
        upper_curve = median_curve + expand_coef * (upper_curve - median_curve)
        lower_curve = median_curve - expand_coef * (median_curve - lower_curve)
        
        # Plot the curves
        ax2.plot(x_curve, median_curve, 'b--', linewidth=2, label='Median (Normal)')
        ax2.plot(x_curve, upper_curve, 'g-', linewidth=2, label='Upper Bound')
        ax2.plot(x_curve, lower_curve, 'g-', linewidth=2, label='Lower Bound')
        
        # Fill the area between boundaries
        ax2.fill_between(x_curve, lower_curve, upper_curve, 
                        alpha=0.2, color='green', label='Normal Spectrum')
        
        # Find points outside the bounds
        x_outside_upper = []
        y_outside_upper = []
        x_outside_lower = []
        y_outside_lower = []
        
        for i, val in enumerate(X[:, 0]):
            closest_idx = np.argmin(np.abs(x_curve - i))
            if closest_idx < len(upper_curve):
                if val > upper_curve[closest_idx] * (1 + tolerance):
                    x_outside_upper.append(i)
                    y_outside_upper.append(val)
                elif val < lower_curve[closest_idx] * (1 + tolerance):
                    x_outside_lower.append(i)
                    y_outside_lower.append(val)
        
        # Plot points outside the bounds
        if x_outside_upper:
            ax2.scatter(x_outside_upper, y_outside_upper, color='red', marker='^',
                      s=80, label='Above Normal Spectrum', zorder=10)
        if x_outside_lower:
            ax2.scatter(x_outside_lower, y_outside_lower, color='orange', marker='v',
                      s=80, label='Below Normal Spectrum', zorder=10)
    
    ax2.set_title('Time Series with Projected Boundaries')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


if __name__ == '__main__':
    # Generate sample time-series data
    n_samples = 10000
    n_features = 1
    
    # Create three distinct patterns for clustering
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Pattern 1: Sine wave (normal)
    X[:n_samples // 3, 0] = np.sin(np.linspace(0, 4 * np.pi, n_samples // 3))
    y[:n_samples // 3] = 0
    
    # Pattern 2: Square wave (normal)
    X[n_samples // 3:2 * n_samples // 3, 0] = np.sign(np.sin(np.linspace(0, 4 * np.pi, n_samples // 3)))
    y[n_samples // 3:2 * n_samples // 3] = 0
    
    # Pattern 3: Sawtooth wave (anomaly)
    X[2 * n_samples // 3:, 0] = (np.linspace(0, 8, n_samples // 3 + 1) % 2) - 1
    y[2 * n_samples // 3:] = 1  # Mark as anomaly
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1), y.reshape(-1,), test_size=0.2, random_state=42)

    X_train = CAEVizT.reshape_data(X_train)
    X_test = CAEVizT.reshape_data(X_test)

    # Create and fit the model
    cae = CAEVizT(
        input_shape=X_train.shape[1:],
        latent_dim=16,
        filters=[32, 8, 4],
        n_clusters=4,
        clustering_model='kmeans'
    )
    
    # Unsupervised training first
    print("\n=== Unsupervised Training Phase ===")
    cae.fit(
        X_train, 
        epochs=15, 
        batch_size=64,
        pretraining_epochs=10,
        spectrum_epochs=4)
    
    # Plot the training history
    plot_history(cae.history_)
    
    # Plot the latent space after unsupervised training
    print("\n=== Latent Space Visualization (After Unsupervised Training) ===")
    plot_latent_space(cae, X_test, y_test)
    plot_silhouette(cae, X_test)
    
    # Plot reconstructions
    plot_reconstruction(cae, X_test[:100])

    # Plot clustered data
    plot_clustered_data(cae, X_test[:100])

    # Plot spectral boundaries (unsupervised)
    fig_unsupervised = plot_latent_contour_spectrum(
        cae, X_test[:100], 
        smothing_window_size=10, 
        tolerance=0.1, 
        spectrum_width_coef=4.0, 
        figsize=(16, 8), 
        show_cluster_labels=True
    )
    
    # Prepare data for semi-supervised fine-tuning
    # Create a smaller labeled dataset for fine-tuning
    n_labeled = 500
    
    # Select random samples
    normal_indices = np.where(y_train == 0)[0]
    anomaly_indices = np.where(y_train == 1)[0]
    
    # Take a subset of normal and anomaly samples
    np.random.seed(42)
    selected_normal = np.random.choice(normal_indices, min(400, len(normal_indices)), replace=False)
    selected_anomaly = np.random.choice(anomaly_indices, min(100, len(anomaly_indices)), replace=False)
    
    # Combine into labeled dataset
    labeled_indices = np.concatenate([selected_normal, selected_anomaly])
    X_labeled = X_train[labeled_indices]
    y_labeled = y_train[labeled_indices]
    
    # Fine-tune with labeled data
    print("\n=== Semi-Supervised Fine-tuning with Labeled Anomalies ===")
    cae.fine_tune_supervised(X_labeled, y_labeled, epochs=20, batch_size=32, validation_split=0.2)
    
    # Evaluate performance on test set
    y_pred, y_scores = cae.predict_anomaly(X_test)
    
    # Calculate performance metrics
    from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score
    print("\n=== Anomaly Detection Performance ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_scores):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve After Semi-Supervised Fine-tuning')
    plt.grid(True)
    plt.show()
    
    # Plot latent space with true anomaly labels
    print("\n=== Latent Space Visualization (After Fine-tuning) ===")
    plot_latent_space(cae, X_test, y_test)
    
    # Plot spectral boundaries using predicted anomalies
    # First identify which clusters are mostly anomalies after fine-tuning
    latent_repr = cae.transform(X_test)
    cluster_labels = cae.predict_clusters(X_test)
    
    # Find which clusters contain most anomalies
    anomaly_clusters = []
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() > 0:
            anomaly_ratio = np.mean(y_test[cluster_mask])
            if anomaly_ratio > 0.5:  # If over 50% of cluster points are anomalies
                anomaly_clusters.append(cluster_id)
    
    if not anomaly_clusters:
        # If no cluster has majority anomalies, find the one with the highest ratio
        max_ratio = 0
        max_cluster = 0
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() > 0:
                anomaly_ratio = np.mean(y_test[cluster_mask])
                if anomaly_ratio > max_ratio:
                    max_ratio = anomaly_ratio
                    max_cluster = cluster_id
        anomaly_clusters = [max_cluster]
    
    print(f"Identified anomaly clusters: {anomaly_clusters}")
    
    # Plot spectral boundaries with supervised anomaly information
    fig_supervised = plot_latent_contour_spectrum(
        cae, X_test[:100], 
        anomalia_clusters=anomaly_clusters,
        smothing_window_size=10, 
        tolerance=0.1, 
        spectrum_width_coef=4.0, 
        figsize=(16, 8), 
        show_cluster_labels=True
    )
    
    # Direct comparison using anomaly predictions
    # Create a sample of 100 points from the test set
    sample_indices = np.random.choice(range(len(X_test)), min(100, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    # Get predictions for the sample
    y_pred_sample, _ = cae.predict_anomaly(X_sample)
    
    # Plot using true anomalies
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(X_sample[:, 0], label='Time Series', color='darkblue', alpha=0.5)
    plt.scatter(np.where(y_sample == 0)[0], X_sample[y_sample == 0, 0], 
              label='Normal (True)', alpha=0.7, color='green')
    plt.scatter(np.where(y_sample == 1)[0], X_sample[y_sample == 1, 0], 
              label='Anomaly (True)', alpha=0.7, color='red', marker='x', s=100)
    plt.title('True Anomalies')
    plt.legend()
    
    # Plot using predicted anomalies
    plt.subplot(1, 2, 2)
    plt.plot(X_sample[:, 0], label='Time Series', color='darkblue', alpha=0.5)
    plt.scatter(np.where(y_pred_sample == 0)[0], X_sample[y_pred_sample == 0, 0], 
              label='Normal (Predicted)', alpha=0.7, color='green')
    plt.scatter(np.where(y_pred_sample == 1)[0], X_sample[y_pred_sample == 1, 0], 
              label='Anomaly (Predicted)', alpha=0.7, color='red', marker='x', s=100)
    plt.title('Predicted Anomalies')
    plt.legend()
    
    plt.tight_layout()
    plt.show()