from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import classification_report
from numpy import array, ndarray, unique, arange, sum, isin, mean as average, cov as covariance, percentile, random
from matplotlib.pyplot import subplots, cm, tight_layout, show
from typing import List, Dict, Any, Optional, Tuple
from core.helpers import sequence_clusters_to_point_clusters, unmake_sequences
from insights.helpers import SampleGenerator
from seaborn import boxplot
from matplotlib.pyplot import show, tight_layout, subplots

class ClusterInfoExtractor:
    """Extracts cluster information from trained model and input data."""
    
    def __init__(self, model):
        """Initialize with a trained model.
        
        Parameters
        ----------
        model : CAEVizT
            A trained CAEVizT model
        """
        self.model = model
    
    def base_cluster_info(self, X:ndarray) -> Dict[str, Any]:
        """Extract cluster assignations from the input data.
        Parameters
        ----------
        X : ndarray
            Input data to extract cluster information from
        Returns
        -------
        Dict[str, Any]
            Dictionary containing latent features, cluster labels, and unique clusters
        """
        latent_features = self.model.transform(X)
        cluster_labels = self.model.predict_clusters(X)
        
        # Initial cluster info dictionary
        cluster_info = {
            'latent_features': latent_features,
            'cluster_labels': cluster_labels,
            'unique_clusters': unique(cluster_labels)
        }
        return cluster_info

    def identify_anomaly_clusters_by_distribution(self, decoder, cluster_info: Dict[str, Any], seq_len: int) -> List[int]:
        """Identify anomaly clusters by analyzing their statistical distributions in latent space.
        
        Parameters
        ----------
        decoder : tf.keras.Model
            The decoder part of the CAEVizT model used to project latent features back to original space
        cluster_info : Dict[str, Any]
            Dictionary containing cluster information
        seq_len : int
            Length of the sequences used in the model
            
        Returns
        -------
        List[int]
            List of cluster IDs identified as anomalies
        """        
        # Generate samples in latent space for each cluster
        clusters_samples = {}
        for cluster in cluster_info['unique_clusters']:
            # Get latent features for this cluster
            cluster_mask = cluster_info['cluster_labels'] == cluster
            cluster_latent = cluster_info['latent_features'][cluster_mask]
            
            # If there are enough samples, generate more in latent space
            if len(cluster_latent) >= 5:
                # Calculate mean and covariance for this cluster
                mean = average(cluster_latent, axis=0)
                cov = covariance(cluster_latent, rowvar=False)
                
                # Generate samples using normal distribution around cluster center
                n_samples = 500
                clusters_samples[cluster] = random.multivariate_normal(
                    mean=mean, cov=cov, size=n_samples)
            else:
                # Not enough samples for this cluster, skip it
                clusters_samples[cluster] = cluster_latent
        
        # Project samples back to original space and calculate statistics
        cluster_stats = {}
        for cluster in cluster_info['unique_clusters']:
            # Project samples from latent space through decoder
            projected = unmake_sequences(
                decoder.predict(clusters_samples[cluster], verbose=0), 
                seq_len=seq_len
            ).reshape(-1,)
            
            # Calculate statistics for this cluster's projections
            q25 = percentile(projected, 25)
            q50 = percentile(projected, 50)
            q75 = percentile(projected, 75)
            iqr = q75 - q25
            mean_val = average(projected)
            
            cluster_stats[cluster] = {
                'q25': q25, 
                'q50': q50, 
                'q75': q75, 
                'iqr': iqr,
                'mean': mean_val,
                'n_samples': len(cluster_info['cluster_labels'][cluster_info['cluster_labels'] == cluster])
            }
        
        # Calculate pairwise overlaps between clusters - lower means more separation
        overlap_scores = {}
        for cluster_a in cluster_info['unique_clusters']:
            stats_a = cluster_stats[cluster_a]
            overlap_scores[cluster_a] = 0
            
            for cluster_b in cluster_info['unique_clusters']:
                if cluster_a == cluster_b:
                    continue
                
                stats_b = cluster_stats[cluster_b]
                
                # Calculate overlap between distributions using IQR ranges
                # A higher overlap score means the cluster is more similar to others
                # No overlap = 0, complete overlap = 1
                range_a = [stats_a['q25'] - 1.5 * stats_a['iqr'], stats_a['q75'] + 1.5 * stats_a['iqr']]
                range_b = [stats_b['q25'] - 1.5 * stats_b['iqr'], stats_b['q75'] + 1.5 * stats_b['iqr']]
                
                overlap_start = max(int(round(range_a[0], 0 )), int(round(range_b[0], 0 )))
                overlap_end = min(int(round(range_a[1], 0 )), int(round(range_b[1], 0 )))
                
                if overlap_end > overlap_start:
                    overlap_length = overlap_end - overlap_start
                    range_a_length = range_a[1] - range_a[0]
                    
                    # Add to overlap score - higher means more similar to other clusters
                    overlap_scores[cluster_a] += overlap_length / range_a_length
        
        # Calculate isolation score - higher means more isolated
        n_clusters = len(cluster_info['unique_clusters'])
        for cluster in overlap_scores:
            if n_clusters > 1:  # Avoid division by zero
                # Average overlap with other clusters (0 = completely isolated, 1 = complete overlap)
                overlap_scores[cluster] /= (n_clusters - 1)
        
        # Identify anomalies as clusters with lowest overlap (most isolated)
        # This approach uses a clearer separation criterion
        anomaly_threshold = 0.6  # Clusters with overlap score over this are considered anomalies
        anomaly_clusters = []
        
        for cluster, overlap in overlap_scores.items():
            if overlap > anomaly_threshold:
                anomaly_clusters.append(cluster)
        
        # If no anomalies found with the threshold approach, use outlier statistics
        if len(anomaly_clusters) == 0:
            # Calculate median of means and median of medians across all clusters
            median_of_means = percentile([stats['mean'] for stats in cluster_stats.values()], 50)
            median_of_medians = percentile([stats['q50'] for stats in cluster_stats.values()], 50)
            
            for cluster, stats in cluster_stats.items():
                # Check if cluster is an outlier in terms of central tendency
                if (abs(stats['mean'] - median_of_means) > 0.5 or 
                    abs(stats['q50'] - median_of_medians) > 0.5):
                    anomaly_clusters.append(cluster)
        
        if len(anomaly_clusters) > (n_clusters // 2):
            anomaly_clusters = list(set(arange(max(anomaly_clusters))) - set(anomaly_clusters))

        # If still no anomalies found, fall back to using the smallest cluster
        if len(anomaly_clusters) == 0:
            min_size_cluster = min(cluster_stats, key=lambda x: cluster_stats[x]['n_samples'])
            anomaly_clusters = [min_size_cluster]
        
        return anomaly_clusters

    def extract_cluster_info(self, X: ndarray, decoder, anomalia_clusters: Optional[List[int]] = None) -> Dict[str, Any]:
        """Extract cluster information from data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        anomalia_clusters : Optional[List[int]]
            List of cluster IDs to consider as anomalies
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with cluster information
        """
        latent_features = self.model.transform(X)
        cluster_labels = self.model.predict_clusters(X)
        cluster_info = self.base_cluster_info(X)

        if anomalia_clusters is None:
            if len(X.shape) == 3:
                seq_len = X.shape[1]
                # Use distribution-based method if seq_len is provided
                anomalia_clusters = self.identify_anomaly_clusters_by_distribution(decoder, cluster_info, seq_len)
            else:
                # Fall back to size-based method if seq_len is not provided
                min_anomalia_ratio = 1.0
                selected_cluster = max(cluster_labels)
                for cluster_id in unique(cluster_labels):
                    cluster_size = sum(cluster_labels == cluster_id)
                    if cluster_size / len(X) < min_anomalia_ratio:
                        min_anomalia_ratio = cluster_size / len(X)
                        selected_cluster = cluster_id
                anomalia_clusters = [selected_cluster]
        
        anomalia_mask = isin(cluster_labels, anomalia_clusters)
        normal_mask = ~anomalia_mask
        
        return {
            'latent_features': latent_features,
            'cluster_labels': cluster_labels,
            'anomalia_clusters': anomalia_clusters,
            'anomalia_mask': anomalia_mask,
            'normal_mask': normal_mask,
            'unique_clusters': unique(cluster_labels)
        }
    
    def visualize_cluster_distributions(self, decoder, cluster_info: Dict[str, Any], seq_len: int):
        """Visualize the statistical distributions of clusters in latent space.
        
        Parameters
        ----------
        decoder : tf.keras.Model
            The decoder part of the CAEVizT model used to project latent features back to original space
        cluster_info : Dict[str, Any]
            Dictionary containing cluster information
        seq_len : int
            Length of sequences used in the model
        """        
        # Create color palette
        palette = self.get_color_palette(len(cluster_info['unique_clusters']))
        
        # Generate and project samples for visualization
        fig, axs = subplots(2, 2, figsize=(10, 7), sharex=True)
        methods = ['smote', 'gradient', 'boundary', 'combined']
        
        for i, method in enumerate(methods):
            clusters_samples = SampleGenerator.flood_latent_space(cluster_info, n_samples=5000, method=method)
            projs = []
            
            for cluster in cluster_info['unique_clusters']:
                projs.append(unmake_sequences(
                    decoder.predict(clusters_samples[cluster], verbose=0), 
                    seq_len=seq_len
                ).reshape(-1,))
                
            boxplot(data=projs, ax=axs[i%2, i//2], palette=palette)
            axs[i%2, i//2].set_title(f'Statistics samples ({method.upper()}) projected from latent space')
            axs[i%2, i//2].set_ylim(-1.2, 1.2)
            
            # Mark anomaly clusters
            for j, cluster in enumerate(cluster_info['unique_clusters']):
                if cluster in cluster_info['anomalia_clusters']:
                    ylim_lower, ylim_upper = axs[i%2, i//2].get_ylim()
                    axs[i%2, i//2].axvline(x=cluster, ymin=ylim_lower*0.7, ymax=ylim_upper*0.7, color='red', linestyle=':', linewidth=2, label='Anomaly Cluster')
        tight_layout()
        show()

    @staticmethod
    def get_color_palette(n_clusters: int) -> Dict[int, Tuple[float, float, float]]:
        """Create a consistent color palette for clusters.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters to generate colors for
            
        Returns
        -------
        Dict[int, Tuple[float, float, float]]
            Mapping of cluster IDs to colors
        """
        import seaborn as sns
        
        if n_clusters <= 10:
            # Use qualitative colormap for fewer clusters
            colors = sns.color_palette("tab10", n_clusters)
        else:
            # Use a larger colormap for many clusters
            colors = sns.color_palette("husl", n_clusters)
        
        unique_clusters = arange(n_clusters)
        return {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}


def analyse_cluster_sequences(X:ndarray, X_seq:ndarray, model, seq_len:int, n_estimators:int=10):
    """Create a classifier that predicts cluster based on time window features.
    From a window_size sequence of cluster assignments, predict the cluster of the next point.

    Parameters
    ----------
    X : ndarray
        The original data points.
    X_seq : ndarray
        The sequenced data points (from X).
    model : CAEVizT
        The trained CAEVizT model used for clustering.
    seq_len : int
        The length of the sequences used in the model.
    window : int, optional
        The size of the window to use for feature extraction, by default 5.
    """
    window_size = 5
    X_features = []

    for i in range(len(X) - window_size):
        features = X[i:i+window_size, 0]  # Use the first feature
        X_features.append(features)

    point_labels = sequence_clusters_to_point_clusters(X, X_seq, model, seq_len)
    X_features = array(X_features)
    y_labels = point_labels[window_size:]  # Corresponding labels

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_features, y_labels)
    predictions = rf.predict(X_features)

    return point_labels, predictions

def plot_random_forest_results(X:ndarray, point_labels:ndarray, predictions:ndarray, window_size:int=5):
    """Plot the results of the random forest classifier.
    Parameters
    ----------
    X : ndarray
        The original data points.
    point_labels : ndarray
        The cluster labels assigned to the original data points.
    predictions : ndarray
        The cluster predictions made by the random forest classifier.
    window_size : int, optional
        The size of the window used for feature extraction, by default 5.
    """
    # Create a figure with 4 subplots
    fig, axs = subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Original Data with Cluster Assignments
    for cluster_id in unique(point_labels):
        mask = point_labels == cluster_id
        axs[0, 0].scatter(arange(len(X))[mask], X[mask], 
                label=f'Cluster {cluster_id}', alpha=0.7)
    axs[0, 0].set_title('Original Data with Cluster Assignments')
    axs[0, 0].legend()

    # Plot 2: Data with Cluster Predictions from Random Forest
    for cluster_id in unique(predictions):
        mask = predictions == cluster_id
        indices = arange(window_size, len(X))[mask]
        axs[0, 1].scatter(indices, X[indices], 
                label=f'Predicted Cluster {cluster_id}', alpha=0.7)
    axs[0, 1].set_title('Data with Cluster Predictions from Random Forest')
    axs[0, 1].legend()

    # Plot 3: Confusion Matrix
    y_true = point_labels[window_size:]
    conf_mat = confusion_matrix(y_true, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap=cm.Blues, ax=axs[1, 0])
    axs[1, 0].set_title("Confusion Matrix")
    
    # Plot 4: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    axs[1, 1].plot([0, 1], [0, 1], color='red', linestyle='--')
    axs[1, 1].set_xlim([0.0, 1.0])
    axs[1, 1].set_ylim([0.0, 1.05])
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_title('ROC Curve')
    axs[1, 1].legend(loc='lower right')
    
    tight_layout()
    show()
    
    # Print classification report
    print(classification_report(y_true, predictions))