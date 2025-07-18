import tensorflow as tf
import numpy as np
from pandas import Series, DataFrame
from .cae_loss import ClusteringLoss, SpectrumLoss, get_cluster_probabilities

class CAEVizTAnalyzer:
    """Responsible for clustering analysis and statistical tools."""
    
    def __init__(self, base):
        """Initialize with reference to the base class instance."""
        self.base = base
        
    @staticmethod
    def compute_stat_spectrum(X, window=10):
        """
        Compute rolling mean ± std for each feature (column) in X.
        
        Parameters
        ----------
        X : 1D array-like
            Input data
        window : int, default=10
            Window size for rolling statistics
        
        Returns
        -------
        in_spectrum : array-like of bool
            Mask where each value is within [mean-std, mean+std]
        out_spectrum : array-like of bool
            Mask where each value is outside [mean-std, mean+std]
        """
        if isinstance(X, tf.Tensor):
            X = X.numpy()

        if len(X.shape) == 2 and X.shape[1] == 1:
            X = X.flatten()
            
        serie = Series(X)
        roll = serie.rolling(window=window, min_periods=1, center=True)
        mean = roll.mean()
        std = roll.std()
        up = (mean + std).values
        low = (mean - std).values
        in_spectrum = (serie.values >= low) & (serie.values <= up)
        out_spectrum = ~in_spectrum
        return np.array(in_spectrum), np.array(out_spectrum)

    @staticmethod
    def calculate_cluster_statistics(X, cluster_labels, feature_spectrums):
        """
        Calculate statistics for each cluster based on the statistical spectrum.
        
        Parameters
        ----------
        X : array-like
            Input data
        cluster_labels : array-like
            Cluster assignments
        feature_spectrums : list of tuples
            List of (in_spectrum, out_spectrum) tuples for each feature
        
        Returns
        -------
        cluster_stats : dict
            Dictionary with cluster statistics
        """
        cluster_stats = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            
            # For each cluster, calculate what percentage of points are in-spectrum vs out-spectrum
            in_count = 0
            out_count = 0
            
            for in_spectrum, out_spectrum in feature_spectrums:
                cluster_in = np.sum(in_spectrum & cluster_mask)
                cluster_out = np.sum(out_spectrum & cluster_mask)
                in_count += cluster_in
                out_count += cluster_out
            
            total = in_count + out_count
            
            cluster_stats[cluster_id] = {
                'in_count': in_count,
                'out_count': out_count,
                'in_percent': (in_count / total * 100) if total > 0 else 0,
                'out_percent': (out_count / total * 100) if total > 0 else 0,
            }
        
        return cluster_stats
    
    def get_cluster_probabilities(self, latent_batch):
        """
        Get soft cluster probabilities for a batch of latent representations.
        
        Parameters
        ----------
        latent_batch : tensor
            Latent space representations
        
        Returns
        -------
        q : tensor
            Soft cluster assignment probabilities
        """
        # If we're not using a clustering layer, calculate probabilities directly
        if not hasattr(self.base, 'current_cluster_centers_') or self.base.current_cluster_centers_ is None:
            return tf.ones((tf.shape(latent_batch)[0], self.base.n_clusters)) / self.base.n_clusters
        
        # Use the helper function from cae_loss
        return get_cluster_probabilities(latent_batch, self.base.current_cluster_centers_)
    
    def get_clustering_loss(self, latent_batch, q):
        """
        Calculate clustering loss based on KL divergence.
        
        Parameters
        ----------
        latent_batch : tensor
            Latent space representations
        q : tensor
            Soft cluster assignments
        
        Returns
        -------
        kl_loss : scalar tensor
            Clustering loss
        """
        # Use the ClusteringLoss from cae_loss
        cluster_loss = ClusteringLoss()
        return cluster_loss(q, latent_batch)
        
    def compute_spectrum_loss(self, x_batch, cluster_probs, feature_spectrums):
        """
        Compute loss that encourages points with similar statistical properties
        to be assigned to the same clusters.
        
        Parameters
        ----------
        x_batch : tensor
            Input batch data
        cluster_probs : tensor
            Soft cluster assignments for the batch
        feature_spectrums : dict
            Dictionary of (feature_idx, cluster_idx) -> (min_val, max_val)
        
        Returns
        -------
        loss : scalar tensor
            Loss value encouraging statistical homogeneity in clusters
        """
        # Create a SpectrumLoss instance with the feature spectrums
        spectrum_loss = SpectrumLoss(feature_spectrums)
        
        # Use the SpectrumLoss from cae_loss
        return spectrum_loss([x_batch, cluster_probs], None)

    def classify_clusters(self, X, stats_window=10, epsilon=5.0)->np.ndarray:
        """
        Classify clusters based on their statistical properties.
        
        Returns
        --------
        anomaly_clusters : ndarray
            List of cluster IDs that are considered anomalies based on statistical properties.
        
        Parameters
        -------
        X : array-like
            Input data used for clustering
        stats_window : int, default=10
            Window size for rolling statistics
        epsilon : float, default=5.0%
            Epsilon value (percent) set for tolerance in change of leading factor in out spectrum clusters
        """
        # Get feature spectrums as list for backward compatibility with calculate_cluster_statistics
        feature_spectrums = []
        X_numpy = X.numpy() if isinstance(X, tf.Tensor) else X
        
        # Handle reshaping if needed
        if len(X_numpy.shape) == 2 and hasattr(self.base, 'seq_len') and hasattr(self.base, 'n_features'):
            # Reshape if X is flattened
            X_reshaped = X_numpy.reshape(X_numpy.shape[0], self.base.seq_len, self.base.n_features)

            for i in range(self.base.n_features):
                feature_data = X_reshaped[:, -1, i].flatten()
                in_spectrum, out_spectrum = self.compute_stat_spectrum(feature_data, window=stats_window)
                feature_spectrums.append((in_spectrum, out_spectrum))
        else:
            # If X is already correctly shaped
            for i in range(X_numpy.shape[1]):
                in_spectrum, out_spectrum = self.compute_stat_spectrum(X_numpy[:, i].flatten(), window=stats_window)
                feature_spectrums.append((in_spectrum, out_spectrum))
                
        clusters_stats = self.calculate_cluster_statistics(X, self.base.labels_, feature_spectrums)

        cluster_stats_df = DataFrame.from_dict(clusters_stats, orient='index')
        if cluster_stats_df.empty:
            raise ValueError("No clusters found in the data. Ensure that the clustering model has been trained and labels are assigned.")

        cluster_stats_df.index.name = "cluster_id"

        cluster_stats_df.sort_values(by='out_percent', ascending=False, inplace=True)

        highest_out_percent = cluster_stats_df['out_percent'].head(1).values[0]
        cluster_stats_df['diff_percent'] = highest_out_percent - cluster_stats_df['out_percent']
        cluster_stats_df['leading_factor'] = cluster_stats_df['diff_percent'].diff().fillna(0.0)
        cluster_stats_df.reset_index(inplace=True)

        # Find the first index where leading_factor exceeds epsilon
        cutoff_idx = cluster_stats_df.index[cluster_stats_df['leading_factor'] > epsilon]
        if len(cutoff_idx) > 0:
            # Get the position of the first occurrence
            cutoff_pos = cluster_stats_df.index.get_indexer([cutoff_idx[0]])[0]
            # Keep only rows before the cutoff
            cluster_stats_df = cluster_stats_df.iloc[:cutoff_pos]
            return cluster_stats_df.cluster_id.values
        return cluster_stats_df.cluster_id.values[:-1]