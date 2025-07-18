import tensorflow as tf
from keras.losses import Loss


class ReconstructionLoss(Loss):
    """Standard Keras loss for reconstruction error (MSE)."""
    
    def __init__(self, name="reconstruction_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        """Calculate MSE reconstruction loss."""
        # Convert inputs to float32 tensors
        y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
        y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
        
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

class ClusteringLoss(Loss):
    """Standard Keras loss for clustering using KL divergence."""
    
    def __init__(self, name="clustering_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        """Calculate KL divergence clustering loss."""
        # Convert inputs to float32 tensors
        q = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
        
        # Calculate p-distribution
        weight = tf.pow(q, 2) / tf.reduce_sum(q, axis=0)
        p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))
        
        # KL divergence
        kl_loss = tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p / q + tf.keras.backend.epsilon()), axis=1))
        return kl_loss

class SpectrumLoss(Loss):
    """Standard Keras loss for statistical spectrum analysis."""
    
    def __init__(self, feature_spectrums=None, name="spectrum_loss", **kwargs):
        """Initialize with feature spectrums."""
        super().__init__(name=name, **kwargs)
        self.feature_spectrums = feature_spectrums or {}
    
    def call(self, y_true, y_pred):
        """Calculate statistical spectrum loss.
        
        Parameters:
        ----------
        y_true : list
            List containing two elements: x_batch (features) and q (cluster probabilities).
        """
        if isinstance(y_true, list) and len(y_true) >= 2:
            # Convert inputs to float32 tensors
            x_batch = tf.cast(tf.convert_to_tensor(y_true[0]), tf.float32)
            q = tf.cast(tf.convert_to_tensor(y_true[1]), tf.float32)
        else:
            # If we don't have proper inputs, return zero loss
            return tf.constant(0.0, dtype=tf.float32)
        
        if not self.feature_spectrums:
            return tf.constant(0.0, dtype=tf.float32)

        batch_size = tf.shape(x_batch)[0]
        n_features = tf.shape(x_batch)[1]
        n_clusters = tf.shape(q)[1]
        
        # Initialize loss
        total_loss = tf.constant(0.0, dtype=tf.float32)
        
        # Convert n_features and n_clusters to Python integers
        n_features_int = int(n_features.numpy())
        n_clusters_int = int(n_clusters.numpy())
        
        for i in range(n_features_int):
            feature_vals = x_batch[:, i]
            
            # For each cluster
            for j in range(n_clusters_int):
                # Get spectrum for this feature-cluster combination
                key = (i, j)
                if any(key == k for k in self.feature_spectrums.keys()):
                    min_val = tf.cast(self.feature_spectrums[key][0], tf.float32)
                    max_val = tf.cast(self.feature_spectrums[key][1], tf.float32)
                    
                    # Check if values are within spectrum
                    within_spectrum = tf.logical_and(
                        tf.greater_equal(feature_vals, min_val),
                        tf.less_equal(feature_vals, max_val)
                    )
                    within_spectrum = tf.cast(within_spectrum, tf.float32)
                    
                    # Weight by cluster probability
                    weighted_loss = (1.0 - within_spectrum) * q[:, j]
                    total_loss += tf.reduce_sum(weighted_loss)
        
        # Normalize by batch size
        return total_loss / tf.cast(batch_size, tf.float32)

class CombinedLoss(Loss):
    """Standard Keras loss that combines reconstruction, clustering and spectrum losses 
    with phase-specific weights."""
    
    def __init__(self, feature_spectrums=None, pretrain_alpha=0.0, train_alpha=0.5, stats_weight=0.1, 
                 name="combined_loss", **kwargs):
        """Initialize with weights and feature spectrums."""
        super().__init__(name=name, **kwargs)
        self.feature_spectrums = feature_spectrums or {}
        self.pretrain_alpha = float(pretrain_alpha)
        self.train_alpha = float(train_alpha)
        self.stats_weight = float(stats_weight)
        self.current_phase = "pretrain"  # Default to pretraining phase
        
        self.reconstruction_loss = ReconstructionLoss()
        self.clustering_loss = ClusteringLoss()
        self.spectrum_loss = SpectrumLoss(feature_spectrums)
        
        # Initialize latent and q as None
        self.latent = None
        self.q = None
    
    def call(self, y_true, y_pred):
        """Calculate combined loss based on the current phase."""
        # Convert inputs to float32 tensors
        y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
        y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
        
        # Always calculate reconstruction loss
        rec_loss = self.reconstruction_loss(y_true, y_pred)
        
        # For pretraining, only use reconstruction loss
        if self.current_phase == "pretrain":
            return rec_loss
        
        # For fine-tuning and spectrum phases, check if we have latent and q
        if self.latent is not None and self.q is not None:
            # Calculate clustering loss if in fine-tuning or spectrum phase
            clust_loss = self.clustering_loss(self.q, self.latent)
            
            # In finetune phase, use reconstruction + clustering loss
            if self.current_phase == "finetune":
                return rec_loss + self.train_alpha * clust_loss
            
            # In spectrum phase, add spectrum loss as well
            elif self.current_phase == "spectrum":
                stats_loss = self.spectrum_loss([y_true, self.q], y_pred)
                return rec_loss + self.train_alpha * clust_loss + self.stats_weight * stats_loss
        
        # Default: return reconstruction loss only
        return rec_loss
    
    def set_phase(self, phase):
        """Set the current training phase."""
        self.current_phase = phase
    
    def update_latent_and_q(self, latent, q):
        """Update the latent representations and cluster probabilities."""
        self.latent = tf.cast(tf.convert_to_tensor(latent), tf.float32)
        self.q = tf.cast(tf.convert_to_tensor(q), tf.float32)
    
    def update_feature_spectrums(self, feature_spectrums):
        """Update the feature spectrums."""
        self.feature_spectrums = feature_spectrums
        self.spectrum_loss.feature_spectrums = feature_spectrums


# Helper function for cluster probabilities (not a loss function)
def get_cluster_probabilities(latent, cluster_centers, alpha=1.0):
    """Calculate soft cluster assignment probabilities (q)."""
    # Convert to tensors if they are not already
    latent = tf.cast(tf.convert_to_tensor(latent), tf.float32)
    cluster_centers = tf.cast(tf.convert_to_tensor(cluster_centers), tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    
    # Calculate squared distances
    q = 1.0 / (1.0 + tf.reduce_sum(
        tf.square(tf.expand_dims(latent, axis=1) - cluster_centers), axis=2) / alpha)
    
    # Power by (alpha + 1) / 2 and normalize
    q = tf.pow(q, (alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    
    return q