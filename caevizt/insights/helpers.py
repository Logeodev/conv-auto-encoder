from numpy import zeros, unique, sum
import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from typing import Dict, Any


class SampleGenerator:
    """Generates synthetic samples from latent space clusters."""
    
    @staticmethod
    def generate_smote_samples(points: np.ndarray, n_samples: int = 1000, 
                             k_neighbors: int = 5) -> np.ndarray:
        """Generate synthetic samples using SMOTE-like approach.
        
        Parameters
        ----------
        points : np.ndarray
            Original data points
        n_samples : int
            Number of samples to generate
        k_neighbors : int
            Number of nearest neighbors to consider
            
        Returns
        -------
        np.ndarray
            Synthetic samples
        """
        if len(points) <= k_neighbors + 1:
            return points  # Not enough points for SMOTE
        
        # Find k nearest neighbors for each point
        nn = NearestNeighbors(n_neighbors=k_neighbors+1).fit(points)
        distances, indices = nn.kneighbors(points)
        
        # Generate synthetic samples
        synthetic_samples = []
        
        # Number of samples to generate per original point
        samples_per_point = max(1, int(n_samples / len(points)))
        
        for i in range(len(points)):
            # For each point, get its neighbors
            neighbors = indices[i, 1:]  # Exclude the point itself
            
            # Generate samples for this point
            for _ in range(samples_per_point):
                # Randomly select a neighbor
                neighbor_idx = neighbors[np.random.randint(0, len(neighbors))]
                
                # Get the original point and its selected neighbor
                orig_point = points[i]
                neighbor_point = points[neighbor_idx]
                
                # Generate a random value between 0 and 1
                random_value = np.random.random()
                
                # Create synthetic sample
                synthetic_sample = orig_point + random_value * (neighbor_point - orig_point)
                synthetic_samples.append(synthetic_sample)
                
        return np.array(synthetic_samples)
    
    @staticmethod
    def generate_density_gradient_samples(points: np.ndarray, n_samples: int = 1000, 
                                       bandwidth: float = 0.5) -> np.ndarray:
        """Generate samples with higher density near boundaries and sparser in cluster centers.
        
        Parameters
        ----------
        points : np.ndarray
            Original data points
        n_samples : int
            Number of samples to generate
        bandwidth : float
            Bandwidth parameter for kernel density estimation
            
        Returns
        -------
        np.ndarray
            Generated samples with varying density
        """
        if len(points) <= 10:
            return points  # Not enough points
        
        # Estimate density of original points
        kde = KernelDensity(bandwidth=bandwidth).fit(points)
        log_density = kde.score_samples(points)
        density = np.exp(log_density)
        
        # Convert density to inverse sampling weights (want more samples in lower density regions)
        inverse_density = 1.0 / (density + 1e-6)  # Avoid division by zero
        
        # Normalize weights
        weights = inverse_density / inverse_density.sum()
        
        # Sample points with replacement according to weights
        indices = np.random.choice(len(points), size=n_samples, p=weights)
        base_points = points[indices]
        
        # Add small noise proportional to local density (less noise in dense regions)
        noise_scale = 0.05  # Base noise scale
        
        # Scale the noise by the normalized density
        scaler = MinMaxScaler()
        normalized_density = scaler.fit_transform(density.reshape(-1, 1)).ravel()
        
        # Apply different noise levels to each selected point
        point_densities = normalized_density[indices]
        
        # Calculate noise for each point (inverse relationship with density)
        noise_factors = noise_scale * (1.0 - point_densities.reshape(-1, 1))
        noise = np.random.normal(0, 1, base_points.shape) * noise_factors
        
        # Generate samples by adding noise to base points
        gradient_samples = base_points + noise
        
        return gradient_samples
    
    @staticmethod
    def generate_boundary_samples(cluster_points_by_id: Dict[int, np.ndarray], 
                               n_samples: int = 1000) -> Dict[int, np.ndarray]:
        """Generate samples focused on the boundaries between clusters.
        
        Parameters
        ----------
        cluster_points_by_id : Dict[int, np.ndarray]
            Dictionary with cluster IDs as keys and corresponding points as values
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary with cluster IDs as keys and boundary samples as values
        """
        if len(cluster_points_by_id) <= 1:
            return cluster_points_by_id  # No boundaries with only one cluster
        
        boundary_samples = {}
        
        # Identify points near cluster boundaries
        for cluster_id, points in cluster_points_by_id.items():
            if len(points) <= 5:  # Skip clusters with too few points
                boundary_samples[cluster_id] = points
                continue
            
            # Collect all other clusters' points
            other_clusters_points = []
            for other_id, other_points in cluster_points_by_id.items():
                if other_id != cluster_id and len(other_points) > 0:
                    other_clusters_points.append(other_points)
                    
            if not other_clusters_points:
                boundary_samples[cluster_id] = points
                continue
                
            other_points = np.vstack(other_clusters_points)
            
            # For each point in current cluster, find distances to nearest other-cluster point
            nn = NearestNeighbors(n_neighbors=1).fit(other_points)
            distances, _ = nn.kneighbors(points)
            
            # Sort points by distance to boundary (closest first)
            boundary_indices = np.argsort(distances.ravel())
            
            # Take the closest x% of points
            boundary_percentage = 0.3  # Focus on closest 30% of points
            n_boundary_points = max(5, int(len(points) * boundary_percentage))
            boundary_points = points[boundary_indices[:n_boundary_points]]
            
            # Generate more samples around boundary points
            boundary_samples[cluster_id] = SampleGenerator.generate_smote_samples(
                boundary_points, 
                n_samples=max(100, int(n_samples / len(cluster_points_by_id))),
                k_neighbors=min(5, len(boundary_points)-1)
            )
            
        return boundary_samples
    
    @classmethod
    def generate_density_samples(cls, cluster_info: Dict[str, Any], 
                              n_samples: int = 1000, 
                              bandwidth: float = 0.2) -> Dict[int, np.ndarray]:
        """Generate KDE-based samples for each cluster.
        
        Parameters
        ----------
        cluster_info : Dict[str, Any]
            Dictionary with cluster data
        n_samples : int
            Number of samples to generate
        bandwidth : float
            Bandwidth for KDE
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping cluster IDs to samples
        """
        latent_features = cluster_info['latent_features']
        cluster_labels = cluster_info['cluster_labels']
        unique_clusters = cluster_info['unique_clusters']
        
        cluster_samples = {}
        samples_per_cluster = max(100, n_samples // len(unique_clusters))
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_points = latent_features[cluster_mask]
            n_cluster_points = len(cluster_points)
            
            # Handle differently based on number of points
            if n_cluster_points <= 2:
                # For extremely small clusters (1-2 points), create synthetic points by adding noise
                print(f"Cluster {cluster_id} has very few points ({n_cluster_points}). Using noise augmentation.")
                # Generate samples by adding noise to existing points
                repeated_points = np.repeat(cluster_points, samples_per_cluster // n_cluster_points + 1, axis=0)
                repeated_points = repeated_points[:samples_per_cluster]  # Cap at desired sample count
                
                # Add small random noise
                noise_scale = 0.05
                noise = np.random.normal(0, noise_scale, size=repeated_points.shape)
                cluster_samples[cluster_id] = repeated_points + noise
                
            elif n_cluster_points <= 5:
                # For small clusters (3-5 points), use simple SMOTE
                print(f"Cluster {cluster_id} has few points ({n_cluster_points}). Using SMOTE-like sampling.")
                k_neighbors = min(2, n_cluster_points-1)  # Use at most 2 neighbors for very small clusters
                cluster_samples[cluster_id] = cls.generate_smote_samples(
                    cluster_points, 
                    n_samples=samples_per_cluster,
                    k_neighbors=k_neighbors
                )
                
            elif n_cluster_points <= 10:
                # For medium-small clusters (6-10 points), use density gradient sampling
                print(f"Cluster {cluster_id} has {n_cluster_points} points. Using density gradient sampling.")
                cluster_samples[cluster_id] = cls.generate_density_gradient_samples(
                    cluster_points,
                    n_samples=samples_per_cluster,
                    bandwidth=bandwidth * 1.5  # Increase bandwidth for more smoothing
                )
                
            else:
                # For normal-sized clusters, use KDE
                latent_dim = cluster_points.shape[1]
                
                if n_cluster_points <= latent_dim:
                    # Apply PCA to avoid singular matrix
                    pca_reducer = PCA(n_components=max(1, n_cluster_points-2))
                    reduced_points = pca_reducer.fit_transform(cluster_points)
                    
                    # Fit KDE
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reduced_points)
                    
                    # Sample
                    samples_reduced = kde.sample(samples_per_cluster)
                    
                    # Project back to full space
                    cluster_samples[cluster_id] = pca_reducer.inverse_transform(samples_reduced)
                else:
                    # Use full-dimensional KDE
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(cluster_points)
                    cluster_samples[cluster_id] = kde.sample(samples_per_cluster)
        
        return cluster_samples
    
    @classmethod
    def flood_latent_space(cls, cluster_info: Dict[str, Any], n_samples: int = 5000, 
                        method: str = 'combined') -> Dict[int, np.ndarray]:
        """Apply latent space flooding to enhance cluster boundary visualization.
        
        Parameters
        ----------
        cluster_info : Dict[str, Any]
            Dictionary with cluster data
        n_samples : int
            Total number of synthetic samples to generate
        method : str
            Method for generating synthetic samples: 'smote', 'boundary', 'gradient', or 'combined'
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary with cluster IDs as keys and enhanced samples as values
        """
        latent_features = cluster_info['latent_features']
        cluster_labels = cluster_info['cluster_labels']
        unique_clusters = cluster_info['unique_clusters']
        
        # Organize points by cluster
        cluster_points = {}
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_points[cluster_id] = latent_features[mask]
        
        # Generate samples based on selected method
        enhanced_samples = {}
        
        if method == 'smote':
            # Simple SMOTE-like upsampling for each cluster
            for cluster_id, points in cluster_points.items():
                samples_per_cluster = max(100, int(n_samples / len(unique_clusters)))
                enhanced_samples[cluster_id] = cls.generate_smote_samples(
                    points, 
                    n_samples=samples_per_cluster
                )
                
        elif method == 'boundary':
            # Focus on boundary regions
            enhanced_samples = cls.generate_boundary_samples(
                cluster_points, 
                n_samples=n_samples
            )
            
        elif method == 'gradient':
            # Density gradient approach
            for cluster_id, points in cluster_points.items():
                samples_per_cluster = max(100, int(n_samples / len(unique_clusters)))
                enhanced_samples[cluster_id] = cls.generate_density_gradient_samples(
                    points, 
                    n_samples=samples_per_cluster
                )
                
        elif method == 'combined':
            # Combine multiple approaches
            for cluster_id, points in cluster_points.items():
                samples_per_cluster = max(100, int(n_samples / len(unique_clusters)))
                
                # Generate samples using different methods
                smote_samples = cls.generate_smote_samples(
                    points, 
                    n_samples=int(samples_per_cluster * 0.4)
                )
                
                gradient_samples = cls.generate_density_gradient_samples(
                    points, 
                    n_samples=int(samples_per_cluster * 0.4)
                )
                
                # Combine the sample sets
                combined_samples = np.vstack([
                    points,  # Include original points
                    smote_samples,
                    gradient_samples
                ])
                
                enhanced_samples[cluster_id] = combined_samples
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return enhanced_samples

def extract_cluster_transition_features(point_labels, window_size=10):
    """
    Extract features based on cluster transitions within windows.
    Gives sequence transition features to use with sklearn models to identify patterns in how the clusters evolve over time.
    
    Parameters:
    -----------
    point_labels : numpy.ndarray
        Cluster labels for each time point
    window_size : int
        Size of the window to consider for transitions
    
    Returns:
    --------
    features : numpy.ndarray
        Features describing cluster transitions
    """
    n_clusters = len(unique(point_labels))
    n_points = len(point_labels)
    
    # Create feature matrix
    n_windows = n_points - window_size + 1
    features = zeros((n_windows, n_clusters + n_clusters**2))
    
    for i in range(n_windows):
        window = point_labels[i:i+window_size]
        
        # Feature 1: Frequency of each cluster in the window
        for cluster in range(n_clusters):
            features[i, cluster] = sum(window == cluster) / window_size
        
        # Feature 2: Frequency of transitions between clusters
        for j in range(window_size-1):
            if window[j] != window[j+1]:  # Transition detected
                from_cluster = window[j]
                to_cluster = window[j+1]
                transition_idx = n_clusters + from_cluster * n_clusters + to_cluster
                features[i, transition_idx] += 1 / (window_size - 1)
    
    return features
