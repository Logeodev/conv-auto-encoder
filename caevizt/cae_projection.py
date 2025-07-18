import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
from traceback import print_exc

from insights.clusters import ClusterInfoExtractor
from insights.helpers import SampleGenerator

class LatentSpaceVisualizer:
    """Visualizes latent space projections with various techniques."""
    
    @staticmethod
    def plot_latent_density(ax: plt.Axes, cluster_info: Dict[str, Any], 
                          cluster_samples: Dict[int, np.ndarray], 
                          color_map: Dict[int, Tuple[float, float, float]], 
                          bandwidth: float = 0.2) -> None:
        """Visualize latent space with density contours.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to plot on
        cluster_info : Dict[str, Any]
            Dictionary with cluster data
        cluster_samples : Dict[int, np.ndarray]
            Dictionary mapping cluster IDs to samples
        color_map : Dict[int, Tuple[float, float, float]]
            Mapping from cluster IDs to colors
        bandwidth : float
            Bandwidth for KDE visualization
        """
        latent_features = cluster_info['latent_features']
        cluster_labels = cluster_info['cluster_labels']
        unique_clusters = cluster_info['unique_clusters']
        anomalia_clusters = cluster_info['anomalia_clusters']
        
        # Use PCA for 2D visualization
        pca_viz = PCA(n_components=2)
        latent_2d = pca_viz.fit_transform(latent_features)
        
        # Create grid for contour plots
        x_min, x_max = latent_2d[:, 0].min() - 1, latent_2d[:, 0].max() + 1
        y_min, y_max = latent_2d[:, 1].min() - 1, latent_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # For each cluster, plot density contours
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_points_2d = latent_2d[cluster_mask]
            is_anomaly = cluster_id in anomalia_clusters
            
            if len(cluster_points_2d) > 10:
                # Create KDE for 2D visualization
                kde_2d = KernelDensity(bandwidth=bandwidth).fit(cluster_points_2d)
                
                # Evaluate density on grid
                density = np.exp(kde_2d.score_samples(grid_points))
                density = density.reshape(xx.shape)
                
                # Plot contours
                levels = np.linspace(0, density.max(), 10)[1:]  # Skip 0 level
                ax.contour(
                    xx, yy, density,
                    levels=levels,
                    colors=[color_map[cluster_id]],
                    alpha=0.7,
                    linewidths=2
                )
                
                # Fill contours with lower alpha
                ax.contourf(
                    xx, yy, density,
                    levels=levels,
                    colors=[color_map[cluster_id]],
                    alpha=0.2
                )
            
            # Plot original points
            marker = 'x' if is_anomaly else 'o'
            marker_size = 40 if is_anomaly else 30
            
            ax.scatter(
                cluster_points_2d[:, 0],
                cluster_points_2d[:, 1],
                color=color_map[cluster_id],
                alpha=0.7,
                marker=marker,
                s=marker_size,
                label=f"Cluster {cluster_id}" + (" (Anomaly)" if is_anomaly else "")
            )
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.legend(loc='best')
    

class ProjectionVisualizer:
    """Visualizes projections from latent to original space."""
    
    @staticmethod
    def plot_original_projection(ax: plt.Axes, X: np.ndarray, 
                              cluster_samples: Dict[int, np.ndarray], 
                              cluster_info: Dict[str, Any], 
                              color_map: Dict[int, Tuple[float, float, float]], 
                              decoder: tf.keras.Model,
                              confidence_interval: Tuple[int, int] = (5, 95)) -> None:
        """Project samples to original space and visualize.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to plot on
        X : np.ndarray
            Original input data
        cluster_samples : Dict[int, np.ndarray]
            Dictionary mapping cluster IDs to latent samples
        cluster_info : Dict[str, Any]
            Dictionary with cluster data
        color_map : Dict[int, Tuple[float, float, float]]
            Mapping from cluster IDs to colors
        decoder : tf.keras.Model
            Model that maps from latent to original space
        confidence_interval : Tuple[int, int]
            Lower and upper percentiles for confidence interval
        """
        anomalia_clusters = cluster_info['anomalia_clusters']
        anomalia_mask = cluster_info['anomalia_mask']
        
        # Plot original data as background
        ax.plot(X[-1, :, 0], alpha=0.2, color='gray', label='All Data')
        
        # Smooth the curves
        def smooth_curve(curve, window_size=5):
            return pd.Series(curve.flatten())\
                .rolling(
                    window=window_size,
                    min_periods=1,
                    center=True
                ).mean().values
        
        # Process each cluster's samples
        for cluster_id, samples in cluster_samples.items():
            if len(samples) == 0:
                continue
                
            is_anomaly = cluster_id in anomalia_clusters
            color = color_map[cluster_id]
            
            try:
                # Project samples to original space
                reconstructed_samples = decoder.predict(samples, verbose=0)
                
                # Calculate statistics
                time_length = min(X.shape[1], reconstructed_samples.shape[1])
                low_percentile, high_percentile = confidence_interval
                
                recon_median = np.median(
                    reconstructed_samples[:, :time_length], 
                    axis=0
                    )
                recon_lower = np.percentile(
                    reconstructed_samples[:, :time_length], 
                    low_percentile, 
                    axis=0
                    )
                recon_upper = np.percentile(
                    reconstructed_samples[:, :time_length],
                    high_percentile, 
                    axis=0
                    )
                
                recon_median = smooth_curve(recon_median)
                recon_lower = smooth_curve(recon_lower)
                recon_upper = smooth_curve(recon_upper)
                
                # Style depends on cluster type
                linestyle = '--' if is_anomaly else '-'
                linewidth = 2.5 if is_anomaly else 1.5
                label_suffix = " (Anomaly)" if is_anomaly else ""
                
                # Plot median line
                ax.plot(
                    np.arange(time_length),
                    recon_median,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=f'Cluster {cluster_id}{label_suffix}'
                )
                
                # Plot confidence interval
                ax.fill_between(
                    np.arange(time_length),
                    recon_lower,
                    recon_upper,
                    color=color,
                    alpha=0.2
                )
                
            except Exception as e:
                print(f"Error projecting cluster {cluster_id}: {e}")
                print_exc()        
        # Plot anomalous points
        # if np.any(anomalia_mask):
        #     ax.scatter(
        #         np.arange(len(X))[anomalia_mask],
        #         X[anomalia_mask, 0],
        #         alpha=0.8,
        #         color='red',
        #         marker='x',
        #         s=40,
        #         label='Anomalous Points'
        #     )
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
    

class CAEVizTProjection:
    """Visualization tool for projecting latent space distributions to original space with enhanced
    boundary visualization through latent space flooding techniques."""
    
    def __init__(self, model: Any):
        """Initialize with a trained CAEVizT or CAEVizT_hnh model.
        
        Parameters
        ----------
        model : Any
            Trained CAEVizT model
        """
        self.model = model
        self.decoder = self._initialize_decoder()
        self.cluster_extractor = ClusterInfoExtractor(model)
        
    def _initialize_decoder(self) -> tf.keras.Model:
        """Extract decoder part from the model for latent space projection.
        
        Returns
        -------
        tf.keras.Model
            Decoder model
        """
        if not hasattr(self.model, 'model_') or not hasattr(self.model, 'latent_dim'):
            print(self.model.__dict__)
            raise AttributeError("Model must be trained before creating projections")
        
        latent_layer_idx = None
        latent_layer_name = "latent_space"
        
        for i, layer in enumerate(self.model.model_.layers):
            if layer.name == latent_layer_name:
                latent_layer_idx = i
                break
        
        if latent_layer_idx is None:
            raise ValueError("Could not find latent space layer in the model")
        
        latent_input = tf.keras.layers.Input(shape=(self.model.latent_dim,))
        x = latent_input
        
        for layer in self.model.model_.layers[latent_layer_idx+1:]:
            x = layer(x)
        
        return tf.keras.Model(latent_input, x)
    
    def plot_flooded_latent_space(self, X: np.ndarray, anomalia_clusters: Optional[List[int]] = None, 
                               n_samples: int = 5000, flood_method: str = 'combined', 
                               figsize: Tuple[int, int] = (18, 8)) -> Tuple[plt.Figure, Dict[int, np.ndarray]]:
        """Generate an enhanced visualization with latent space flooding.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        anomalia_clusters : Optional[List[int]]
            List of cluster IDs to consider as anomalies
        n_samples : int
            Number of samples to generate for flooding
        flood_method : str
            Method for flooding: 'smote', 'boundary', 'gradient', or 'combined'
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        Tuple[plt.Figure, Dict[int, np.ndarray]]
            Matplotlib figure and flooded samples dictionary
        """
        # Get cluster information
        cluster_info = self.cluster_extractor.extract_cluster_info(X, anomalia_clusters)
        unique_clusters = cluster_info['unique_clusters']
        
        # Create color palette
        color_map = ClusterInfoExtractor.get_color_palette(len(unique_clusters))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Generate flooded samples
        flooded_samples = SampleGenerator.flood_latent_space(
            cluster_info, 
            n_samples=n_samples,
            method=flood_method
        )
        
        # Visualize in latent space using PCA for dimension reduction
        latent_features = cluster_info['latent_features']
        cluster_labels = cluster_info['cluster_labels']
        
        # Use PCA for 2D visualization
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_features)
        
        # 1. Original latent space
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            axes[0].scatter(
                latent_2d[mask, 0],
                latent_2d[mask, 1],
                color=color_map[cluster_id],
                label=f"Cluster {cluster_id}",
                alpha=0.7,
                s=30
            )
        
        axes[0].set_title('Original Latent Space')
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')
        axes[0].legend()
        
        # 2. Flooded latent space
        for cluster_id, samples in flooded_samples.items():
            # Project to 2D using the same PCA
            samples_2d = pca.transform(samples)
            
            # Plot with lower alpha to show density
            axes[1].scatter(
                samples_2d[:, 0],
                samples_2d[:, 1],
                color=color_map[cluster_id],
                label=f"Cluster {cluster_id}",
                alpha=0.2,
                s=10
            )
            
            # Plot original points with higher alpha
            mask = cluster_labels == cluster_id
            axes[1].scatter(
                latent_2d[mask, 0],
                latent_2d[mask, 1],
                color=color_map[cluster_id],
                alpha=0.9,
                s=40,
                edgecolor='black',
                linewidth=0.5
            )
        
        axes[1].set_title(f'Flooded Latent Space ({flood_method})')
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        
        # 3. Density visualization handled by LatentSpaceVisualizer
        LatentSpaceVisualizer.plot_latent_density(axes[2], cluster_info, flooded_samples, color_map)
        
        axes[2].set_title('Density Contours with Flooding')
        
        plt.tight_layout()
        return fig, flooded_samples
    
    def plot_kde_projection(self, X: np.ndarray, anomalia_clusters: Optional[List[int]] = None, 
                         n_samples: int = 1000, figsize: Tuple[int, int] = (16, 8), 
                         visualization_type: str = 'density', bandwidth: float = 0.2, 
                         confidence_interval: Tuple[int, int] = (5, 95)) -> Tuple[plt.Figure, Dict[int, np.ndarray]]:
        """Plot KDE projection from latent to original space with multiple visualization options.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        anomalia_clusters : Optional[List[int]]
            List of cluster IDs to consider as anomalies
        n_samples : int
            Number of samples to generate from KDE
        figsize : Tuple[int, int]
            Figure size (width, height)
        visualization_type : str
            Type of visualization: 'density', 'boundary', or 'mixed'
        bandwidth : float
            Bandwidth parameter for KDE estimation
        confidence_interval : Tuple[int, int]
            Lower and upper percentiles for confidence intervals
            
        Returns
        -------
        Tuple[plt.Figure, Dict[int, np.ndarray]]
            Matplotlib figure and cluster samples dictionary
        """
        # Get cluster information
        cluster_info = self.cluster_extractor.extract_cluster_info(X, anomalia_clusters)
        unique_clusters = cluster_info['unique_clusters']
        # Create color palette
        color_map = ClusterInfoExtractor.get_color_palette(len(unique_clusters))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Generate samples based on visualization type
        if visualization_type == 'density':
            # Generate KDE-based samples for each cluster
            cluster_samples = SampleGenerator.generate_density_samples(
                cluster_info,
                n_samples=n_samples,
                bandwidth=bandwidth
            )
            # Visualize latent space with density contours
            LatentSpaceVisualizer.plot_latent_density(ax1, cluster_info, cluster_samples, color_map, bandwidth)
            
        elif visualization_type == 'boundary':
            # Organize points by cluster for boundary sample generation
            cluster_points = {}
            for cluster_id in unique_clusters:
                mask = cluster_info['cluster_labels'] == cluster_id
                cluster_points[cluster_id] = cluster_info['latent_features'][mask]
            
            # Generate boundary-focused samples
            cluster_samples = SampleGenerator.generate_boundary_samples(
                cluster_points,
                n_samples=n_samples
            )
            # Visualize latent space with boundary regions
            LatentSpaceVisualizer.plot_latent_density(ax1, cluster_info, cluster_samples, color_map, bandwidth)
            
        elif visualization_type == 'mixed':
            # Combined approach (simplified)
            cluster_samples = SampleGenerator.flood_latent_space(
                cluster_info,
                n_samples=n_samples,
                method='combined'
            )
            # Use density visualization for mixed samples as well
            LatentSpaceVisualizer.plot_latent_density(ax1, cluster_info, cluster_samples, color_map, bandwidth)
            
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        # Project to original space
        ProjectionVisualizer.plot_original_projection(
            ax2, X, cluster_samples, cluster_info, color_map, self.decoder, confidence_interval
        )
        
        # Add titles based on visualization type
        title_map = {
            'density': 'Density Contours',
            'boundary': 'Boundary Regions',
            'mixed': 'Density Contours and Boundary Regions'
        }
        
        ax1.set_title(f'Latent Space: {title_map[visualization_type]}')
        ax2.set_title(f'Original Space: Projected {title_map[visualization_type]}')
        
        plt.tight_layout()
        return fig, cluster_samples

    def plot_boundary_strength(self, X: np.ndarray, anomalia_clusters: Optional[List[int]] = None, 
                            n_samples: int = 5000, flood_method: str = 'combined', 
                            figsize: Tuple[int, int] = (16, 8)) -> Tuple[plt.Figure, Dict[int, np.ndarray]]:
        """Visualize the strength/clarity of cluster boundaries (simplified implementation).
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        anomalia_clusters : Optional[List[int]]
            List of cluster IDs to consider as anomalies
        n_samples : int
            Number of samples to generate for flooding
        flood_method : str
            Method for flooding
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        Tuple[plt.Figure, Dict[int, np.ndarray]]
            Matplotlib figure and flooded samples dictionary
        """
        # Get cluster information
        cluster_info = self.cluster_extractor.extract_cluster_info(X, anomalia_clusters)
        
        # Create figure (implementation would be similar to plot_flooded_latent_space)
        # This is just a placeholder for the refactored interface
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "Boundary Strength Visualization (Placeholder)", 
               ha='center', va='center', fontsize=14)
        
        # For demonstration purposes, return some sample data
        flooded_samples = SampleGenerator.flood_latent_space(
            cluster_info, 
            n_samples=n_samples,
            method=flood_method
        )
        
        return fig, flooded_samples
    
    def plot_normal_behavior(self, X: np.ndarray, normal_clusters: Optional[List[int]] = None, 
                          n_samples: int = 1000, figsize: Tuple[int, int] = (16, 8), 
                          visualization_type: str = 'density', bandwidth: float = 0.2, 
                          confidence_interval: Tuple[int, int] = (5, 95)) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize aggregated normal behavior from selected clusters (simplified implementation).
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        normal_clusters : Optional[List[int]]
            List of cluster IDs to consider as normal
        n_samples : int
            Number of samples to generate
        figsize : Tuple[int, int]
            Figure size
        visualization_type : str
            Type of visualization
        bandwidth : float
            Bandwidth parameter for KDE
        confidence_interval : Tuple[int, int]
            Lower and upper percentiles for confidence intervals
            
        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib figure and normal samples array
        """
        # Get cluster information
        cluster_info = self.cluster_extractor.extract_cluster_info(X)
        
        # Create figure (implementation would be similar to plot_kde_projection)
        # This is just a placeholder for the refactored interface
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "Normal Behavior Visualization (Placeholder)", 
               ha='center', va='center', fontsize=14)
        
        # For demonstration purposes, return some sample data
        normal_samples = np.zeros((n_samples, self.model.latent_dim))
        
        return fig, normal_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from cae import CAEVizT
    
    # Generate some example data (multivariate time series)
    n_samples = 2000
    seq_len = 7
    n_features = 1
    
    # Create sine waves with different frequencies as an example
    time = np.linspace(0, 10, seq_len)
    X = np.zeros((n_samples, seq_len, n_features))
    
    # First half: sin(t)
    for i in range(n_samples // 2):
        X[i, :, 0] = np.sin(time) + np.random.normal(0, 0.1, seq_len)
    
    # Second half: sin(2t)
    for i in range(n_samples // 2, n_samples):
        X[i, :, 0] = np.sin(2 * time) + np.random.normal(0, 0.1, seq_len)
    
    # Create and fit the model
    model = CAEVizT(
        input_shape=X.shape[1:],
        latent_dim=16,
        filters=[16, 32, 64],
        clustering_model='kmeans',
        n_clusters=2,
        top_k=2,
        seq_len=seq_len,
    )

    model.fit(X, 
              batch_size=32, 
              verbose=1, 
              stats_window=60,
              epochs=4,
              pretraining_epochs=2,
              spectrum_epochs=3
              )
        
    if model is not None:
        # Sample data for testing
        X_test_sample = X[np.random.choice(range(len(X)), int(n_samples/10))]
    
        # Create the projection visualizer
        projector = CAEVizTProjection(model)
    
        # Basic usage
        fig, samples = projector.plot_kde_projection(X_test_sample, n_samples=1000)
        plt.show()
    
        # With specified anomaly clusters
        fig, samples = projector.plot_kde_projection(X_test_sample, anomalia_clusters=[1], n_samples=1000)
        plt.show()