import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.base import ClusterMixin
from core.cae_base import CAEVizTBase
from core.cae_builder import CAEVizTBuilder
from core.cae_trainer import CAEVizTTrainer
from core.cae_analyzer import CAEVizTAnalyzer
from core.cae_predictor import CAEVizTPredictor
from core.cae_loss import CombinedLoss

class CAEVizT(KerasRegressor, ClusterMixin):
    """Convolutional Auto-Encoder for Visualization with clustering and statistical spectrum analysis.
    
    This class implements the architecture and training procedure, described in [the article by Hooshmand & Huchaiah](https://doi.org/10.54963/dtra.v1i2.64), combining dimensionality reduction through a convolutional autoencoder with clustering algorithms.

    Note that in this implementation, another step has been added: statistical spectrum analysis, which enhances the clustering performance by analyzing the statistical properties of the data, allowing pseudo-classification of latent clusters.
    """
    
    def __init__(self, 
                 input_shape=(30, 1),
                 latent_dim=64, 
                 filters=[32, 64, 128],
                 dropout_rate=0.2,
                 clustering_model='kmeans',
                 n_clusters=10,
                 clustering_params=None,
                 pretrain_alpha=0.0,
                 train_alpha=0.5,
                 cluster_update_interval=3,
                 top_k=10,
                 **kwargs):
        """Initialize the CAEVizT model."""
        # Create the base component
        self.base = CAEVizTBase(
            input_shape=input_shape,
            latent_dim=latent_dim,
            filters=filters,
            dropout_rate=dropout_rate,
            clustering_model=clustering_model,
            n_clusters=n_clusters,
            clustering_params=clustering_params,
            pretrain_alpha=pretrain_alpha,
            train_alpha=train_alpha,
            cluster_update_interval=cluster_update_interval,
            top_k=top_k,
            **kwargs
        )
        
        # Create unified loss function for all phases
        self.loss_function = CombinedLoss(
            pretrain_alpha=pretrain_alpha,
            train_alpha=train_alpha,
            stats_weight=0.1  # Default value, will be updated during training
        )
        
        # Create component instances
        self.builder = CAEVizTBuilder(self.base, loss_function=self.loss_function)
        self.trainer = CAEVizTTrainer(self.base, loss_function=self.loss_function)
        self.analyzer = CAEVizTAnalyzer(self.base)
        self.predictor = CAEVizTPredictor(self.base)
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Make sure we're using our combined loss in compilation
        if 'loss' not in kwargs:
            kwargs['loss'] = self.loss_function
            
        # Build the model
        self.model_ = self.builder._keras_build_fn(compile_kwargs=kwargs)
    
    def fit(self, X, y=None, **kwargs):
        """Fit the autoencoder and clustering model.
        
        This method performs the following steps:
        1. Pretraining phase: Train the autoencoder using reconstruction loss.
        2. Clustering phase: Compute initial clusters using KMeans or GMM.
        3. Fine-tuning phase: Update the autoencoder using clustering loss and selective channel updates.
        4. Statistical spectrum enhancement: Apply statistical spectrum analysis to improve clustering.
                
        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            Training data
        y : ignored
            Not used, present for API consistency
        **kwargs : dict
            Additional arguments to pass to the autoencoder's fit method
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Make sure validation data is available for early stopping
        if 'validation_split' not in kwargs and 'validation_data' not in kwargs:
            kwargs['validation_split'] = 0.1
        
        # Set up callbacks
        if 'callbacks' in kwargs and isinstance(kwargs['callbacks'], list):
            kwargs['callbacks'].extend(self.base._callbacks)
        else:
            kwargs['callbacks'] = self.base._callbacks
        
        # Create clustering model
        self.base._init_clustering_model()
        
        # Determine number of epochs for pretraining
        total_epochs = kwargs.get('epochs', 100)
        self.base.pretraining_epochs_ = kwargs.get('pretraining_epochs', int(total_epochs * 0.5))

        # Phase 1: Pretraining
        if 'pretraining_epochs' in kwargs and kwargs['pretraining_epochs'] > 0:
            # Set loss function to pretraining phase
            self.loss_function.set_phase("pretrain")
            self.trainer.train_phase_reconstruction(self.model_, X, **kwargs)

        # Phase 2: Fine-tuning
        if 'epochs' in kwargs and kwargs['epochs'] > 0:
            # Set loss function to fine-tuning phase
            self.loss_function.set_phase("finetune")
            self.trainer.train_phase_clustering(self.model_, X, **kwargs)
        
        # Phase 3: Statistical Spectrum Enhancement
        if 'spectrum_epochs' in kwargs and kwargs['spectrum_epochs'] > 0:
            # Set loss function to spectrum enhancement phase
            self.loss_function.set_phase("spectrum")
            self.trainer.train_phase_spectrum(self.model_, X, **kwargs)
        
        return self
    
    def transform(self, X):
        """Transform data to the latent space representation."""
        return self.predictor.transform(X)
    
    def predict(self, X, verbose=0):
        """Reconstruct input data using the autoencoder."""
        return self.predictor.predict(X, self.model_, verbose=verbose)
    
    def predict_clusters(self, X=None):
        """Predict cluster labels for samples in X."""
        return self.predictor.predict_clusters(X)
    
    def fit_predict(self, X, y=None, **kwargs):
        """Fit the model and predict cluster labels for X."""
        self.fit(X, **kwargs)
        return self.base.labels_
        
    def fine_tune_supervised(self, X_labeled, y_labeled, **kwargs):
        """Fine-tune the model using labeled anomaly data."""
        return self.predictor.fine_tune_supervised(X_labeled, y_labeled, **kwargs)
    
    def predict_anomaly(self, X, threshold=0.5):
        """Predict if samples in X are anomalies using the supervised model."""
        return self.predictor.predict_anomaly(X, threshold)
    
    @property
    def labels_(self):
        """Get the cluster labels from the base component."""
        return self.base.labels_
    
    @property
    def cluster_centers_(self):
        """Get the cluster centers from the base component."""
        return self.base.current_cluster_centers_
    
    @property
    def encoder_(self):
        """Get the encoder model from the base component."""
        return self.base.encoder_
    
    @property
    def latent_dim(self):
        """Get the latent dimension from the base component."""
        return self.base.latent_dim
    
    @property
    def history_(self):
        """Get the training history from the base component."""
        return self.base.history_
    
    @property
    def cluster_model_(self):
        """Get the clustering model from the base component."""
        return self.base.cluster_model_
    
    @property
    def n_clusters(self):
        """Get the number of clusters from the base component."""
        return self.base.n_clusters
    
    @staticmethod
    def reshape_data(X):
        """Format data to the expected shape for the model : (n_samples, seq_len, n_features)"""
        if X.ndim == 1:
            return X.reshape(-1, 1, 1)
        elif X.ndim == 2:
            n_features = X.shape[1]
            return X.reshape(-1, 1, n_features)
        elif X.ndim == 3:
            return X
        else:
            raise ValueError("Input data must be up to 3D array-like. : \n\t1D: (n_samples, ) -> (n_samples, 1, 1)\n\t2D: (n_samples, n_features) -> (n_samples, 1, n_features)\n\t3D: (n_samples, seq_len, n_features) -> (n_samples, seq_len, n_features)")
        
    @staticmethod
    def flatten_data(X:np.ndarray):
        """Flatten data to 2D array-like."""
        if X.ndim == 1:
            return X.reshape(-1, 1)
        elif X.ndim == 2:
            return X
        elif X.ndim == 3:
            return X[:, -1, :]
        else:
            raise ValueError("Input data must be up to 3D array-like. : \n\t1D: (n_samples, ) -> (n_samples, 1)\n\t2D: (n_samples, n_features) -> (n_samples, n_features)\n\t3D: (n_samples, seq_len, n_features) -> (n_samples, n_features) with supposed continuity in sequences")


# Main execution is left unchanged for compatibility
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import silhouette_score
    
    # Generate some example data (multivariate time series)
    n_samples = 200
    seq_len = 30
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
    
    X = CAEVizT.reshape_data(X)

    # Create and fit the model
    model = CAEVizT(
        input_shape=X.shape[1:],
        latent_dim=16,
        filters=[16, 32, 64],
        clustering_model='kmeans',
        n_clusters=5,
        top_k=2
    )

    model.fit(X, 
              batch_size=32, 
              verbose=1, 
              stats_window=60,
              epochs=20,
              pretraining_epochs=10,
              spectrum_epochs=30
              )
    
    # Get clusters
    clusters = model.predict_clusters(X)
    
    # Visualize latent space with PCA
    from sklearn.decomposition import PCA
    
    latent_repr = model.transform(X)
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_repr)
    
    # Plot clusters
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.title('Latent Space Visualization with Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Add silhouette score
    sil_score = silhouette_score(latent_repr, clusters)
    plt.suptitle(f'Silhouette Score: {sil_score:.3f}')

    plt.subplot(2, 1, 2)
    plt.plot(model.base.history_['loss'], label='Training Loss')
    plt.plot(model.base.history_['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()