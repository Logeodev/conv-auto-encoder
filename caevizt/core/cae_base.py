import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class CAEVizTBase:
    """Base class with shared functionality for the CAEVizT model family."""
    
    CLUSTERING_MODELS = {
        'kmeans': KMeans,
        'gmix': GaussianMixture
    }

    SPECIFIC_KWARGS = [
            'stats_window', 'stats_weight', 'spectrum_epochs', 'epsilon',
            'pretraining_epochs', 'seq_len', 'n_features', 'filters'
        ]
    
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
        """Initialize shared attributes for the CAEVizT model family."""
        # Store original dimensions if input is flattened
        if isinstance(input_shape, tuple) and len(input_shape) == 1:
            # Assume input is flattened (samples, seq_len*features)
            # You need to specify sequence length and features
            self.seq_len = kwargs.pop('seq_len', 30)
            self.n_features = kwargs.pop('n_features', 1)
        elif isinstance(input_shape, tuple) and len(input_shape) == 2:
            # Input is already 3D (seq_len, features)
            self.seq_len = input_shape[0]
            self.n_features = input_shape[1]
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.clustering_model = clustering_model
        self.n_clusters = n_clusters
        self.clustering_params = {} if clustering_params is None else clustering_params
        self.pretrain_alpha = pretrain_alpha
        self.train_alpha = train_alpha
        self.cluster_update_interval = cluster_update_interval
        self.top_k = top_k
        
        # Additional shared attributes
        self.encoder_ = None
        self.latent_model_ = None
        self._callbacks = None
        self.cluster_model_ = None
        self.labels_ = None
        self.current_cluster_centers_ = None
        self.previous_cluster_centers_ = None
        self.top_k_indices_ = None
        self.training_phase_ = "pretrain"  # Can be "pretrain" or "finetune"
        self.pretraining_epochs_ = 0
        self.current_epoch_ = 0
        self.anomaly_clusters_ = None
        self.history_ = None
        
    def _init_clustering_model(self):
        """Initialize the clustering model based on the selected algorithm."""
        if self.clustering_model.lower() not in self.CLUSTERING_MODELS:
            raise ValueError(f"Unsupported clustering model: {self.clustering_model}. "
                             f"Supported models are: {list(self.CLUSTERING_MODELS.keys())}")
                
        model_class = self.CLUSTERING_MODELS[self.clustering_model.lower()]
        params = self.clustering_params.copy()
        
        # Add n_clusters parameter for algorithms that need it
        if self.clustering_model.lower() in ['kmeans']:
            params.setdefault('n_clusters', self.n_clusters)
        elif self.clustering_model.lower() in ['gmix']:
            params.setdefault('n_components', self.n_clusters)
                
        self.cluster_model_ = model_class(**params)

    def _append_history(self, history):
        """Append training history to the model's history.
        
        Parameters
        ----------
        history : History
            The training history to append.
        """
        if self.history_ is None:
            self.history_ = history
        else:
            for key in history.keys():
                if key in self.history_:
                    self.history_[key].extend(history[key])
                else:
                    self.history_[key] = history[key]

    def _compute_max_euclidean_distance(self, centers_prev, centers_curr):
        """Compute the maximum Euclidean distance between corresponding centers."""
        distances = np.linalg.norm(centers_prev - centers_curr, axis=1)
        return np.max(distances)
    
    @staticmethod
    def remove_specific_kwargs(kwargs):
        """Remove specific kwargs that are not needed for the base class."""
        for key in CAEVizTBase.SPECIFIC_KWARGS:
            kwargs.pop(key, None)
        return kwargs