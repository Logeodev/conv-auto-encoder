import tensorflow as tf

class CAEVizTPredictor:
    """Responsible for making predictions with the CAEVizT model."""
    
    def __init__(self, base):
        """Initialize with reference to the base class instance."""
        self.base = base
    
    def transform(self, X):
        """Transform data to the latent space representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            Input data
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, latent_dim)
            Latent representation of the input data
        """
        return self.base.encoder_.predict(X, verbose=0)
    
    def predict(self, X, model, verbose=0):
        """Reconstruct input data using the autoencoder.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            Input data
        model : keras.Model
            The autoencoder model
            
        Returns
        -------
        X_reconstructed : array-like
            Reconstructed data
        """
        return model.predict(X, verbose=verbose)
    
    def predict_clusters(self, X=None):
        """Predict cluster labels for samples in X.
        
        If X is not provided, returns the precomputed labels from fit.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features), default=None
            New data for clustering. If None, returns the labels from fit.
            
        Returns
        -------
        labels : array-like of shape (n_samples,)
            Cluster labels
        """
        if X is None:
            if self.base.labels_ is None:
                raise ValueError("Model has not been fitted yet.")
            return self.base.labels_
        
        encoded = self.transform(X)
        return self.base.cluster_model_.predict(encoded)
    
    def fine_tune_supervised(self, X_labeled, y_labeled, **kwargs):
        """Fine-tune the model using labeled anomaly data.
        
        Parameters
        ----------
        X_labeled : array-like of shape (n_samples, seq_len, n_features) or (n_samples, seq_len*n_features)
            Labeled input data
        y_labeled : array-like of shape (n_samples,)
            Labels (0 for normal, 1 for anomaly)
        **kwargs : dict
            Additional arguments to pass to the fine-tuning process
            
        Returns
        -------
        self : object
            Fine-tuned estimator
        """
        # Ensure model was already trained in unsupervised mode
        if self.base.encoder_ is None:
            raise ValueError("Model must be fitted with unsupervised data first before fine-tuning")
        
        # Set default parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 20)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        validation_split = kwargs.get('validation_split', 0.2)
        
        print("\n=== Supervised Fine-tuning Phase ===")
        
        # Create a classifier head on top of the encoder
        latent_input = tf.keras.layers.Input(shape=(self.base.latent_dim,))
        x = tf.keras.layers.Dense(32, activation='relu')(latent_input)
        x = tf.keras.layers.Dropout(0.3)(x)
        classifier_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.base.classifier_ = tf.keras.Model(latent_input, classifier_output)
        
        # Create end-to-end model by combining encoder and classifier
        input_layer = tf.keras.layers.Input(shape=self.base.input_shape)
        latent = self.base.encoder_(input_layer)
        output = self.base.classifier_(latent)
        self.base.supervised_model_ = tf.keras.Model(input_layer, output)
        
        # Compile the supervised model
        self.base.supervised_model_.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train on device
        gpus = tf.config.list_physical_devices('GPU')
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            # Fine-tune with labeled data
            self.base.supervised_model_.fit(
                X_labeled, y_labeled,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        
        # Update cluster assignments with the new latent space
        encoded = self.transform(X_labeled)
        self.base.cluster_model_.fit(encoded)
        
        # Update centers and assignments
        if hasattr(self.base.cluster_model_, 'cluster_centers_'):
            self.base.current_cluster_centers_ = self.base.cluster_model_.cluster_centers_
        else:  # For GMM
            self.base.current_cluster_centers_ = self.base.cluster_model_.means_
        
        return self.base
    
    def predict_anomaly(self, X, threshold=0.5):
        """Predict if samples in X are anomalies using the supervised model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features) or (n_samples, seq_len*n_features)
            Input data
        threshold : float, default=0.5
            Decision threshold for binary classification
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels (0 for normal, 1 for anomaly)
        y_scores : array-like of shape (n_samples,)
            Anomaly scores (probability of being an anomaly)
        """
        if not hasattr(self.base, 'supervised_model_'):
            raise ValueError("Model has not been fine-tuned with labeled data yet.")
        
        # Get anomaly scores
        y_scores = self.base.supervised_model_.predict(X, verbose=0)
        y_scores = y_scores.ravel()
        
        # Convert to binary predictions
        y_pred = (y_scores >= threshold).astype(int)
        
        return y_pred, y_scores