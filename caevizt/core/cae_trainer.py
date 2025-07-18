import tensorflow as tf
import numpy as np
from .cae_analyzer import CAEVizTAnalyzer
from .cae_loss import get_cluster_probabilities

class CAEVizTTrainer:
    """Responsible for training the CAEVizT model."""
    
    def __init__(self, base, loss_function=None):
        """Initialize with reference to the base class instance."""
        self.base = base
        self.loss_function = loss_function
        
    def _rank_channels(self, centers_prev, centers_curr, top_k):
        """Rank the latent channels by computing the maximum absolute difference."""
        differences = np.abs(centers_prev - centers_curr)
        channel_differences = np.max(differences, axis=0)
        sorted_indices = np.argsort(-channel_differences)
        return sorted_indices[:top_k]
    
    def _apply_selective_gradients(self, grads, trainable_vars, top_k_indices, update_top_k=True):
        """Apply gradients selectively based on top_k_indices."""
        masked_grads = []
        for var, grad in zip(trainable_vars, grads):
            if "latent_space" in var.name:
                if "kernel" in var.name:
                    # For kernel, shape is (input_dim, latent_dim)
                    mask = np.ones(grad.shape[1], dtype=np.float32)
                    if update_top_k:
                        mask[:] = 0.0
                        mask[top_k_indices] = 1.0
                    else:
                        mask[top_k_indices] = 0.0
                    mask = tf.constant(mask, dtype=grad.dtype)
                    mask = tf.reshape(mask, (1, -1))  # Broadcast over input dimension
                    masked_grad = grad * mask
                    masked_grads.append(masked_grad)
                elif "bias" in var.name:
                    # For bias, shape is (latent_dim,)
                    mask = np.ones(grad.shape[0], dtype=np.float32)
                    if update_top_k:
                        mask[:] = 0.0
                        mask[top_k_indices] = 1.0
                    else:
                        mask[top_k_indices] = 0.0
                    mask = tf.constant(mask, dtype=grad.dtype)
                    masked_grad = grad * mask
                    masked_grads.append(masked_grad)
                else:
                    # Unexpected variable in latent layer
                    masked_grads.append(grad if update_top_k else tf.zeros_like(grad))
            else:
                # Other layers
                masked_grads.append(tf.zeros_like(grad) if update_top_k else grad)
        
        return masked_grads
        
    def train_phase_reconstruction(self, model, X, **kwargs):
        """Phase 1: Pretraining - update using only reconstruction loss."""
        print("=== Pre-training Phase ===")
        self.base.training_phase_ = "pretrain"
        self.base.current_epoch_ = 0
        
        # Set the loss function phase
        if self.loss_function:
            self.loss_function.set_phase("pretrain")
        
        # Set epochs to pretraining epochs for first phase
        pretrain_kwargs = kwargs.copy()
        pretrain_kwargs = self.base.remove_specific_kwargs(pretrain_kwargs)
        pretrain_kwargs['epochs'] = self.base.pretraining_epochs_
        
        # Train on device
        gpus = tf.config.list_physical_devices('GPU')
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            model.fit(X, X, **pretrain_kwargs)

        self.base._append_history(model.history.history)
        
        # After pretraining, compute initial clusters
        encoded = self.base.encoder_.predict(X, verbose=0)
        self.base.cluster_model_.fit(encoded)
        
        # Initialize cluster centers and labels
        if hasattr(self.base.cluster_model_, 'cluster_centers_'):
            self.base.current_cluster_centers_ = self.base.cluster_model_.cluster_centers_
            self.base.previous_cluster_centers_ = self.base.current_cluster_centers_.copy()
        else:  # For GMM
            self.base.current_cluster_centers_ = self.base.cluster_model_.means_
            self.base.previous_cluster_centers_ = self.base.current_cluster_centers_.copy()
        
        if hasattr(self.base.cluster_model_, 'labels_'):
            self.base.labels_ = self.base.cluster_model_.labels_
        else:
            self.base.labels_ = self.base.cluster_model_.predict(encoded)
        
        # Compute initial rank of channels (all equal at start)
        self.base.top_k_indices_ = np.arange(self.base.latent_dim)[:self.base.top_k]

    def train_phase_clustering(self, model, X, **kwargs):
        """Phase 2: Fine-tuning - selective updates with clustering loss."""
        print("\n=== Fine-tuning Phase : Deep Clustering ===")
        self.base.training_phase_ = "finetune"
        
        # Set the loss function phase
        if self.loss_function:
            self.loss_function.set_phase("finetune")
        
        # Set up remaining epochs
        total_epochs = kwargs.get('epochs', 100)
        finetune_epochs = total_epochs - self.base.pretraining_epochs_
        
        # Define a custom training loop for fine-tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get('learning_rate', 0.001))
        batch_size = kwargs.get('batch_size', 32)
        
        # Get validation data
        validation_data = kwargs.get('validation_data', None)
        validation_split = kwargs.get('validation_split', 0.0)
        X_val = None
        
        if validation_data is not None and isinstance(validation_data, tuple):
            X_val = validation_data[0]  # Use the first element as validation features
        elif validation_split > 0:
            # Use a portion of X as validation data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            X = X_train  # Update X to only contain training data
        
        # Create dataset for training data
        dataset = tf.data.Dataset.from_tensor_slices((X, X))
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
        
        # Get device
        gpus = tf.config.list_physical_devices('GPU')
        device = '/GPU:0' if gpus else '/CPU:0'
        
        history = {'loss': [], 'val_loss': []}
        for epoch in range(finetune_epochs):
            self.base.current_epoch_ = self.base.pretraining_epochs_ + epoch + 1
            epoch_loss = []
            
            for x_batch, y_batch in dataset:
                y_batch = tf.cast(y_batch, tf.float32)
                with tf.device(device):
                    # Get latent representations and cluster probabilities
                    latent = self.base.encoder_(x_batch, training=False)
                    q = get_cluster_probabilities(latent, self.base.current_cluster_centers_)
                    
                    # Update the loss function with current latent and q
                    if self.loss_function:
                        self.loss_function.update_latent_and_q(latent, q)
                    
                    # First pass: Update only top K channels
                    with tf.GradientTape() as tape1:
                        predictions = model(x_batch, training=True)
                        # Use our loss function
                        loss = model.loss(y_batch, predictions)
                    
                    grads1 = tape1.gradient(loss, model.trainable_variables)
                    masked_grads1 = self._apply_selective_gradients(
                        grads1, model.trainable_variables, self.base.top_k_indices_, update_top_k=True
                    )
                    optimizer.apply_gradients(zip(masked_grads1, model.trainable_variables))
                    
                    # Second pass: Update all except top K channels
                    with tf.GradientTape() as tape2:
                        predictions = model(x_batch, training=True)
                        # Use our loss function
                        loss = model.loss(y_batch, predictions)
                    
                    grads2 = tape2.gradient(loss, model.trainable_variables)
                    masked_grads2 = self._apply_selective_gradients(
                        grads2, model.trainable_variables, self.base.top_k_indices_, update_top_k=False
                    )
                    optimizer.apply_gradients(zip(masked_grads2, model.trainable_variables))
                    
                    epoch_loss.append(loss.numpy())
            
            avg_loss = np.mean(epoch_loss)
            history['loss'].append(avg_loss)
            
            # Calculate validation loss
            with tf.device(device):
                # Get latent and q for validation data
                val_latent = self.base.encoder_(X_val, training=False)
                val_q = get_cluster_probabilities(val_latent, self.base.current_cluster_centers_)
                
                # Update loss function with validation latent and q
                if self.loss_function:
                    self.loss_function.update_latent_and_q(val_latent, val_q)
                
                val_predictions = model.predict(X_val, batch_size=batch_size, verbose=0)
                val_loss = model.loss(X_val, val_predictions).numpy()
                history['val_loss'].append(val_loss)
            
            print(f"Fine-tune Epoch {self.base.current_epoch_}/{total_epochs}  Loss: {avg_loss:.4f}  Val Loss: {val_loss:.4f}")
            
            # Update clusters every cluster_update_interval epochs
            if (epoch + 1) % self.base.cluster_update_interval == 0:
                encoded = self.base.encoder_.predict(X, verbose=0)
                self.base.cluster_model_.fit(encoded)
                
                # Update centers and assignments
                self.base.previous_cluster_centers_ = self.base.current_cluster_centers_.copy()
                
                if hasattr(self.base.cluster_model_, 'cluster_centers_'):
                    self.base.current_cluster_centers_ = self.base.cluster_model_.cluster_centers_
                else:  # For GMM
                    self.base.current_cluster_centers_ = self.base.cluster_model_.means_
                
                if hasattr(self.base.cluster_model_, 'labels_'):
                    self.base.labels_ = self.base.cluster_model_.labels_
                else:
                    self.base.labels_ = self.base.cluster_model_.predict(encoded)
                
                # Re-rank channels
                self.base.top_k_indices_ = self._rank_channels(
                    self.base.previous_cluster_centers_, 
                    self.base.current_cluster_centers_, 
                    self.base.top_k
                )
                
                max_dist = self.base._compute_max_euclidean_distance(
                    self.base.previous_cluster_centers_, 
                    self.base.current_cluster_centers_
                )
                
                print(f"Epoch {self.base.current_epoch_}: Updated clusters. "
                    f"Top {self.base.top_k} channels: {self.base.top_k_indices_}. "
                    f"Max Euclidean dist: {max_dist:.4f}")
                
                if max_dist < 1e-8:
                    print("Convergence reached. Stopping training.")
                    break
        # Store the training history
        self.base._append_history(history)

    def train_phase_spectrum(self, model, X, **kwargs):
        """Phase 3: Statistical Spectrum Enhancement."""
        print("\n=== Statistical Spectrum Enhancement Phase ===")
        
        # Set the loss function phase
        if self.loss_function:
            self.loss_function.set_phase("spectrum")
        
        stats_window = kwargs.get('stats_window', 10)
        stats_weight = kwargs.get('stats_weight', 0.1)
        spectrum_epochs = kwargs.get('spectrum_epochs', 10)
        epsilon_tolerance = kwargs.get('epsilon', 5)
        
        # Get validation data
        validation_data = kwargs.get('validation_data', None)
        validation_split = kwargs.get('validation_split', 0.0)
        X_val = None
        
        if validation_data is not None and isinstance(validation_data, tuple):
            X_val = validation_data[0]  # Use the first element as validation features
        elif validation_split > 0:
            # Use a portion of X as validation data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            X = X_train  # Update X to only contain training data

        analyzer = CAEVizTAnalyzer(self.base)
        
        # Compute statistical spectrum for each feature
        feature_spectrums_list = [analyzer.compute_stat_spectrum(X[:, j, i], stats_window) for i in range(self.base.n_features) for j in range(self.base.seq_len)]

        feature_spectrums = {}
        for i, spectrum_list in enumerate(feature_spectrums_list):
            for j, spectrums in enumerate(spectrum_list):
                min_val = spectrums[0]
                max_val = spectrums[1]
                feature_spectrums[(i, j)] = (min_val, max_val)

        # Update the loss function with feature spectrums and stats weight
        if self.loss_function:
            self.loss_function.update_feature_spectrums(feature_spectrums)
            self.loss_function.stats_weight = stats_weight
        
        # Training loop for statistical spectrum enhancement
        optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get('learning_rate', 0.001))
        batch_size = kwargs.get('batch_size', 32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, X))
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
        
        # Get device
        gpus = tf.config.list_physical_devices('GPU')
        device = '/GPU:0' if gpus else '/CPU:0'
        
        history = {'loss': [], 'val_loss': []}
        for epoch in range(spectrum_epochs):
            epoch_losses = []
            
            for x_batch, y_batch in dataset:
                with tf.device(device):
                    # Ensure x_batch is a float tensor
                    x_batch = tf.cast(x_batch, tf.float32)
                    
                    # Get latent representation and cluster probabilities
                    latent = self.base.encoder_(x_batch, training=False)
                    q = get_cluster_probabilities(latent, self.base.current_cluster_centers_)
                    
                    # Update the loss function
                    if self.loss_function:
                        self.loss_function.update_latent_and_q(latent, q)
                    
                    with tf.GradientTape() as tape:
                        # Reconstruction
                        reconstructed = tf.cast(model(x_batch, training=True), tf.float32)
                        
                        # Use our unified loss function
                        total_loss = model.loss(x_batch, reconstructed)
                    
                    # Get and apply gradients
                    trainable_vars = model.trainable_variables
                    gradients = tape.gradient(total_loss, trainable_vars)
                    optimizer.apply_gradients(zip(gradients, trainable_vars))
                    
                    epoch_losses.append(total_loss.numpy())
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            # Calculate validation loss
            with tf.device(device):
                X_val_float = tf.cast(X_val, tf.float32)
                val_latent = self.base.encoder_(X_val_float, training=False)
                val_reconstructed = model(X_val_float, training=False)
                val_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(X_val_float - val_reconstructed), axis=1))
                val_q = analyzer.get_cluster_probabilities(val_latent)
                val_stats_loss = analyzer.compute_spectrum_loss(
                    X_val_float, 
                    val_q, 
                    feature_spectrums
                    )
                val_total_loss = val_rec_loss + self.base.train_alpha * analyzer.get_clustering_loss(val_latent, val_q) + stats_weight * val_stats_loss
                history['val_loss'].append(val_total_loss.numpy())
            
            print(f"Spectrum Enhancement Epoch {epoch+1}/{spectrum_epochs}  Loss: {avg_loss:.4f}  Val Loss: {val_total_loss:.4f}")
            
            # Update clusters at the end of each epoch
            if (epoch + 1) % self.base.cluster_update_interval == 0 or epoch == spectrum_epochs - 1:
                encoded = self.base.encoder_.predict(X, verbose=0)

                self.base.cluster_model_.fit(encoded)
                
                # Update assignments
                if hasattr(self.base.cluster_model_, 'labels_'):
                    self.base.labels_ = self.base.cluster_model_.labels_
                else:
                    self.base.labels_ = self.base.cluster_model_.predict(encoded)
                
                # Calculate cluster statistics
                cluster_stats = analyzer.calculate_cluster_statistics(X, self.base.labels_, feature_spectrums)
                print("Cluster statistics based on statistical spectrum:")
                for cluster_id, stats in cluster_stats.items():
                    print(f"\tCluster {cluster_id}: {stats['in_percent']:.1f}% in-spectrum, {stats['out_percent']:.1f}% out-spectrum")
        
        # Store the training history
        self.base._append_history(history)
        # Classify clusters based on statistical spectrum
        self.base.anomaly_clusters_ = analyzer.classify_clusters(
            X, 
            stats_window=stats_window, 
            epsilon=epsilon_tolerance
            )