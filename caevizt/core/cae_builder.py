import tensorflow as tf
from .cae_base import CAEVizTBase

class CAEVizTBuilder:
    """Responsible for building the CAEVizT model architecture."""
    
    def __init__(self, base, loss_function=None):
        """Initialize with reference to the base class instance."""
        self.base = base
        self.loss_function = loss_function
        
    def _keras_build_fn(self, compile_kwargs):
        """Build function that creates and returns the autoencoder model."""
        # Input is flattened (2D), so we need to reshape it back to 3D
        if hasattr(self.base, 'seq_len') and hasattr(self.base, 'n_features'):
            seq_len = self.base.seq_len
            features = self.base.n_features
        else:
            seq_len = 30  # Default value if not provided 
            features = 1  # Default number of features
        # Extract sequence length and features from the input_shape
        if len(self.base.input_shape) == 1:  # If input is flattened
            # For flattened input, we need to know the original 3D shape
            # Assuming input_shape is (seq_len * features,)
            # Let's reconstruct based on known pattern - the original model expects (seq_len, features)
            
            # Create the model with flattened input
            inputs = tf.keras.layers.Input(shape=self.base.input_shape)
            
            # Reshape flattened input back to 3D for convolutional layers
            x = tf.keras.layers.Reshape((seq_len, features))(inputs)
        else:
            # If input is already 3D, use it directly
            inputs = tf.keras.layers.Input(shape=self.base.input_shape)
            x = inputs
            features = self.base.input_shape[1]  # Number of features in the input shape
        
        # Encoder - now using the reshaped input 'x'
        skip_connections = []
        
        for f in self.base.filters:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=3, activation='relu', padding='same')(x)
            skip_connections.append(tf.keras.layers.Lambda(lambda y: y)(x))
            x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
            x = tf.keras.layers.Dropout(self.base.dropout_rate)(x)
        
        # Latent space
        latent = tf.keras.layers.Conv1D(self.base.latent_dim, kernel_size=3, activation='relu', padding='same')(x)
        flat_latent = tf.keras.layers.Flatten()(latent)
        latent_dense = tf.keras.layers.Dense(self.base.latent_dim, activation='relu', name="latent_space")(flat_latent)
        
        # Decoder
        x = tf.keras.layers.Dense(tf.keras.backend.int_shape(flat_latent)[1], activation='relu')(latent_dense)
        x = tf.keras.layers.Reshape(tf.keras.backend.int_shape(latent)[1:])(x)
        
        for i, f in enumerate(reversed(self.base.filters)):
            x = tf.keras.layers.Conv1DTranspose(f, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling1D(2)(x)
            # Crop to match the corresponding encoder layer's shape if needed
            if i < len(skip_connections):
                target_shape = skip_connections[-(i+1)].shape[1]
                if x.shape[1] != target_shape:
                    x = tf.keras.layers.Cropping1D(
                        cropping=(0, x.shape[1] - target_shape)
                    )(x)
            x = tf.keras.layers.Dropout(self.base.dropout_rate)(x)
        
        # 3D output (seq_len, features)
        x = tf.keras.layers.Conv1DTranspose(features, 3, activation='linear', padding='same')(x)
        
        # If input was flattened, flatten the output too to match
        if len(self.base.input_shape) == 1:
            outputs = tf.keras.layers.Flatten()(x)
        else:
            outputs = x
        
        autoencoder = tf.keras.Model(inputs, outputs)
        
        # Create encoder model for later use - need to consider the reshape
        # For latent representation, we'll always output the dense latent space
        self.base.encoder_ = tf.keras.Model(inputs, latent_dense)
        
        # Use the provided loss function if available, otherwise use the default
        if self.loss_function and 'loss' not in compile_kwargs:
            compile_kwargs['loss'] = self.loss_function
        
        # Add early stopping callbacks
        self.base._callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=2,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', 
                patience=2,
                restore_best_weights=True,
                verbose=1
            )
        ]

        autoencoder.compile(**CAEVizTBase.remove_specific_kwargs(compile_kwargs))
        
        return autoencoder