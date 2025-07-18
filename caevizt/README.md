# CAEVizT : Key Components of the Implementation

## Architecture

The autoencoder has three main components:

### Encoder

Uses 1D convolutional layers followed by max pooling to compress the time series data into a lower-dimensional latent space

### Latent Space

A dense layer that represents the bottleneck where dimensionality reduction occurs

### Decoder

Uses upsampling and convolutional layers to reconstruct the original time series from the latent representation

## scikit-learn Compatibility

The class implements `ClustererMixin` (from `sklearn`) and `KerasRegressor` (from `scikeras`). This enables the class to be completely scikit-learn compatible and wraps any superior mechanisms.

## Clustering Capabilities

The class inherits from `ClusterMixin` and performs clustering in the latent space:

1. The encoder extracts meaningful features from the time series data
2. `clusterer_model_` is applied to these features to apply clustering of latent space features
3. The class comes with methods to visualize the latent space with cluster assignments (_cf._ `./cae_plot.py`, "## Visualisation Tools")

## Visualization Tools

The class provides several visualization methods:

`plot_history`: Shows the training loss over epochs
`plot_latent_space`: Visualizes the latent representations with cluster assignments
`plot_reconstruction`: Compares original and reconstructed time series

## Attention mechanism

In the case of multi-variate features, the CAEVizT implements an _attention layer_ at the end of the encoder part. This lets the model focus on the most important features and time steps in your multivariate time series data. It works by learning weights that indicate how much "attention" to pay to different parts of the input.

**How It Works**

- Dense layer with tanh activation: Creates a hidden representation to compute attention scores
- Flatten and softmax: Normalizes the attention scores so they sum to 1
- Reshape: Returns the weights to the original temporal dimension
- Multiply: Applies the attention weights to the input features

**Why ?**

- Feature Importance: In multivariate time series, not all features are equally important. Attention helps identify and focus on the most significant features.

- Temporal Focus: Some time steps contain more valuable information than others. Attention helps the model distinguish between informative and less informative time periods.

- Interpretability: The attention weights can be visualized to understand which features and time steps the model is focusing on, adding transparency to your model.

- Improved Performance: By focusing on the most relevant parts of the input, attention mechanisms often lead to better performance, especially for complex patterns.

- Handling Variable Information Density: In real-world time series, information is not uniformly distributed. Attention helps to efficiently allocate computational resources to the most information-rich segments.

# Usage Example

The artifact includes a comprehensive example at the bottom that demonstrates:

- Generating synthetic time series data with three distinct patterns
- Training the model on this data
- Visualizing the results
- Evaluating clustering performance