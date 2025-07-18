from numpy import ndarray, arange, unique
from typing import Tuple
from matplotlib.pyplot import figure, plot, fill_between, scatter, show, xlabel, ylabel, title, legend
from core.helpers import make_sequences, unmake_sequences

def spectrum_projection(X:ndarray, model, seq_len:int, scale:float=2.0) -> Tuple[ndarray, ndarray, ndarray]:
    """Project the original data and its sequences through the model to obtain spectrum projections.
    
    Parameters
    ----------
    X : ndarray
        The original data points.
    model : CAEVizT
        The trained CAEVizT model used for clustering.
    seq_len : int
        The length of the sequences used in the model.
    scale : float, optional
        The scale factor for the standard deviation used to create upper and lower bounds, by default 2.0.
    
    Returns
    -------
    tuple
        A tuple containing the projected sequences and their upper and lower bounds as (X_proj, X_proj_upper, X_proj_lower).
    """
    X_seq = make_sequences(X, seq_len=seq_len, shuffle=False)

    # Calculate standard deviation along sequence dimension
    seq_std = scale * X_seq.std(axis=1, keepdims=True)

    # Create upper and lower bound sequences
    X_seq_upper = X_seq + seq_std
    X_seq_lower = X_seq - seq_std

    # Project original and bound sequences through the model
    X_proj = model.predict(X_seq, verbose=0)
    X_proj_upper = model.predict(X_seq_upper, verbose=0)
    X_proj_lower = model.predict(X_seq_lower, verbose=0)

    return unmake_sequences(X_proj, seq_len), \
        unmake_sequences(X_proj_upper, seq_len), \
        unmake_sequences(X_proj_lower, seq_len)

def plot_spectrum_projection(X:ndarray,
                            X_pred:ndarray,
                            X_pred_upper:ndarray,
                            X_pred_lower:ndarray,
                            point_labels:ndarray,
                            len_to_plot:int=-1) -> None:
    """Plot the original data, predicted data, and confidence bands with cluster colors.
    Parameters
    ----------
    X : ndarray
        The original data points.
    X_pred : ndarray
        The predicted data points from the model.
    X_pred_upper : ndarray
        The upper bound of the predicted data points.
    X_pred_lower : ndarray
        The lower bound of the predicted data points.
    point_labels : ndarray
        The cluster labels for each point in the original data.
    len_to_plot : int, optional
        The number of points to plot from the beginning of the sequences, by default -1,(all).
    """
    if len_to_plot == -1:
        len_to_plot = X.shape[0]
    
    figure(figsize=(15, 7))

    # Plot original data and prediction
    plot(X[:len_to_plot], label='Original Data', alpha=0.7, color='#1f77b4')
    plot(X_pred[:len_to_plot], label='Predicted Data', alpha=0.7, color="#0812d2")

    # Plot upper and lower bounds as a shaded area
    fill_between(
        arange(len_to_plot),
        X_pred_lower[:len_to_plot, 0],
        X_pred_upper[:len_to_plot, 0],
        alpha=0.4,
        color="#0812d2",
        label="Confidence Band"
    )

    # Scatter points with cluster colors
    for cluster_id in unique(point_labels):
        mask = point_labels[:len_to_plot] == cluster_id
        indices = arange(len_to_plot)[mask]
        scatter(indices, X[:len_to_plot, 0][mask], 
                label=f'Cluster {cluster_id}', s=50, alpha=0.7)

    title('Original vs Predicted Data with Confidence Bands')
    xlabel('Time')
    ylabel('Value')
    legend()
    show()