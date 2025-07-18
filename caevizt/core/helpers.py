from pandas import DataFrame
from numpy import array, random, full, where

def sequence_clusters_to_point_clusters(X, X_sequences, model, seq_len):
    """
    Convert sequence-based cluster assignments (output of CAEVizT's `predict_clusters` on sequences inputs) to point-based assignments (cluster assigned to last sequence point).
    
    Parameters:
    -----------
    X : numpy.ndarray
        Original time series data without sequencing
    X_sequences : numpy.ndarray
        Sequences created from X
    model : CAEVizT
        Trained CAEVizT model
    seq_len : int
        Length of each sequence
    
    Returns:
    --------
    point_labels : numpy.ndarray
        Cluster assignment for each point in the original time series
    """
    # Get cluster assignments for sequences
    sequence_clusters = model.predict_clusters(X_sequences)
    
    # Total time length in the original data
    n_sequences = len(X_sequences)
    time_length = len(X)
    
    # Initialize arrays for point classifications
    point_labels = full(time_length, -1)  # -1 means not yet assigned
    
    # Assign each point the cluster of the sequence where it appears last
    # This means points take the label of the most recent information
    for i in range(n_sequences):
        # Each sequence covers points from i to i+seq_len-1
        # We assign the cluster to the last point in the sequence
        last_point_idx = i + seq_len - 1
        if last_point_idx < time_length:
            point_labels[last_point_idx] = sequence_clusters[i]
    
    # Forward fill any points at the beginning that weren't assigned
    first_valid_idx = where(point_labels >= 0)[0][0]
    point_labels[:first_valid_idx] = point_labels[first_valid_idx]
    
    # Check if there are any remaining unassigned points and assign them
    if -1 in point_labels:
        # Backward fill for any points at the end
        last_valid_idx = where(point_labels >= 0)[0][-1]
        point_labels[last_valid_idx+1:] = point_labels[last_valid_idx]
    
    return point_labels

def shuffle(X, y=None):
    """
    Shuffle the data and labels in unison.
    
    Parameters:
    -----------
    X : array-like
        Input data to be shuffled.
    y : array-like, optional
        Labels corresponding to the data. If provided, will be shuffled in unison with X.
    
    Returns:
    --------
    X_shuffled : array
        Shuffled data.
    y_shuffled : array, optional
        Shuffled labels if y was provided.
    """
    indices = random.permutation(len(X))
    X_shuffled = X[indices]
    
    if y is not None:
        y_shuffled = y[indices]
        return X_shuffled, y_shuffled
    
    return X_shuffled

def make_sequences(X, seq_len=7, shuffle=True):
    """
    Converts a time series or multivariate time series into sequences of a specified length.

    Parameters:
    X : array-like
        Input time series data, can be a 1D array or a 2D array (multivariate).
    seq_len : int, optional
        Length of each sequence to be created from the time series. Default is 7.
    shuffle : bool, optional
        If True, the sequences will be shuffled. Default is True.
    
    Returns:
    sequences : array
        An array of sequences, where each sequence is of length `seq_len`.
    """
    if isinstance(X, DataFrame):
        X = X.values
    sequences = []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
    sequences = array(sequences)
    if shuffle:
        sequences = shuffle(sequences)
    return sequences

def unmake_sequences(X, seq_len=7):
    """
    Converts sequences back to the original time series format assuming they have not been shuffled.
    Reverts the operation done by `make_sequences` with shuffle=False.
    Parameters:
    X : array-like
        Input sequences, should be a 2D or 3D array where each row is a sequence.
    seq_len : int, optional
        Length of each sequence. Default is 7.
    Returns:
    unmade : array
        The original time series data reconstructed from the sequences.
    """
    if isinstance(X, DataFrame):
        X = X.values
    n_samples = X.shape[0]
    if len(X.shape) == 3:
        seq_len = X.shape[1]
    unmade = []
    for i in range(n_samples):
        unmade.append(X[i, 0])
    for i in range(seq_len):
        unmade.append(X[-1, i])
    unmade = array(unmade)
    return unmade