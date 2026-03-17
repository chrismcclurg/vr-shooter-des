# -----------------------------------------------------------------------------
# CA McClurg
# Model definition for predicting shooter transitions
# -----------------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from spektral.layers import GraphSageConv
from src.gnn.common import make_weighted_loss
from src.utils.paths import GNN_MODEL_DIR, ensure_dir


class OneHeadPredictor(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        reg = tf.keras.regularizers.l2(1e-4)
        self.gcn1 = GraphSageConv(hidden_dim, activation='elu', kernel_regularizer=reg)
        self.gcn2 = GraphSageConv(hidden_dim, activation='elu', kernel_regularizer=reg)
        self.gcn3 = GraphSageConv(hidden_dim, activation='elu', kernel_regularizer=reg)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=reg)
        ])

    def encode(self, X, A, training=False):
        x1 = self.gcn1([X, A])
        x1 = self.dropout(x1, training=training)

        x2 = self.gcn2([x1, A])
        x2 = x2 + x1
        x2 = self.dropout(x2, training=training)

        x3 = self.gcn3([x2, A])
        x3 = x3 + x2
        x3 = self.dropout(x3, training=training)
        return x3

    def call(self, inputs, training=False):
        X, A = inputs
        return self.encode(X, A, training)

    def classify_edges(self, node_embs, c_batch):
        """Score candidate edges given node embeddings and candidate neighbors."""
        neighbors = tf.convert_to_tensor(c_batch, dtype=tf.int32)
        neighbors_clipped = tf.where(neighbors == -1, 0, neighbors)
        edge_embs = tf.gather(node_embs, neighbors_clipped, batch_dims=1)

        logits = tf.squeeze(self.classifier(edge_embs), axis=-1)
        return logits

@tf.function
def train(model, xTrain, A_tensor, yTrain, cTrain, optimizer, eh_mask, loss_fn, batch_size=32):
    total_loss = 0.0
    total_loss_easy = 0.0
    total_loss_hard = 0.0
    n_samples = tf.shape(xTrain)[0]
    num_batches = tf.cast(tf.math.ceil(tf.cast(n_samples, tf.float32) / batch_size), tf.int32)
    eh_mask = tf.convert_to_tensor(eh_mask, dtype=tf.int32)

    for i in tf.range(num_batches):
        start_idx = i * batch_size
        end_idx = tf.minimum(start_idx + batch_size, n_samples)

        x_batch = xTrain[start_idx:end_idx]
        y_batch = yTrain[start_idx:end_idx]
        c_batch = cTrain[start_idx:end_idx]
        eh_batch = eh_mask[start_idx:end_idx]
        A_batch = A_tensor

        with tf.GradientTape() as tape:
            node_embs = model([x_batch, A_batch], training=True)
            masked_neighbors = tf.where(c_batch == -1, 0, c_batch)
            edge_embs = tf.gather(node_embs, masked_neighbors, batch_dims=1)
            logits = tf.squeeze(model.classifier(edge_embs), axis=-1)

            # use pre-built weighted loss
            loss = loss_fn(y_batch, logits, eh_batch)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += loss

        # Optional: compute subgroup losses for monitoring
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_batch, logits, from_logits=True)
        total_loss_easy += tf.reduce_mean(ce[eh_batch == 0]) if tf.reduce_any(eh_batch == 0) else 0.0
        total_loss_hard += tf.reduce_mean(ce[eh_batch == 1]) if tf.reduce_any(eh_batch == 1) else 0.0

    loss = total_loss / tf.cast(num_batches, tf.float32)
    loss_e = total_loss_easy / tf.cast(num_batches, tf.float32)
    loss_h = total_loss_hard / tf.cast(num_batches, tf.float32)

    return loss, loss_e, loss_h

@tf.function
def validate(model, xValid, A_tensor, yValid, cValid, eh_mask, loss_fn, batch_size=32):
    total_loss = 0.0
    total_loss_easy = 0.0
    total_loss_hard = 0.0
    n_samples = tf.shape(xValid)[0]
    num_batches = tf.cast(tf.math.ceil(tf.cast(n_samples, tf.float32) / batch_size), tf.int32)
    eh_mask = tf.convert_to_tensor(eh_mask, dtype=tf.int32)

    for i in tf.range(num_batches):
        start_idx = i * batch_size
        end_idx = tf.minimum(start_idx + batch_size, n_samples)

        x_batch = xValid[start_idx:end_idx]
        y_batch = yValid[start_idx:end_idx]
        c_batch = cValid[start_idx:end_idx]
        eh_batch = eh_mask[start_idx:end_idx]
        A_batch = A_tensor

        # Forward pass
        node_embs = model([x_batch, A_batch], training=False)
        neighbors = tf.where(c_batch == -1, 0, c_batch)
        edge_embs = tf.gather(node_embs, neighbors, batch_dims=1)
        logits = tf.squeeze(model.classifier(edge_embs), axis=-1)

        # Weighted loss from custom loss_fn
        loss = loss_fn(y_batch, logits, eh_batch)

        total_loss += loss

        # Subgroup diagnostic losses (raw CE, unweighted)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_batch, logits, from_logits=True)
        total_loss_easy += tf.reduce_mean(ce[eh_batch == 0]) if tf.reduce_any(eh_batch == 0) else 0.0
        total_loss_hard += tf.reduce_mean(ce[eh_batch == 1]) if tf.reduce_any(eh_batch == 1) else 0.0

    loss = total_loss / tf.cast(num_batches, tf.float32)
    loss_e = total_loss_easy / tf.cast(num_batches, tf.float32)
    loss_h = total_loss_hard / tf.cast(num_batches, tf.float32)
    return loss, loss_e, loss_h

def test(model, xTest, A_tensor, yTest, cTest, eh_mask, batch_size=32):
    n_samples = len(xTest)
    yPred = []
    yTrue = []
    easy_idx = []
    hard_idx = []

    eh_mask = tf.convert_to_tensor(eh_mask, dtype=tf.int32)


    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        x_batch = tf.convert_to_tensor(xTest[start_idx:end_idx], dtype=tf.float32)
        c_batch = cTest[start_idx:end_idx]
        y_batch = yTest[start_idx:end_idx]

        eh_batch = eh_mask[start_idx:end_idx]
        A_batch = A_tensor

        # Forward pass
        node_embs = model([x_batch, A_batch], training=False)
        neighbors = tf.convert_to_tensor(c_batch, dtype=tf.int32)
        neighbors_clipped = tf.where(neighbors == -1, 0, neighbors)
        edge_embs = tf.gather(node_embs, neighbors_clipped, batch_dims=1)
        logits = tf.squeeze(model.classifier(edge_embs), axis=-1)

        # Mask invalid neighbors
        masked_logits = tf.where(neighbors != -1, logits, tf.fill(tf.shape(logits), -1e9))
        preds = tf.argmax(masked_logits, axis=-1)

        # Collect outputs
        y_pred = [c_batch[ix, xi] for ix, xi in enumerate(preds.numpy())]
        y_true = [c_batch[ix, xi] for ix, xi in enumerate(y_batch)]
        yPred.extend(y_pred)
        yTrue.extend(y_true)

        # Track easy/hard indices
        eh_batch_np = eh_batch.numpy()
        easy_idx.extend(np.where(eh_batch_np == 0)[0] + start_idx)
        hard_idx.extend(np.where(eh_batch_np == 1)[0] + start_idx)

    yPred = np.array(yPred)
    yTrue = np.array(yTrue)

    # Accuracies
    acc = np.mean(yPred == yTrue)
    acc_easy = np.mean(yPred[easy_idx] == yTrue[easy_idx]) if len(easy_idx) > 0 else 0.0
    acc_hard = np.mean(yPred[hard_idx] == yTrue[hard_idx]) if len(hard_idx) > 0 else 0.0

    return acc, acc_easy, acc_hard, yPred, yTrue

def create_model(hidden_dim, n_nodes, n_features, eh_train=None, path=None):

    model = OneHeadPredictor(hidden_dim)
    dummy_X = tf.zeros((1, n_nodes, n_features), dtype=tf.float32)
    dummy_A = tf.sparse.from_dense(tf.zeros((n_nodes, n_nodes), dtype=tf.float32))
    _ = model([dummy_X, dummy_A])
    _ = model.classifier(tf.zeros((1, 5, hidden_dim), dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam()
    if eh_train is not None:
        loss_fn = make_weighted_loss(eh_train)
        model.summary()
    else:
        loss_fn = None

    if path is not None:
        model.load_weights(path)
        # print(f"=> Loaded model weights from {path}")

    return model, optimizer, loss_fn

def get_info():
    ans = dict()
    model_name = "model03"
    base_dir = ensure_dir(GNN_MODEL_DIR / model_name)
    ans['name'] = model_name
    ans['ref'] = base_dir
    ans['features'] = ['dir_sim', 'recency', 'has_target', 'betweenness', 'is_entrance', 'is_outside']
    ans['weights'] = ensure_dir(base_dir / "train" / "weights")
    ans['logs']    = ensure_dir(base_dir / "train" / "logs")
    ans['curves']  = ensure_dir(base_dir / "train" / "curves")
    ans['output']  = ensure_dir(base_dir / "test" / "logs")
    ans['greedy']  = ensure_dir(base_dir / "greedy" / "logs")
    return ans

def neighbor_probs(model, x_current, A_sparse, node_order, recency_col=1, exclude_node=200):
    """
    Compute transition probabilities over neighbors of the current node.

    Parameters
    ----------
    model : OneHeadPredictor
        Trained GNN model with a single classifier head.
    x_current : np.ndarray or tf.Tensor, shape (n_nodes, n_features)
        Current feature state for all nodes.
    A_sparse : tf.SparseTensor
        Sparse adjacency matrix (shape n_nodes x n_nodes).
    node_order : list
        List mapping indices to node ids.
    recency_col : int
        Column index in x_current indicating recency (used to find current node).
    exclude_node : int
        Optional node id to exclude from selection (e.g., node 200 = outside).
    """
    x_current = tf.convert_to_tensor(x_current, dtype=tf.float32)
    shooter_idx = int(tf.argmax(x_current[:, recency_col]))  # Find most recent node

    # Convert adjacency to dense and get neighbors of shooter_idx
    A_dense = tf.sparse.to_dense(A_sparse)
    neighbors_idx = tf.where(A_dense[shooter_idx] == 1)
    neighbors_idx = neighbors_idx.numpy().flatten()

    if neighbors_idx.size == 0:
        # No neighbors — fallback to uniform or return zeros
        return np.ones(len(node_order)) / len(node_order)

    # Get node embeddings
    tf.config.run_functions_eagerly(True)
    node_emb = model([x_current, A_sparse], training=False)
    tf.config.run_functions_eagerly(False)

    # Compute logits for all nodes
    logits = tf.squeeze(model.classifier(node_emb), axis=-1)  # (N,)

    # Mask all non-neighbors to -inf
    valid_mask = np.full(len(node_order), -1e9)
    valid_mask[neighbors_idx] = 0

    # Optionally exclude a specific node (e.g., node 200)
    if exclude_node in node_order:
        exclude_idx = node_order.index(exclude_node)
        valid_mask[exclude_idx] = -1e9

    logits_with_mask = logits + valid_mask
    probs = tf.nn.softmax(logits_with_mask).numpy()

    # Renormalize to make sure neighbors sum to 1
    total_prob = probs[neighbors_idx].sum()
    if total_prob > 0:
        probs = probs / total_prob

    return probs