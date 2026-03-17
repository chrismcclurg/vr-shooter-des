#CA McClurg
import numpy as np
import tensorflow as tf

class CallbackManager:
    def __init__(self, patience=15, min_delta=0.0005,
                 lr_patience=10, factor=0.5, min_lr=1e-6):
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.patience = patience
        self.patience_count = 0
        self.min_delta = min_delta
        self.lr_patience = lr_patience
        self.factor = factor
        self.min_lr = min_lr
        self.plateau_count = 0

    def on_epoch_end(self, model, optimizer, val_loss, f=None):
        stop = False

        if val_loss + self.min_delta < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = model.get_weights()
            self.patience_count = 0
            self.plateau_count = 0
        else:
            self.patience_count += 1
            self.plateau_count += 1

        # Early stopping
        if self.patience_count >= self.patience:
            log("=> Early stopping triggered", f)
            model.set_weights(self.best_weights)
            stop = True

        # Reduce LR on plateau
        if self.plateau_count >= self.lr_patience:
            old_lr = float(tf.keras.backend.get_value(optimizer.lr))
            new_lr = max(old_lr * self.factor, self.min_lr)
            tf.keras.backend.set_value(optimizer.lr, new_lr)
            log(f"=> LR reduced: {old_lr:.6f} -> {new_lr:.6f}", f)
            self.plateau_count = 0

        return stop

def log(message, f = None):
    print(message)
    if f is not None:
        f.write(message + "\n")

def get_acc(y_true, y_pred, eh_test):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eh_test = np.asarray(eh_test)

    acc = np.mean(y_true == y_pred)

    easy_mask = (eh_test == 0)
    hard_mask = (eh_test == 1)

    acc_easy = np.mean(y_true[easy_mask] == y_pred[easy_mask]) if np.any(easy_mask) else np.nan
    acc_hard = np.mean(y_true[hard_mask] == y_pred[hard_mask]) if np.any(hard_mask) else np.nan

    return (acc, acc_easy, acc_hard)

def get_acc_random(c_test, eh_test):
    def expected_acc(subset):
        counts = [len([xi for xi in neighs if xi != -1]) for neighs in subset]
        return np.mean([1.0 / count if count > 0 else 0.0 for count in counts]) if counts else np.nan

    rand = expected_acc(c_test)

    c_easy = [c for c, e in zip(c_test, eh_test) if e == 0]
    c_hard = [c for c, e in zip(c_test, eh_test) if e == 1]

    rand_easy = expected_acc(c_easy)
    rand_hard = expected_acc(c_hard)

    return (rand, rand_easy, rand_hard)

def build_opt(model, optimizer, loss_fn, x_sample, A_tensor, y_sample, c_sample, eh_sample):
    eh_sample = ensure_tensor(eh_sample)

    with tf.GradientTape() as tape:
        node_embs = model([x_sample, A_tensor], training=True)
        masked_neighbors = tf.where(c_sample == -1, 0, c_sample)
        edge_embs = tf.gather(node_embs, masked_neighbors, batch_dims=1)
        logits = tf.squeeze(model.classifier(edge_embs), axis=-1)
        loss = loss_fn(y_sample, logits, eh_sample)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
      
def ensure_tensor(x, dtype=tf.int32):
    """Ensure x is a TensorFlow tensor of the given dtype."""
    if x is not None and not isinstance(x, tf.Tensor):
        return tf.convert_to_tensor(x, dtype=dtype)
    return x

def make_weighted_loss(eh_train):
    eh_train = np.asarray(eh_train, dtype=np.int32).reshape(-1)
    n_easy = np.sum(eh_train == 0)
    n_hard = np.sum(eh_train == 1)
    total = n_easy + n_hard

    w_easy = total / (2.0 * max(1, n_easy))
    w_hard = total / (2.0 * max(1, n_hard))

    print(f"=> class weights: easy={w_easy:.3f}, hard={w_hard:.3f}")

    def weighted_ce(y_true, logits, eh_batch):
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True)
        weights = tf.where(tf.cast(eh_batch, tf.bool), w_hard, w_easy)
        return tf.reduce_mean(ce * tf.cast(weights, tf.float32))

    return weighted_ce
