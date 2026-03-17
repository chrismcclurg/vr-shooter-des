import numpy as np

class PrioritizedReplay:
    def __init__(self, capacity=200_000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, X, action, reward, next_X, done,
             r1_idx, r2_idx, r1_nbrs, r2_nbrs):

        # ensure Python lists, not numpy arrays
        r1_nbrs = list(r1_nbrs)
        r2_nbrs = list(r2_nbrs)

        data = (X, action, reward, next_X, done,
                int(r1_idx), int(r2_idx), r1_nbrs, r2_nbrs)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        max_prio = self.priorities[:len(self)].max(initial=1.0)
        self.priorities[self.pos] = max_prio


        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty replay buffer.")
    
        prios = self.priorities[:len(self.buffer)]
        prios = np.maximum(prios, 1e-6)
        probs = prios ** self.alpha
    
        # cumulative distribution
        cum = np.cumsum(probs)
        total_prob = cum[-1]
    
        # sample uniform points in [0, total_prob)
        rand_vals = np.random.rand(batch_size).astype(np.float32) * total_prob
    
        # find the indices using binary search
        indices = np.searchsorted(cum, rand_vals)
    
        # importance sampling weights
        weights = (len(self.buffer) * (probs[indices] / total_prob)) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
    
        # unpack samples
        Xs, actions, rewards, next_Xs, dones = [], [], [], [], []
        r1_idx, r2_idx, r1_nbrs, r2_nbrs = [], [], [], []
    
        for idx in indices:
            X, a, r, nxt, dn, i1, i2, n1, n2 = self.buffer[idx]
            Xs.append(X)
            actions.append(a)
            rewards.append(r)
            next_Xs.append(nxt)
            dones.append(dn)
            r1_idx.append(i1)
            r2_idx.append(i2)
            r1_nbrs.append(n1)
            r2_nbrs.append(n2)
    
        return (
            np.array(Xs, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_Xs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
            np.array(r1_idx, dtype=np.int32),
            np.array(r2_idx, dtype=np.int32),
            r1_nbrs,
            r2_nbrs,
        )


    def update_priorities(self, indices, new_prios):
        # Clip to avoid dead buffer (too-small) and destabilizing outliers (too-large)
        new_prios = np.clip(new_prios, 1e-6, 50.0)
    
        for idx, prio in zip(indices, new_prios):
            self.priorities[idx] = float(prio)

