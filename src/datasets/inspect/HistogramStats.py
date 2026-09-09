"""Class to manage histogram on datastes. Used to avoid many processing when simple graphical changes are applied to plots."""

import numpy as np

class HistogramStats:
    def __init__(self, bins, vmin, vmax):
        self.bins = bins
        self.vmin = vmin
        self.vmax = vmax

        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

        self.hist = np.zeros(bins, dtype=np.int64)
        self.edges = np.linspace(vmin, vmax, bins + 1)

    def update(self, values):
        values = np.atleast_1d(values)
        # Avoid division by zero if vmin == vmax
        range_val = self.vmax - self.vmin
        if range_val <= 0:
            range_val = 1e-9 

        for x in values:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.M2 += delta * (x - self.mean)

            if self.vmin <= x <= self.vmax:
                b = int((x - self.vmin) / range_val * self.bins)
                b = min(self.bins - 1, max(0, b))
                self.hist[b] += 1

    @property
    def std(self):
        # Welford's algorithm for running standard deviation
        return np.sqrt(self.M2 / (self.count - 1)) if self.count > 1 else 0.0

    def quantile(self, q):
        target = q * self.count
        cumsum = 0
        for i, c in enumerate(self.hist):
            cumsum += c
            if cumsum >= target:
                return 0.5 * (self.edges[i] + self.edges[i + 1])
        return self.vmax

    def summary(self):
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "q05": self.quantile(0.05),
            "q32": self.quantile(0.32),
            "q50": self.quantile(0.50),
            "q64": self.quantile(0.64),
            "q95": self.quantile(0.95),
            "histogram": {
                "bin_edges": self.edges.tolist(),
                "counts": self.hist.tolist(),
            },
        }