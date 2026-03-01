"""Neural network with sparse adaptive topology and genetic algorithm."""

import numpy as np


class NeuralNetwork:
    """Feedforward MLP with per-connection masks that evolve over generations.

    Genome layout per layer: [weights_flat | mask_flat | biases].
    Mask values are 0.0 (dormant) or 1.0 (active). New connections are
    activated by structural mutations in GeneticAlgorithm.
    """

    def __init__(self, layer_sizes: list[int], connection_rate: float = 1.0) -> None:
        """Initialize network with He weights and random sparse mask."""

        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / n_in)
            self.weights.append(np.random.randn(n_out, n_in) * scale)
            self.biases.append(np.zeros(n_out))
            self.masks.append(
                (np.random.random((n_out, n_in)) < connection_rate).astype(np.float32)
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass applying active connections only."""

        for w, mask, b in zip(self.weights, self.masks, self.biases):
            x = np.tanh((w * mask) @ x + b)
        return x

    def forward_with_activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Run forward pass and return activations for every layer including input."""

        acts = [x.copy()]
        for w, mask, b in zip(self.weights, self.masks, self.biases):
            x = np.tanh((w * mask) @ x + b)
            acts.append(x.copy())
        return acts

    def get_genome(self) -> np.ndarray:
        """Flatten all weights, masks and biases into a single genome vector."""

        parts = []
        for w, mask, b in zip(self.weights, self.masks, self.biases):
            parts.append(w.flatten())
            parts.append(mask.flatten())
            parts.append(b)
        return np.concatenate(parts)

    def set_genome(self, genome: np.ndarray) -> None:
        """Load weights, masks and biases from a flat genome vector."""

        idx = 0
        for i in range(len(self.weights)):
            size = self.weights[i].size
            n_b = self.biases[i].size
            self.weights[i] = genome[idx: idx + size].reshape(self.weights[i].shape)
            idx += size
            self.masks[i] = (genome[idx: idx + size] > 0.5).astype(np.float32).reshape(self.masks[i].shape)
            idx += size
            self.biases[i] = genome[idx: idx + n_b]
            idx += n_b

    @property
    def genome_size(self) -> int:
        """Total number of values in the genome (weights + masks + biases)."""

        return sum(w.size * 2 + b.size for w, b in zip(self.weights, self.biases))

    @property
    def active_connections(self) -> int:
        """Total number of currently active (unmasked) connections."""

        return int(sum(m.sum() for m in self.masks))

    @classmethod
    def from_genome(cls, layer_sizes: list[int], genome: np.ndarray) -> "NeuralNetwork":
        """Construct a network and restore all state from a flat genome."""

        nn = cls.__new__(cls)
        nn.layer_sizes = layer_sizes
        nn.weights = []
        nn.biases = []
        nn.masks = []

        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            nn.weights.append(np.empty((n_out, n_in)))
            nn.masks.append(np.empty((n_out, n_in), dtype=np.float32))
            nn.biases.append(np.empty(n_out))

        nn.set_genome(genome)
        return nn


class GeneticAlgorithm:
    """Genetic algorithm with weight mutation and structural connection growth."""

    def __init__(self, config: dict, layer_sizes: list[int]) -> None:
        """Initialize hyper-parameters and precompute mask slice positions in genome."""

        self.elite_count: int = config["elite_count"]
        self.mutation_rate: float = config["mutation_rate"]
        self.mutation_strength: float = config["mutation_strength"]
        self.crossover_rate: float = config["crossover_rate"]
        self.connection_add_rate: float = config.get("connection_add_rate", 0.02)
        self.explorer_mutation_strength: float = config.get("explorer_mutation_strength", 2.0)
        self.explorer_connection_add_rate: float = config.get("explorer_connection_add_rate", 0.2)

        self._mask_slices = self._compute_mask_slices(layer_sizes)

    @staticmethod
    def _compute_mask_slices(layer_sizes: list[int]) -> list[tuple[int, int]]:
        """Return (start, end) genome index ranges that correspond to mask entries."""

        slices = []
        idx = 0
        for i in range(len(layer_sizes) - 1):
            n_w = layer_sizes[i + 1] * layer_sizes[i]
            n_b = layer_sizes[i + 1]
            slices.append((idx + n_w, idx + 2 * n_w))
            idx += 2 * n_w + n_b
        return slices

    def evolve(
        self, genomes: list[np.ndarray], fitnesses: list[float]
    ) -> tuple[list[np.ndarray], int]:
        """Produce the next generation. Returns (genomes, explorer_index).

        The last slot is always the explorer: a heavily mutated copy of the
        best genome that tries tactics different from the rest of the population.
        """

        n = len(genomes)
        order = np.argsort(fitnesses)[::-1]
        new_genomes: list[np.ndarray] = []

        for i in range(min(self.elite_count, n)):
            new_genomes.append(genomes[order[i]].copy())

        # Fill n-1 slots with normal offspring; last slot reserved for explorer.
        while len(new_genomes) < n - 1:
            p1 = genomes[self._tournament(fitnesses)]
            p2 = genomes[self._tournament(fitnesses)]
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            new_genomes.append(child)

        # Explorer: take best genome and mutate aggressively.
        explorer = self._mutate_explorer(genomes[order[0]].copy())
        new_genomes.append(explorer)
        explorer_idx = n - 1

        return new_genomes, explorer_idx

    def _mutate_explorer(self, genome: np.ndarray) -> np.ndarray:
        """Apply aggressive weight and structural mutation for the explorer agent."""

        is_mask = np.zeros(len(genome), dtype=bool)
        for start, end in self._mask_slices:
            is_mask[start:end] = True

        mutate = ~is_mask & (np.random.random(len(genome)) < self.mutation_rate)
        genome += mutate * np.random.randn(len(genome)) * self.explorer_mutation_strength

        for start, end in self._mask_slices:
            seg = genome[start:end]
            dormant = seg < 0.5
            grow = dormant & (np.random.random(len(seg)) < self.explorer_connection_add_rate)
            seg[grow] = 1.0
            genome[start:end] = seg

        return genome

    def _tournament(self, fitnesses: list[float], k: int = 3) -> int:
        """Return the index of the fittest genome from a random k-tournament."""

        idx = np.random.choice(len(fitnesses), size=min(k, len(fitnesses)), replace=False)
        return int(idx[np.argmax([fitnesses[i] for i in idx])])

    def _crossover(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """Produce a child genome via uniform crossover."""

        if np.random.random() < self.crossover_rate:
            mask = np.random.random(len(g1)) < 0.5
            return np.where(mask, g1, g2)
        return g1.copy()

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Apply weight mutation and structural connection growth."""

        genome = genome.copy()

        # Mark which positions are mask entries (excluded from weight noise).
        is_mask = np.zeros(len(genome), dtype=bool)
        for start, end in self._mask_slices:
            is_mask[start:end] = True

        # Weight + bias mutation (skip mask positions).
        mutate = ~is_mask & (np.random.random(len(genome)) < self.mutation_rate)
        genome += mutate * np.random.randn(len(genome)) * self.mutation_strength

        # Structural mutation: activate dormant connections (0 → 1, never prune).
        for start, end in self._mask_slices:
            seg = genome[start:end]
            dormant = seg < 0.5
            grow = dormant & (np.random.random(len(seg)) < self.connection_add_rate)
            seg[grow] = 1.0
            genome[start:end] = seg

        return genome
