import numpy as np


class NeuralNetwork:
    """Feedforward MLP with tanh activation. Genome = flattened weights+biases."""

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * scale)
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for w, b in zip(self.weights, self.biases):
            x = np.tanh(w @ x + b)
        return x

    def forward_with_activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Returns activations for every layer including input."""
        acts = [x.copy()]
        for w, b in zip(self.weights, self.biases):
            x = np.tanh(w @ x + b)
            acts.append(x.copy())
        return acts

    def get_genome(self) -> np.ndarray:
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b)
        return np.concatenate(parts)

    def set_genome(self, genome: np.ndarray) -> None:
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            self.weights[i] = genome[idx: idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
            self.biases[i] = genome[idx: idx + b_size]
            idx += b_size

    @property
    def genome_size(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    @classmethod
    def from_genome(cls, layer_sizes: list[int], genome: np.ndarray) -> "NeuralNetwork":
        nn = cls.__new__(cls)
        nn.layer_sizes = layer_sizes
        nn.weights = []
        nn.biases = []
        for i in range(len(layer_sizes) - 1):
            nn.weights.append(np.empty((layer_sizes[i + 1], layer_sizes[i])))
            nn.biases.append(np.empty(layer_sizes[i + 1]))
        nn.set_genome(genome)
        return nn


class GeneticAlgorithm:
    """Genetic algorithm operating on flat numpy genome arrays."""

    def __init__(self, config: dict):
        self.elite_count: int = config["elite_count"]
        self.mutation_rate: float = config["mutation_rate"]
        self.mutation_strength: float = config["mutation_strength"]
        self.crossover_rate: float = config["crossover_rate"]

    def evolve(
        self, genomes: list[np.ndarray], fitnesses: list[float]
    ) -> list[np.ndarray]:
        n = len(genomes)
        order = np.argsort(fitnesses)[::-1]
        new_genomes: list[np.ndarray] = []

        # Elites survive unchanged
        for i in range(min(self.elite_count, n)):
            new_genomes.append(genomes[order[i]].copy())

        # Fill rest with crossover + mutation
        while len(new_genomes) < n:
            p1 = genomes[self._tournament(fitnesses)]
            p2 = genomes[self._tournament(fitnesses)]
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            new_genomes.append(child)

        return new_genomes

    def _tournament(self, fitnesses: list[float], k: int = 3) -> int:
        idx = np.random.choice(len(fitnesses), size=min(k, len(fitnesses)), replace=False)
        return int(idx[np.argmax([fitnesses[i] for i in idx])])

    def _crossover(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(len(g1)) < 0.5
            return np.where(mask, g1, g2)
        return g1.copy()

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = np.random.random(len(genome)) < self.mutation_rate
        noise = np.random.randn(len(genome)) * self.mutation_strength
        return genome + mask * noise
