"""
Neural Network AI System for VALORANT Tactical Decisions

This module implements a proper neural network-based AI that:
1. Learns from real player movement and decision data
2. Makes decisions through neural network inference
3. Adapts to individual player styles

Architecture:
- Input Layer: 64 features (game state encoding)
- Hidden Layer 1: 128 neurons (ReLU activation)
- Hidden Layer 2: 64 neurons (ReLU activation)
- Hidden Layer 3: 32 neurons (ReLU activation)
- Output Layer: Action probabilities (softmax)

Features (64 total):
- Position (x, y, normalized): 2
- Velocity direction (sin, cos): 2
- Health & armor (normalized): 2
- Time pressure: 1
- Role encoding (one-hot, 5 roles): 5
- Team (attack/defense): 1
- Spike state (has_spike, planted, defusing): 3
- Knowledge state (enemies seen, uncertainty): 5
- Nearby enemy features (closest 3 enemies x 6 features): 18
- Nearby teammate features (closest 2 teammates x 6 features): 12
- Site distances: 3
- Utility state (flash, smoke, molly): 3
- Movement state (running, moving): 2
- Time since last action: 1
- Phase encoding (one-hot, 4 phases): 4

Actions (10 total):
- HOLD, ADVANCE, RETREAT, EXECUTE, PLANT, DEFUSE, ROTATE, PEEK, TRADE, LURK
"""

import math
import random
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum


class Action(Enum):
    """Possible actions the neural network can output."""
    HOLD = 0
    ADVANCE = 1
    RETREAT = 2
    EXECUTE = 3
    PLANT = 4
    DEFUSE = 5
    ROTATE = 6
    PEEK = 7
    TRADE = 8
    LURK = 9


@dataclass
class GameStateFeatures:
    """
    Raw game state that will be converted to neural network features.
    """
    # Player position
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)

    # Health & Economy
    health: int = 100
    armor: int = 0

    # Time
    time_ms: int = 0
    round_time_ms: int = 100000

    # Role & Team
    role: str = "support"
    team: str = "attack"

    # Spike
    has_spike: bool = False
    spike_planted: bool = False
    is_planting: bool = False
    is_defusing: bool = False

    # Knowledge
    enemies_seen: int = 0
    enemies_confirmed_dead: int = 0
    uncertainty: float = 0.5  # How uncertain about enemy positions
    threat_level: float = 0.5  # Perceived threat at current position

    # Nearby entities (up to 3 enemies, 2 teammates)
    nearby_enemies: List[Dict] = field(default_factory=list)
    nearby_teammates: List[Dict] = field(default_factory=list)

    # Site distances
    distance_to_site_a: float = 1.0
    distance_to_site_b: float = 1.0
    distance_to_site_c: float = 1.0  # For 3-site maps

    # Utility
    has_flash: bool = False
    has_smoke: bool = False
    has_molly: bool = False

    # Movement
    is_running: bool = True
    is_moving: bool = False

    # Phase
    phase: str = "mid_round"

    # Time since last action change
    time_since_last_action_ms: int = 0


class NeuralNetwork:
    """
    A simple feedforward neural network implemented in pure Python.

    This can be trained on real player data or used with pre-initialized
    weights based on known good strategies.
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        Initialize neural network with given layer sizes.

        Args:
            layer_sizes: List of neurons per layer [input, hidden1, ..., output]
            learning_rate: Learning rate for training
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases with Xavier initialization
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier initialization for better gradient flow
            limit = math.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])]
                 for _ in range(layer_sizes[i])]
            b = [0.0 for _ in range(layer_sizes[i + 1])]
            self.weights.append(w)
            self.biases.append(b)

        # Cache for backpropagation
        self._layer_outputs = []
        self._layer_inputs = []

    def relu(self, x: float) -> float:
        """ReLU activation function."""
        return max(0, x)

    def relu_derivative(self, x: float) -> float:
        """Derivative of ReLU."""
        return 1.0 if x > 0 else 0.0

    def softmax(self, values: List[float]) -> List[float]:
        """Softmax activation for output layer."""
        # Numerical stability: subtract max
        max_val = max(values)
        exp_values = [math.exp(v - max_val) for v in values]
        total = sum(exp_values)
        return [e / total for e in exp_values]

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward pass through the network.

        Args:
            inputs: Input feature vector

        Returns:
            Output probabilities for each action
        """
        self._layer_outputs = [inputs]
        self._layer_inputs = []

        current = inputs

        for i in range(self.num_layers - 1):
            # Matrix multiplication: current @ weights[i] + biases[i]
            layer_input = []
            for j in range(self.layer_sizes[i + 1]):
                s = self.biases[i][j]
                for k in range(len(current)):
                    s += current[k] * self.weights[i][k][j]
                layer_input.append(s)

            self._layer_inputs.append(layer_input)

            # Apply activation
            if i < self.num_layers - 2:
                # Hidden layers use ReLU
                current = [self.relu(x) for x in layer_input]
            else:
                # Output layer uses softmax
                current = self.softmax(layer_input)

            self._layer_outputs.append(current)

        return current

    def backward(self, target: List[float]) -> float:
        """
        Backward pass for training.

        Args:
            target: One-hot encoded target action

        Returns:
            Loss value
        """
        output = self._layer_outputs[-1]

        # Cross-entropy loss
        loss = 0
        for i in range(len(output)):
            if target[i] > 0:
                loss -= target[i] * math.log(max(output[i], 1e-10))

        # Output layer gradients (softmax + cross-entropy simplifies)
        output_gradients = [output[i] - target[i] for i in range(len(output))]

        # Backpropagate through layers
        gradients = output_gradients

        for layer in range(self.num_layers - 2, -1, -1):
            prev_output = self._layer_outputs[layer]
            layer_input = self._layer_inputs[layer]

            # Compute weight gradients and update
            new_gradients = [0.0] * len(prev_output)

            for j in range(len(gradients)):
                # For hidden layers, apply ReLU derivative
                if layer < self.num_layers - 2:
                    d = gradients[j] * self.relu_derivative(layer_input[j])
                else:
                    d = gradients[j]

                # Update bias
                self.biases[layer][j] -= self.learning_rate * d

                # Update weights and accumulate gradients for next layer
                for k in range(len(prev_output)):
                    self.weights[layer][k][j] -= self.learning_rate * d * prev_output[k]
                    new_gradients[k] += d * self.weights[layer][k][j]

            gradients = new_gradients

        return loss

    def train_step(self, inputs: List[float], target: List[float]) -> float:
        """Single training step."""
        self.forward(inputs)
        return self.backward(target)

    def save_weights(self, filepath: str):
        """Save weights to JSON file."""
        data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_weights(self, filepath: str):
        """Load weights from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.layer_sizes = data['layer_sizes']
        self.weights = data['weights']
        self.biases = data['biases']
        self.num_layers = len(self.layer_sizes)


class FeatureExtractor:
    """
    Extracts neural network features from game state.

    Converts raw game state into a normalized feature vector
    suitable for neural network input.
    """

    FEATURE_SIZE = 64
    NUM_ENEMY_FEATURES = 6  # per enemy
    NUM_TEAMMATE_FEATURES = 6  # per teammate
    MAX_ENEMIES = 3
    MAX_TEAMMATES = 2

    # Role encoding
    ROLES = ['entry', 'support', 'controller', 'sentinel', 'lurk']

    # Phase encoding
    PHASES = ['opening', 'mid_round', 'post_plant', 'retake']

    def extract(self, state: GameStateFeatures) -> List[float]:
        """
        Extract feature vector from game state.

        Returns:
            64-dimensional feature vector, all values normalized to [0, 1] or [-1, 1]
        """
        features = []

        # 1. Position (2 features)
        features.append(state.position[0])  # x normalized 0-1
        features.append(state.position[1])  # y normalized 0-1

        # 2. Velocity direction (2 features)
        vx, vy = state.velocity
        v_mag = math.sqrt(vx*vx + vy*vy) + 1e-6
        features.append(vx / v_mag)  # normalized direction x
        features.append(vy / v_mag)  # normalized direction y

        # 3. Health & Armor (2 features)
        features.append(state.health / 100.0)
        features.append(state.armor / 50.0)

        # 4. Time pressure (1 feature)
        features.append(state.time_ms / state.round_time_ms)

        # 5. Role encoding (5 features, one-hot)
        role_lower = state.role.lower()
        for role in self.ROLES:
            features.append(1.0 if role == role_lower else 0.0)

        # 6. Team (1 feature)
        features.append(1.0 if state.team == 'attack' else 0.0)

        # 7. Spike state (3 features)
        features.append(1.0 if state.has_spike else 0.0)
        features.append(1.0 if state.spike_planted else 0.0)
        features.append(1.0 if state.is_planting or state.is_defusing else 0.0)

        # 8. Knowledge state (5 features)
        features.append(state.enemies_seen / 5.0)
        features.append(state.enemies_confirmed_dead / 5.0)
        features.append(state.uncertainty)
        features.append(state.threat_level)
        # Alive enemies estimation
        features.append((5 - state.enemies_confirmed_dead - state.enemies_seen) / 5.0)

        # 9. Nearby enemies (18 features: 3 enemies x 6 features each)
        for i in range(self.MAX_ENEMIES):
            if i < len(state.nearby_enemies):
                enemy = state.nearby_enemies[i]
                # Distance (normalized)
                features.append(min(enemy.get('distance', 1.0), 1.0))
                # Angle to enemy (sin, cos for continuity)
                angle = enemy.get('angle', 0)
                features.append(math.sin(angle))
                features.append(math.cos(angle))
                # Confidence in position (0 = stale, 1 = exact)
                features.append(enemy.get('confidence', 0.5))
                # Is visible now
                features.append(1.0 if enemy.get('visible', False) else 0.0)
                # Time since seen (normalized)
                features.append(min(enemy.get('time_since_seen', 10000), 10000) / 10000)
            else:
                # No enemy in this slot
                features.extend([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

        # 10. Nearby teammates (12 features: 2 teammates x 6 features each)
        for i in range(self.MAX_TEAMMATES):
            if i < len(state.nearby_teammates):
                teammate = state.nearby_teammates[i]
                features.append(min(teammate.get('distance', 1.0), 1.0))
                angle = teammate.get('angle', 0)
                features.append(math.sin(angle))
                features.append(math.cos(angle))
                features.append(teammate.get('health', 100) / 100.0)
                features.append(1.0 if teammate.get('has_util', False) else 0.0)
                features.append(1.0 if teammate.get('is_alive', True) else 0.0)
            else:
                features.extend([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])

        # 11. Site distances (3 features)
        features.append(min(state.distance_to_site_a, 1.0))
        features.append(min(state.distance_to_site_b, 1.0))
        features.append(min(state.distance_to_site_c, 1.0))

        # 12. Utility (3 features)
        features.append(1.0 if state.has_flash else 0.0)
        features.append(1.0 if state.has_smoke else 0.0)
        features.append(1.0 if state.has_molly else 0.0)

        # 13. Movement state (2 features)
        features.append(1.0 if state.is_running else 0.0)
        features.append(1.0 if state.is_moving else 0.0)

        # 14. Time since last action (1 feature)
        features.append(min(state.time_since_last_action_ms, 5000) / 5000)

        # 15. Phase encoding (4 features, one-hot)
        phase_lower = state.phase.lower()
        for phase in self.PHASES:
            features.append(1.0 if phase == phase_lower else 0.0)

        # Ensure we have exactly 64 features
        assert len(features) == self.FEATURE_SIZE, f"Expected {self.FEATURE_SIZE} features, got {len(features)}"

        return features


class NeuralAIPlayer:
    """
    Neural network-based AI for individual player decisions.

    Each player can have their own trained network, allowing
    for diverse play styles that emerge from training data.
    """

    def __init__(self, player_id: str, learning_mode: bool = False):
        """
        Initialize neural AI for a player.

        Args:
            player_id: Unique player identifier
            learning_mode: If True, network learns from observations
        """
        self.player_id = player_id
        self.learning_mode = learning_mode

        # Network architecture: 64 -> 128 -> 64 -> 32 -> 10
        self.network = NeuralNetwork(
            layer_sizes=[64, 128, 64, 32, 10],
            learning_rate=0.0005
        )

        self.feature_extractor = FeatureExtractor()

        # Experience replay buffer for training
        self.experience_buffer: List[Tuple[List[float], Action]] = []
        self.buffer_size = 1000

        # Decision history for analysis
        self.decision_history: List[Tuple[int, Action, List[float]]] = []

        # Temperature for action selection (lower = more deterministic)
        self.temperature = 0.3

        # Initialize with strategic priors
        self._initialize_strategic_priors()

    def _initialize_strategic_priors(self):
        """
        Initialize network with basic strategic knowledge.

        This gives the network reasonable starting behavior
        before any real training data is available.
        """
        # Generate synthetic training examples based on known good strategies
        strategic_examples = [
            # Low health -> retreat
            (self._create_state(health=25, threat_level=0.8), Action.RETREAT),
            (self._create_state(health=20, threat_level=0.6), Action.RETREAT),
            # High health, enemy nearby -> peek
            (self._create_state(health=100, enemies_nearby=True, threat_level=0.4), Action.PEEK),
            (self._create_state(health=80, enemies_nearby=True, threat_level=0.5), Action.PEEK),
            # At site with spike, site clear -> plant
            (self._create_state(has_spike=True, at_site=True, threat_level=0.2), Action.PLANT),
            (self._create_state(has_spike=True, at_site=True, threat_level=0.1), Action.PLANT),
            # Spike planted, defender -> advance/defuse
            (self._create_state(team='defense', spike_planted=True, at_site=True), Action.DEFUSE),
            (self._create_state(team='defense', spike_planted=True, threat_level=0.3), Action.ADVANCE),
            # Early round, attack -> advance
            (self._create_state(team='attack', time_ratio=0.1, threat_level=0.3), Action.ADVANCE),
            (self._create_state(team='attack', time_ratio=0.2, threat_level=0.4), Action.ADVANCE),
            # Late round, no spike planted, attack -> execute
            (self._create_state(team='attack', time_ratio=0.7, has_spike=True), Action.EXECUTE),
            # Defender, unknown enemies -> hold
            (self._create_state(team='defense', uncertainty=0.8), Action.HOLD),
            (self._create_state(team='defense', uncertainty=0.6, at_site=True), Action.HOLD),
            # Man advantage -> advance
            (self._create_state(teammates_alive=4, enemies_alive=2), Action.ADVANCE),
            # Man disadvantage -> play safe
            (self._create_state(teammates_alive=2, enemies_alive=4), Action.HOLD),
            (self._create_state(teammates_alive=1, enemies_alive=3), Action.RETREAT),
        ]

        # Train on strategic priors (reduced passes for faster initialization)
        for features, action in strategic_examples:
            target = [0.0] * 10
            target[action.value] = 1.0
            # Few passes to establish basic behavior
            for _ in range(5):
                self.network.train_step(features, target)

    def _create_state(self, health: int = 100, threat_level: float = 0.5,
                      has_spike: bool = False, at_site: bool = False,
                      spike_planted: bool = False, team: str = 'attack',
                      time_ratio: float = 0.5, uncertainty: float = 0.5,
                      enemies_nearby: bool = False, teammates_alive: int = 5,
                      enemies_alive: int = 5) -> List[float]:
        """Create synthetic feature vector for training."""
        state = GameStateFeatures(
            position=(0.5, 0.5),
            health=health,
            armor=25,
            time_ms=int(time_ratio * 100000),
            round_time_ms=100000,
            role='support',
            team=team,
            has_spike=has_spike,
            spike_planted=spike_planted,
            uncertainty=uncertainty,
            threat_level=threat_level,
            enemies_seen=5 - enemies_alive if enemies_nearby else 0,
            enemies_confirmed_dead=5 - enemies_alive,
            nearby_enemies=[{'distance': 0.15, 'visible': True, 'confidence': 0.9}] if enemies_nearby else [],
            distance_to_site_a=0.05 if at_site else 0.3,
            distance_to_site_b=0.4,
            is_moving=True
        )
        return self.feature_extractor.extract(state)

    def decide(self, state: GameStateFeatures) -> Tuple[Action, List[float]]:
        """
        Make a decision based on current game state.

        Args:
            state: Current game state

        Returns:
            Tuple of (chosen_action, action_probabilities)
        """
        # Extract features
        features = self.feature_extractor.extract(state)

        # Forward pass through network
        probabilities = self.network.forward(features)

        # Apply valid action mask
        valid_probs = self._apply_action_mask(state, probabilities)

        # Select action based on temperature
        action = self._select_action(valid_probs)

        # Record for analysis
        self.decision_history.append((state.time_ms, action, probabilities))

        return action, valid_probs

    def _apply_action_mask(self, state: GameStateFeatures,
                           probs: List[float]) -> List[float]:
        """
        Mask out invalid actions for current state.

        This ensures the network doesn't select impossible actions
        like planting when not holding spike.
        """
        masked = probs.copy()

        # Can't plant without spike or if not at site
        if not state.has_spike or state.distance_to_site_a > 0.1:
            masked[Action.PLANT.value] = 0.0

        # Can't defuse if spike not planted
        if not state.spike_planted:
            masked[Action.DEFUSE.value] = 0.0

        # Attackers can't defuse, defenders can't plant
        if state.team == 'attack':
            masked[Action.DEFUSE.value] = 0.0
        else:
            masked[Action.PLANT.value] = 0.0

        # Can't execute as single player (team action)
        # This is handled at team level

        # Renormalize
        total = sum(masked)
        if total > 0:
            masked = [p / total for p in masked]
        else:
            # Fallback to hold if all masked
            masked = [0.0] * 10
            masked[Action.HOLD.value] = 1.0

        return masked

    def _select_action(self, probs: List[float]) -> Action:
        """
        Select action using temperature-scaled softmax sampling.

        Lower temperature = more deterministic (picks highest prob)
        Higher temperature = more exploratory
        """
        if self.temperature < 0.01:
            # Deterministic - pick highest
            max_idx = probs.index(max(probs))
            return Action(max_idx)

        # Temperature scaling
        scaled = [p ** (1 / self.temperature) for p in probs]
        total = sum(scaled)
        if total > 0:
            scaled = [p / total for p in scaled]
        else:
            return Action.HOLD

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(scaled):
            cumulative += p
            if r <= cumulative:
                return Action(i)

        return Action(len(scaled) - 1)

    def learn_from_observation(self, state: GameStateFeatures, action: Action,
                               reward: float = 1.0):
        """
        Learn from an observed action (e.g., from real player data).

        Args:
            state: Game state when action was taken
            action: The action that was taken
            reward: How good the action was (1.0 = good, 0.0 = bad)
        """
        if not self.learning_mode:
            return

        features = self.feature_extractor.extract(state)

        # Add to experience buffer
        self.experience_buffer.append((features, action))
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

        # Train on this example
        target = [0.0] * 10
        target[action.value] = reward

        # Also add small probability to similar actions
        if action in [Action.ADVANCE, Action.EXECUTE]:
            target[Action.ADVANCE.value] = max(target[Action.ADVANCE.value], 0.2)
            target[Action.EXECUTE.value] = max(target[Action.EXECUTE.value], 0.2)
        if action in [Action.HOLD, Action.RETREAT]:
            target[Action.HOLD.value] = max(target[Action.HOLD.value], 0.2)

        # Normalize
        total = sum(target)
        target = [t / total for t in target]

        self.network.train_step(features, target)

    def batch_train(self, batch_size: int = 32) -> float:
        """
        Train on a random batch from experience buffer.

        Returns average loss.
        """
        if len(self.experience_buffer) < batch_size:
            return 0.0

        batch = random.sample(self.experience_buffer, batch_size)
        total_loss = 0.0

        for features, action in batch:
            target = [0.0] * 10
            target[action.value] = 1.0
            loss = self.network.train_step(features, target)
            total_loss += loss

        return total_loss / batch_size


class NeuralAISystem:
    """
    Manages neural network AI for all players in simulation.

    Uses a shared network for efficient inference - all players share
    the same trained weights by default. This avoids the overhead of
    training strategic priors for each player separately.
    """

    # Shared network for all players (class variable)
    _shared_network: Optional[NeuralNetwork] = None
    _shared_initialized: bool = False

    def __init__(self, learning_mode: bool = False, use_shared: bool = True):
        """
        Initialize neural AI system.

        Args:
            learning_mode: If True, AIs learn from gameplay
            use_shared: If True, use shared network for all players (faster)
        """
        self.learning_mode = learning_mode
        self.use_shared = use_shared
        self.player_ais: Dict[str, NeuralAIPlayer] = {}
        self.feature_extractor = FeatureExtractor()

        # Initialize shared network once
        if use_shared and not NeuralAISystem._shared_initialized:
            self._initialize_shared_network()

    @classmethod
    def _initialize_shared_network(cls):
        """Initialize the shared network with strategic priors (once)."""
        if cls._shared_initialized:
            return

        # Create and train shared network
        cls._shared_network = NeuralNetwork(
            layer_sizes=[64, 128, 64, 32, 10],
            learning_rate=0.0005
        )

        # Train with strategic priors
        fe = FeatureExtractor()
        strategic_examples = cls._get_strategic_examples(fe)

        for features, action in strategic_examples:
            target = [0.0] * 10
            target[action.value] = 1.0
            for _ in range(5):
                cls._shared_network.train_step(features, target)

        cls._shared_initialized = True

    @staticmethod
    def _get_strategic_examples(fe: FeatureExtractor) -> List[Tuple[List[float], Action]]:
        """Generate strategic training examples."""
        examples = []

        def make_state(**kwargs) -> List[float]:
            state = GameStateFeatures(position=(0.5, 0.5), **kwargs)
            return fe.extract(state)

        # Low health -> retreat
        examples.append((make_state(health=25, team='attack'), Action.RETREAT))
        examples.append((make_state(health=20, team='defense'), Action.RETREAT))

        # High health, near enemy -> peek
        examples.append((make_state(health=100, team='attack',
            nearby_enemies=[{'distance': 0.15, 'visible': True, 'confidence': 0.9, 'angle': 0, 'time_since_seen': 0}]),
            Action.PEEK))

        # At site with spike -> plant
        examples.append((make_state(has_spike=True, distance_to_site_a=0.05, team='attack'), Action.PLANT))

        # Spike planted, defender -> defuse
        examples.append((make_state(spike_planted=True, distance_to_site_a=0.05, team='defense'), Action.DEFUSE))

        # Early round attack -> advance
        examples.append((make_state(time_ms=10000, team='attack'), Action.ADVANCE))

        # Late round with spike -> execute
        examples.append((make_state(time_ms=70000, has_spike=True, team='attack'), Action.EXECUTE))

        # Defender, uncertainty -> hold
        examples.append((make_state(team='defense', uncertainty=0.8), Action.HOLD))

        return examples

    def get_or_create_ai(self, player_id: str) -> NeuralAIPlayer:
        """Get or create AI for a player."""
        if player_id not in self.player_ais:
            # Create lightweight player AI that uses shared network
            ai = NeuralAIPlayer.__new__(NeuralAIPlayer)
            ai.player_id = player_id
            ai.learning_mode = self.learning_mode
            ai.feature_extractor = self.feature_extractor
            ai.experience_buffer = []
            ai.buffer_size = 1000
            ai.decision_history = []
            ai.temperature = 0.3

            # Use shared network instead of creating new one
            if self.use_shared and NeuralAISystem._shared_network:
                ai.network = NeuralAISystem._shared_network
            else:
                # Create individual network (slower)
                ai.network = NeuralNetwork(layer_sizes=[64, 128, 64, 32, 10])
                ai._initialize_strategic_priors()

            self.player_ais[player_id] = ai
        return self.player_ais[player_id]

    def make_decision(self, player_id: str,
                      state: GameStateFeatures) -> Tuple[Action, List[float]]:
        """
        Make decision for a player using neural network.

        Returns:
            Tuple of (action, probabilities)
        """
        ai = self.get_or_create_ai(player_id)
        return ai.decide(state)

    def convert_to_legacy_decision(self, action: Action) -> str:
        """
        Convert neural action to legacy Decision enum value.

        For compatibility with existing simulation code.
        """
        action_map = {
            Action.HOLD: 'hold',
            Action.ADVANCE: 'advance',
            Action.RETREAT: 'retreat',
            Action.EXECUTE: 'execute',
            Action.PLANT: 'plant',
            Action.DEFUSE: 'defuse',
            Action.ROTATE: 'rotate',
            Action.PEEK: 'peek',
            Action.TRADE: 'trade',
            Action.LURK: 'lurk',
        }
        return action_map.get(action, 'hold')

    def learn_from_round(self, player_decisions: List[Dict], round_outcome: str):
        """
        Learn from a completed round.

        Args:
            player_decisions: List of {player_id, state, action, timestamp}
            round_outcome: 'attack_win' or 'defense_win'
        """
        if not self.learning_mode:
            return

        for decision in player_decisions:
            player_id = decision['player_id']
            state = decision['state']
            action = decision['action']

            ai = self.get_or_create_ai(player_id)

            # Calculate reward based on outcome
            is_attack = state.team == 'attack'
            won = (round_outcome == 'attack_win' and is_attack) or \
                  (round_outcome == 'defense_win' and not is_attack)

            reward = 1.0 if won else 0.3

            ai.learn_from_observation(state, action, reward)

    def save_all_models(self, directory: str):
        """Save all player models to directory."""
        import os
        os.makedirs(directory, exist_ok=True)

        for player_id, ai in self.player_ais.items():
            filepath = os.path.join(directory, f"{player_id}_model.json")
            ai.network.save_weights(filepath)

    def load_model(self, player_id: str, filepath: str):
        """Load a pre-trained model for a player."""
        ai = self.get_or_create_ai(player_id)
        ai.network.load_weights(filepath)
