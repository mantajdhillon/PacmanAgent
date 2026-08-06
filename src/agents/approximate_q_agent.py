import random
import numpy as np
from src.core.feature_extractor import FeatureExtractor

class ApproximateQAgent:
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9, epsilon: float = 0.1):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        self.extractor = FeatureExtractor()
        
        self.weights = None 

    def get_q_value(self, state, action) -> float:
        successor_state = state.generate_successor(0, action)
        
        features = self.extractor.get_q_features(state, successor_state)

        if self.weights is None:
            self.weights = np.zeros(len(features), dtype=np.float32)

        return float(np.dot(self.weights, features))

    def get_value(self, state) -> float:
        legal_actions = state.get_legal_actions(0)
        
        if not legal_actions:
            return 0.0

        return max(self.get_q_value(state, action) for action in legal_actions)

    def get_action(self, state) -> tuple:
        legal_actions = state.get_legal_actions(0)
        
        if not legal_actions:
            return (0, 0)

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        best_value = float('-inf')
        best_action = None

        for action in legal_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action if best_action is not None else random.choice(legal_actions)

    def update(self, state, action, next_state, reward: float):
        successor_state = state.generate_successor(0, action)
        features = self.extractor.get_q_features(state, successor_state)

        if self.weights is None:
            self.weights = np.zeros(len(features), dtype=np.float32)

        # Calculate the Temporal Difference (TD) error
        current_q = self.get_q_value(state, action)
        next_max_q = self.get_value(next_state)
        
        difference = (reward + self.gamma * next_max_q) - current_q

        # Update the weight vector
        self.weights += self.alpha * difference * features