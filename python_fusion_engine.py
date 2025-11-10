"""
Decision Fusion Engine - Adaptive Multi-Modal AI Trading System
Combines signals from 5 layers using Bayesian belief networks
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayerSignal:
    """Signal from individual layer"""
    layer_name: str
    score: float  # -1 (strong sell) to +1 (strong buy)
    confidence: float  # 0 to 1
    timestamp: float
    metadata: Dict


@dataclass
class FusedSignal:
    """Final trading signal after fusion"""
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    layer_scores: Dict[str, float]
    layer_weights: Dict[str, float]
    risk_params: Dict
    veto_triggered: bool
    reasoning: str


class AdaptiveFusionEngine:
    """
    Core decision engine that fuses signals from multiple AI layers
    using adaptive Bayesian belief networks
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initial layer weights (will adapt over time)
        self.layer_weights = {
            'social_sentiment': config.get('social_weight', 0.18),
            'news_nlp': config.get('news_weight', 0.22),
            'lstm_prediction': config.get('lstm_weight', 0.25),
            'ensemble_ml': config.get('ensemble_weight', 0.20),
            'technical': config.get('technical_weight', 0.15),
        }
        
        # Performance tracking for adaptive weighting
        self.layer_performance = {k: deque(maxlen=100) for k in self.layer_weights.keys()}
        
        # Veto thresholds
        self.veto_threshold = config.get('veto_threshold', -0.7)
        self.confidence_threshold = config.get('confidence_threshold', 0.70)
        
        # Bayesian priors
        self.market_regime_prior = 0.5  # Neutral prior
        self.update_count = 0
        
        logger.info(f"Fusion Engine initialized with weights: {self.layer_weights}")
    
    def fuse_signals(self, signals: List[LayerSignal]) -> FusedSignal:
        """
        Main fusion logic: Combines all layer signals into final decision
        
        Process:
        1. Check for veto signals (any layer <-0.7)
        2. Calculate weighted score
        3. Apply Bayesian belief update
        4. Compute confidence
        5. Return fused signal
        """
        
        if not signals or len(signals) == 0:
            return self._create_hold_signal("No signals available")
        
        # Convert signals to dict
        signal_dict = {s.layer_name: s for s in signals}
        
        # 1. VETO CHECK: Any strongly negative signal kills trade
        veto_check = self._check_veto(signal_dict)
        if veto_check['vetoed']:
            return self._create_hold_signal(
                f"Veto triggered by {veto_check['veto_layer']}: {veto_check['reason']}"
            )
        
        # 2. WEIGHTED SCORE: Combine all layer scores
        weighted_score = self._calculate_weighted_score(signal_dict)
        
        # 3. BAYESIAN UPDATE: Incorporate prior beliefs
        posterior_score = self._bayesian_update(weighted_score, signal_dict)
        
        # 4. CONFIDENCE CALCULATION: How much do layers agree?
        confidence = self._calculate_confidence(signal_dict, posterior_score)
        
        # 5. DECISION MAKING
        direction = self._determine_direction(posterior_score, confidence)
        
        # 6. RISK PARAMETERS
        risk_params = self._calculate_risk_params(posterior_score, confidence, signal_dict)
        
        return FusedSignal(
            direction=direction,
            confidence=confidence,
            layer_scores={k: v.score for k, v in signal_dict.items()},
            layer_weights=self.layer_weights.copy(),
            risk_params=risk_params,
            veto_triggered=False,
            reasoning=self._generate_reasoning(signal_dict, posterior_score, confidence)
        )
    
    def _check_veto(self, signals: Dict[str, LayerSignal]) -> Dict:
        """
        Veto system: If any layer has strong negative signal, block trade
        This prevents trading when one model sees major risk
        """
        for layer_name, signal in signals.items():
            if signal.score < self.veto_threshold:
                return {
                    'vetoed': True,
                    'veto_layer': layer_name,
                    'reason': f'Strong negative signal: {signal.score:.3f}'
                }
        return {'vetoed': False}
    
    def _calculate_weighted_score(self, signals: Dict[str, LayerSignal]) -> float:
        """
        Weighted average of all layer scores
        Layers with better historical performance get higher weight
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for layer_name, signal in signals.items():
            weight = self.layer_weights.get(layer_name, 0.0)
            weighted_sum += signal.score * weight * signal.confidence
            total_weight += weight * signal.confidence
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _bayesian_update(self, weighted_score: float, signals: Dict[str, LayerSignal]) -> float:
        """
        Bayesian belief update incorporating market regime prior
        
        P(Signal | Market) = P(Market | Signal) * P(Signal) / P(Market)
        """
        # Likelihood: How likely is this signal given market regime?
        likelihood = self._calculate_likelihood(weighted_score, signals)
        
        # Prior: Current belief about market state
        prior = self.market_regime_prior
        
        # Posterior: Updated belief
        # Using simplified Bayesian update (full would require MCMC)
        posterior = (likelihood * weighted_score + prior * 0.3) / 1.3
        
        # Update prior for next iteration
        self.market_regime_prior = 0.9 * self.market_regime_prior + 0.1 * posterior
        self.update_count += 1
        
        return np.clip(posterior, -1.0, 1.0)
    
    def _calculate_likelihood(self, score: float, signals: Dict[str, LayerSignal]) -> float:
        """
        Calculate likelihood of signal given current market conditions
        Uses layer agreement as proxy for likelihood
        """
        # Count how many layers agree with the direction
        direction = np.sign(score)
        agreeing_layers = sum(1 for s in signals.values() if np.sign(s.score) == direction)
        agreement_ratio = agreeing_layers / len(signals) if signals else 0.5
        
        # High agreement = high likelihood
        return 0.5 + 0.5 * agreement_ratio
    
    def _calculate_confidence(self, signals: Dict[str, LayerSignal], final_score: float) -> float:
        """
        Confidence based on:
        1. Layer agreement (how many agree on direction?)
        2. Signal strength (how strong are individual signals?)
        3. Confidence of individual layers
        """
        if not signals:
            return 0.0
        
        # 1. Direction agreement
        direction = np.sign(final_score)
        agreeing = [s for s in signals.values() if np.sign(s.score) == direction]
        agreement_score = len(agreeing) / len(signals)
        
        # 2. Signal strength
        avg_strength = np.mean([abs(s.score) for s in signals.values()])
        
        # 3. Layer confidence
        avg_confidence = np.mean([s.confidence for s in signals.values()])
        
        # Combine with weights
        overall_confidence = (
            0.4 * agreement_score +
            0.3 * avg_strength +
            0.3 * avg_confidence
        )
        
        return np.clip(overall_confidence, 0.0, 1.0)
    
    def _determine_direction(self, score: float, confidence: float) -> str:
        """Determine trade direction based on score and confidence"""
        if confidence < self.confidence_threshold:
            return 'HOLD'
        
        if score > 0.15:
            return 'BUY'
        elif score < -0.15:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_risk_params(
        self, 
        score: float, 
        confidence: float, 
        signals: Dict[str, LayerSignal]
    ) -> Dict:
        """
        Calculate position sizing, stop loss, take profit
        Risk scales with confidence
        """
        # Base risk (% of capital)
        base_risk = self.config.get('base_risk_percent', 0.01)
        
        # Adjust risk by confidence (higher confidence = larger position)
        adjusted_risk = base_risk * (0.5 + 0.5 * confidence)
        
        # Get volatility estimate from technical layer
        tech_signal = signals.get('technical')
        volatility_multiplier = 2.0
        if tech_signal and 'atr' in tech_signal.metadata:
            atr = tech_signal.metadata['atr']
            volatility_multiplier = max(1.5, min(3.0, atr / tech_signal.metadata.get('avg_atr', atr)))
        
        return {
            'position_size_pct': adjusted_risk,
            'stop_loss_atr_multiple': volatility_multiplier,
            'take_profit_atr_multiple': volatility_multiplier * 1.5,
            'max_hold_minutes': int(60 / confidence),  # Lower confidence = shorter hold
            'trailing_stop_enabled': confidence > 0.75
        }
    
    def _generate_reasoning(
        self, 
        signals: Dict[str, LayerSignal], 
        score: float, 
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for the decision"""
        lines = [f"Score: {score:.3f}, Confidence: {confidence:.1%}"]
        
        # Sort layers by signal strength
        sorted_layers = sorted(
            signals.items(), 
            key=lambda x: abs(x[1].score), 
            reverse=True
        )
        
        lines.append("\nLayer Contributions:")
        for layer_name, signal in sorted_layers[:3]:  # Top 3
            direction = "↑ Bullish" if signal.score > 0 else "↓ Bearish"
            lines.append(
                f"  • {layer_name}: {direction} ({signal.score:+.3f}, "
                f"confidence: {signal.confidence:.1%})"
            )
        
        return "\n".join(lines)
    
    def update_performance(self, layer_name: str, was_correct: bool):
        """
        Update layer performance tracking for adaptive weighting
        Called after trade closes to track which layers performed well
        """
        if layer_name in self.layer_performance:
            self.layer_performance[layer_name].append(1.0 if was_correct else 0.0)
            
            # Rebalance weights every 20 trades
            if self.update_count % 20 == 0:
                self._rebalance_weights()
    
    def _rebalance_weights(self):
        """
        Adaptive weighting: Increase weights of better-performing layers
        Uses exponential moving average of success rate
        """
        if not self.config.get('adaptive_weighting', True):
            return
        
        # Calculate performance scores
        performance_scores = {}
        for layer, history in self.layer_performance.items():
            if len(history) > 10:  # Need minimum history
                # EMA of success rate
                scores = list(history)
                ema = scores[0]
                for score in scores[1:]:
                    ema = 0.9 * ema + 0.1 * score
                performance_scores[layer] = ema
        
        if not performance_scores:
            return
        
        # Normalize to sum to 1.0
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            new_weights = {
                k: v / total_performance 
                for k, v in performance_scores.items()
            }
            
            # Smooth transition (70% old, 30% new)
            for layer in self.layer_weights:
                if layer in new_weights:
                    self.layer_weights[layer] = (
                        0.7 * self.layer_weights[layer] + 
                        0.3 * new_weights[layer]
                    )
            
            logger.info(f"Weights rebalanced: {self.layer_weights}")
    
    def _create_hold_signal(self, reason: str) -> FusedSignal:
        """Create a HOLD signal with reasoning"""
        return FusedSignal(
            direction='HOLD',
            confidence=0.0,
            layer_scores={},
            layer_weights=self.layer_weights.copy(),
            risk_params={},
            veto_triggered=True,
            reasoning=reason
        )
    
    def get_current_weights(self) -> Dict[str, float]:
        """Return current layer weights for monitoring"""
        return self.layer_weights.copy()
    
    def reset_priors(self):
        """Reset Bayesian priors (e.g., at market close/open)"""
        self.market_regime_prior = 0.5
        self.update_count = 0
        logger.info("Bayesian priors reset")


# Example usage
if __name__ == "__main__":
    # Initialize engine
    config = {
        'social_weight': 0.18,
        'news_weight': 0.22,
        'lstm_weight': 0.25,
        'ensemble_weight': 0.20,
        'technical_weight': 0.15,
        'veto_threshold': -0.7,
        'confidence_threshold': 0.70,
        'base_risk_percent': 0.01,
        'adaptive_weighting': True
    }
    
    engine = AdaptiveFusionEngine(config)
    
    # Example signals from different layers
    signals = [
        LayerSignal('social_sentiment', 0.65, 0.72, 1234567890, {'source': 'twitter'}),
        LayerSignal('news_nlp', 0.71, 0.85, 1234567890, {'headlines': 5}),
        LayerSignal('lstm_prediction', 0.82, 0.78, 1234567890, {'horizon': '15min'}),
        LayerSignal('ensemble_ml', 0.68, 0.80, 1234567890, {'models': 4}),
        LayerSignal('technical', 0.45, 0.65, 1234567890, {'atr': 25.5, 'avg_atr': 22.0}),
    ]
    
    # Fuse signals
    result = engine.fuse_signals(signals)
    
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Reasoning:\n{result.reasoning}")
    print(f"\nRisk Parameters:")
    for k, v in result.risk_params.items():
        print(f"  {k}: {v}")
