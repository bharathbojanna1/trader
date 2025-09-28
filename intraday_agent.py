import math
import random
import argparse
import json
import csv
from dataclasses import dataclass, asdict
from typing import Deque, List, Tuple, Optional, Dict, Any
from collections import deque
import time
from datetime import datetime, timedelta
import os

import numpy as np

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	from transformers import AutoTokenizer, AutoModel
	import pandas as pd
	except_import_error = None
except Exception as e:  # pragma: no cover
	except_import_error = e


@dataclass
class Config:
	starting_capital: float = 100000.0
	allow_leverage: bool = False
	trailing_stop_pct: float = 0.01  # 1% trail for winners
	transaction_brokerage_pct: float = 0.0005  # 5 bps
	transaction_slippage_bps_open_close: float = 4.0  # higher slippage at open/close
	transaction_slippage_bps_mid: float = 1.5
	rolling_vol_window: int = 20
	obs_lookback: int = 60  # timesteps for transformer context
	max_position_fraction: float = 1.0  # of NAV (no leverage)
	seed: int = 42
	device: str = "cuda" if torch.cuda.is_available() else "cpu"

	# SAC settings (lightweight)
	gamma: float = 0.99
	tau: float = 0.005
	alpha: float = 0.2  # Initial alpha, will be learned if auto_entropy=True
	auto_entropy: bool = True  # SAC++ automatic entropy tuning
	target_entropy: float = -1.0  # Target entropy (negative of action dim)
	alpha_lr: float = 3e-4  # Learning rate for entropy coefficient
	actor_lr: float = 3e-4
	critic_lr: float = 3e-4
	batch_size: int = 128
	replay_capacity: int = 200_000
	start_training_after: int = 2_000
	train_iterations_per_step: int = 1
	critic_ensemble: int = 5  # SAC++ style ensemble
	max_steps: int = 50_000
	print_every: int = 1_000
	
	# Risk-adjusted reward settings
	risk_penalty_lambda: float = 0.1  # Penalty weight for drawdowns
	sharpe_bonus_weight: float = 0.05  # Bonus weight for Sharpe ratio
	use_risk_adjusted_rewards: bool = True
	
	# Multi-transformer architecture settings
	use_multi_transformer: bool = True
	transformer_d_model: int = 64
	transformer_nhead: int = 4
	transformer_num_layers: int = 2
	
	# Technical indicators settings
	use_technical_indicators: bool = True
	indicator_window: int = 20  # Window for most indicators
	
	# Sentiment analysis settings
	use_sentiment: bool = True
	sentiment_model_name: str = "ProsusAI/finbert"  # or "yiyanghkust/finbert-tone"
	sentiment_window: int = 10  # Number of recent sentiment scores to include
	
	# Logging settings
	log_dir: str = "trading_logs"
	log_timestep_details: bool = True
	log_episode_summaries: bool = True
	log_trade_decisions: bool = True


class TechnicalIndicators:
	"""Comprehensive technical indicators calculator."""
	
	def __init__(self, window: int = 20):
		self.window = window
		self.price_history = deque(maxlen=window * 3)  # Keep more history for calculations
		self.volume_history = deque(maxlen=window * 3)
		
	def update(self, price: float, volume: float = 1000.0):
		"""Update with new price and volume data."""
		self.price_history.append(price)
		self.volume_history.append(volume)
	
	def get_all_indicators(self) -> Dict[str, float]:
		"""Calculate all technical indicators."""
		if len(self.price_history) < 2:
			return self._get_default_indicators()
		
		prices = np.array(self.price_history)
		volumes = np.array(self.volume_history)
		
		indicators = {}
		
		# Moving averages
		indicators['ema_20'] = self._ema(prices, 20)
		indicators['sma_50'] = self._sma(prices, min(50, len(prices)))
		
		# Momentum indicators
		indicators['rsi_14'] = self._rsi(prices, 14)
		indicators['roc_10'] = self._roc(prices, 10)
		
		# MACD
		macd_line, macd_signal = self._macd(prices)
		indicators['macd_line'] = macd_line
		indicators['macd_signal'] = macd_signal
		
		# Volatility indicators
		indicators['atr_14'] = self._atr(prices, 14)
		bb_upper, bb_lower = self._bollinger_bands(prices, 20)
		indicators['bb_upper'] = bb_upper
		indicators['bb_lower'] = bb_lower
		
		# Trend indicators
		indicators['adx_14'] = self._adx(prices, 14)
		indicators['supertrend'] = self._supertrend(prices)
		
		# Volume indicators
		indicators['obv'] = self._obv(prices, volumes)
		indicators['vwap'] = self._vwap(prices, volumes)
		
		return indicators
	
	def _get_default_indicators(self) -> Dict[str, float]:
		"""Return default values when insufficient data."""
		return {
			'ema_20': 0.0, 'sma_50': 0.0, 'rsi_14': 50.0, 'macd_line': 0.0,
			'macd_signal': 0.0, 'roc_10': 0.0, 'atr_14': 0.0, 'bb_upper': 0.0,
			'bb_lower': 0.0, 'adx_14': 0.0, 'supertrend': 0.0, 'obv': 0.0, 'vwap': 0.0
		}
	
	def _ema(self, prices: np.ndarray, period: int) -> float:
		"""Exponential Moving Average."""
		if len(prices) < period:
			return float(np.mean(prices))
		alpha = 2.0 / (period + 1)
		ema = prices[0]
		for price in prices[1:]:
			ema = alpha * price + (1 - alpha) * ema
		return float(ema)
	
	def _sma(self, prices: np.ndarray, period: int) -> float:
		"""Simple Moving Average."""
		if len(prices) < period:
			return float(np.mean(prices))
		return float(np.mean(prices[-period:]))
	
	def _rsi(self, prices: np.ndarray, period: int) -> float:
		"""Relative Strength Index."""
		if len(prices) < period + 1:
			return 50.0
		
		deltas = np.diff(prices)
		gains = np.where(deltas > 0, deltas, 0)
		losses = np.where(deltas < 0, -deltas, 0)
		
		avg_gain = np.mean(gains[-period:])
		avg_loss = np.mean(losses[-period:])
		
		if avg_loss == 0:
			return 100.0
		
		rs = avg_gain / avg_loss
		rsi = 100 - (100 / (1 + rs))
		return float(rsi)
	
	def _roc(self, prices: np.ndarray, period: int) -> float:
		"""Rate of Change."""
		if len(prices) < period + 1:
			return 0.0
		return float((prices[-1] - prices[-period-1]) / prices[-period-1] * 100)
	
	def _macd(self, prices: np.ndarray) -> Tuple[float, float]:
		"""MACD Line and Signal."""
		if len(prices) < 26:
			return 0.0, 0.0
		
		ema12 = self._ema(prices, 12)
		ema26 = self._ema(prices, 26)
		macd_line = ema12 - ema26
		
		# For signal line, we'd need MACD history, simplified here
		macd_signal = macd_line * 0.9  # Simplified signal line
		return float(macd_line), float(macd_signal)
	
	def _atr(self, prices: np.ndarray, period: int) -> float:
		"""Average True Range."""
		if len(prices) < 2:
			return 0.0
		
		highs = prices  # Simplified: using prices as highs
		lows = prices * 0.99  # Simplified: using prices as lows
		closes = np.roll(prices, 1)
		closes[0] = prices[0]
		
		tr1 = highs - lows
		tr2 = np.abs(highs - closes)
		tr3 = np.abs(lows - closes)
		
		true_range = np.maximum(tr1, np.maximum(tr2, tr3))
		return float(np.mean(true_range[-period:]))
	
	def _bollinger_bands(self, prices: np.ndarray, period: int) -> Tuple[float, float]:
		"""Bollinger Bands."""
		if len(prices) < period:
			return float(prices[-1]), float(prices[-1])
		
		sma = self._sma(prices, period)
		std = np.std(prices[-period:])
		
		upper = sma + (2 * std)
		lower = sma - (2 * std)
		return float(upper), float(lower)
	
	def _adx(self, prices: np.ndarray, period: int) -> float:
		"""Average Directional Index (simplified)."""
		if len(prices) < period + 1:
			return 0.0
		
		# Simplified ADX calculation
		deltas = np.diff(prices[-period-1:])
		pos_dm = np.sum(np.where(deltas > 0, deltas, 0))
		neg_dm = np.sum(np.where(deltas < 0, -deltas, 0))
		
		if pos_dm + neg_dm == 0:
			return 0.0
		
		dx = abs(pos_dm - neg_dm) / (pos_dm + neg_dm) * 100
		return float(dx)
	
	def _supertrend(self, prices: np.ndarray) -> float:
		"""Supertrend indicator (simplified)."""
		if len(prices) < 10:
			return float(prices[-1])
		
		atr = self._atr(prices, 10)
		hl2 = (np.max(prices[-10:]) + np.min(prices[-10:])) / 2
		
		upper_band = hl2 + (3 * atr)
		lower_band = hl2 - (3 * atr)
		
		# Simplified supertrend calculation
		if prices[-1] > upper_band:
			return float(lower_band)
		else:
			return float(upper_band)
	
	def _obv(self, prices: np.ndarray, volumes: np.ndarray) -> float:
		"""On-Balance Volume."""
		if len(prices) < 2 or len(volumes) < 2:
			return 0.0
		
		obv = 0.0
		for i in range(1, len(prices)):
			if prices[i] > prices[i-1]:
				obv += volumes[i]
			elif prices[i] < prices[i-1]:
				obv -= volumes[i]
		
		return float(obv)
	
	def _vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
		"""Volume Weighted Average Price."""
		if len(prices) == 0 or len(volumes) == 0:
			return 0.0
		
		total_volume = np.sum(volumes)
		if total_volume == 0:
			return float(prices[-1])
		
		vwap = np.sum(prices * volumes) / total_volume
		return float(vwap)


class SentimentAnalyzer:
	"""Sentiment analysis for financial news/tweets."""
	
	def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
		self.device = device
		self.model_name = model_name
		self.sentiment_history = deque(maxlen=100)  # Keep recent sentiment scores
		
		# Initialize sentiment model (fallback to simple scoring if model unavailable)
		try:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name)
			self.model = AutoModel.from_pretrained(model_name).to(device)
			self.model.eval()
			self.use_finbert = True
		except Exception as e:
			print(f"Warning: Could not load {model_name}, using simple sentiment scoring: {e}")
			self.use_finbert = False
	
	def analyze_sentiment(self, text: str) -> float:
		"""Analyze sentiment of financial text. Returns score in [-1, 1]."""
		if not text or not text.strip():
			return 0.0
		
		if self.use_finbert:
			return self._finbert_sentiment(text)
		else:
			return self._simple_sentiment(text)
	
	def _finbert_sentiment(self, text: str) -> float:
		"""Use FinBERT for sentiment analysis."""
		try:
			inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
			inputs = {k: v.to(self.device) for k, v in inputs.items()}
			
			with torch.no_grad():
				outputs = self.model(**inputs)
				# Use the last hidden state mean as sentiment score
				sentiment_score = torch.mean(outputs.last_hidden_state).item()
				# Normalize to [-1, 1] range
				sentiment_score = np.tanh(sentiment_score)
			
			return float(sentiment_score)
		except Exception as e:
			print(f"FinBERT sentiment analysis failed: {e}")
			return self._simple_sentiment(text)
	
	def _simple_sentiment(self, text: str) -> float:
		"""Simple keyword-based sentiment analysis."""
		text_lower = text.lower()
		
		# Positive keywords
		positive_words = [
			'bullish', 'rise', 'gain', 'profit', 'growth', 'positive', 'strong',
			'up', 'increase', 'surge', 'rally', 'breakout', 'momentum', 'buy'
		]
		
		# Negative keywords
		negative_words = [
			'bearish', 'fall', 'loss', 'decline', 'negative', 'weak', 'down',
			'decrease', 'crash', 'sell', 'correction', 'volatility', 'risk'
		]
		
		pos_count = sum(1 for word in positive_words if word in text_lower)
		neg_count = sum(1 for word in negative_words if word in text_lower)
		
		if pos_count + neg_count == 0:
			return 0.0
		
		sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
		return float(sentiment_score)
	
	def update_sentiment(self, sentiment_score: float):
		"""Update sentiment history."""
		self.sentiment_history.append(sentiment_score)
	
	def get_recent_sentiment(self, window: int = 10) -> List[float]:
		"""Get recent sentiment scores."""
		return list(self.sentiment_history)[-window:]


class IntradayTradingEnv:
	"""
	Enhanced single-asset intraday environment with:
	- Multi-transformer architecture support
	- Technical indicators integration
	- Sentiment analysis
	- Comprehensive logging
	- Risk-adjusted rewards
	- Historical timestamp tracking
	
	Observations: Multi-stream features [price/returns, technical indicators, position/cash, sentiment]
	Actions: target position fraction in [-1, 1] of NAV (respecting max_position_fraction)
	Rewards: risk-adjusted change in NAV with Sharpe/Sortino bonuses and drawdown penalties
	"""

	def __init__(self, prices: np.ndarray, times: np.ndarray, config: Config, 
				 news_data: Optional[List[str]] = None, volumes: Optional[np.ndarray] = None):
		assert prices.ndim == 1
		self.prices = prices.astype(np.float64)
		self.times = times  # minutes from market open; used for slippage schedule
		self.cfg = config
		random.seed(self.cfg.seed)
		np.random.seed(self.cfg.seed)

		# Initialize technical indicators and sentiment analyzer
		self.indicators = TechnicalIndicators(window=self.cfg.indicator_window)
		self.sentiment_analyzer = SentimentAnalyzer(
			model_name=self.cfg.sentiment_model_name, 
			device=self.cfg.device
		) if self.cfg.use_sentiment else None
		
		# News data for sentiment analysis
		self.news_data = news_data or []
		self.volumes = volumes if volumes is not None else np.full_like(prices, 1000.0)
		
		# Logging setup
		self.log_dir = self.cfg.log_dir
		os.makedirs(self.log_dir, exist_ok=True)
		
		# Initialize logging files
		self.timestep_log_file = os.path.join(self.log_dir, "timestep_logs.csv")
		self.episode_log_file = os.path.join(self.log_dir, "episode_logs.csv")
		self.trade_log_file = os.path.join(self.log_dir, "trade_logs.csv")
		
		# Initialize CSV headers
		self._init_log_files()
		
		# Episode tracking
		self.episode_start_time = None
		self.episode_trades = []
		self.current_timestamp = None
		
		self.window = self.cfg.obs_lookback
		self.reset()

	def _init_log_files(self):
		"""Initialize CSV log files with headers."""
		# Timestep log headers
		timestep_headers = [
			'timestamp', 'step', 'price', 'action', 'fill_price', 'units', 'cash', 'NAV', 'P&L',
			'position_fraction', 'cash_fraction', 'volatility', 'raw_reward', 'risk_adjusted_reward'
		]
		
		# Add technical indicator headers
		if self.cfg.use_technical_indicators:
			indicator_names = [
				'ema_20', 'sma_50', 'rsi_14', 'macd_line', 'macd_signal', 'roc_10',
				'atr_14', 'bb_upper', 'bb_lower', 'adx_14', 'supertrend', 'obv', 'vwap'
			]
			timestep_headers.extend(indicator_names)
		
		# Add sentiment headers
		if self.cfg.use_sentiment:
			timestep_headers.extend(['sentiment_score', 'sentiment_ma'])
		
		# Write timestep headers
		with open(self.timestep_log_file, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(timestep_headers)
		
		# Episode log headers
		episode_headers = [
			'episode', 'start_time', 'end_time', 'duration_minutes', 'total_return_pct',
			'final_nav', 'max_nav', 'max_drawdown_pct', 'sharpe_ratio', 'sortino_ratio',
			'avg_position_pct', 'max_position_pct', 'num_trades', 'win_rate_pct',
			'avg_trade_pnl', 'total_fees', 'total_slippage'
		]
		
		with open(self.episode_log_file, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(episode_headers)
		
		# Trade log headers
		trade_headers = [
			'timestamp', 'episode', 'trade_id', 'action_type', 'entry_price', 'exit_price',
			'size', 'direction', 'duration_minutes', 'pnl', 'fees', 'slippage',
			'stop_loss', 'take_profit', 'reason_exit'
		]
		
		# Add technical indicators at trade time
		if self.cfg.use_technical_indicators:
			indicator_names = [
				'ema_20', 'sma_50', 'rsi_14', 'macd_line', 'macd_signal', 'roc_10',
				'atr_14', 'bb_upper', 'bb_lower', 'adx_14', 'supertrend', 'obv', 'vwap'
			]
			trade_headers.extend([f'trade_{name}' for name in indicator_names])
		
		# Add sentiment at trade time
		if self.cfg.use_sentiment:
			trade_headers.extend(['trade_sentiment_score', 'trade_sentiment_ma'])
		
		with open(self.trade_log_file, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(trade_headers)

	def _calc_vol(self, t: int) -> float:
		start = max(1, t - self.cfg.rolling_vol_window)
		rets = np.diff(self.prices[start - 1:t + 1]) / self.prices[start - 1:t]
		if rets.size < 2:
			return 0.0
		return float(np.std(rets))

	def _slippage_bps_for_time(self, minute_index: int) -> float:
		# Assume minutes since open, with last 10 minutes and first 10 minutes more volatile
		minute = int(self.times[minute_index]) if minute_index < len(self.times) else 0
		if minute < 10 or minute > (self.times[-1] - 10):
			return self.cfg.transaction_slippage_bps_open_close
		return self.cfg.transaction_slippage_bps_mid

	def _apply_fills(self, desired_units: float, price: float, minute_index: int) -> Tuple[float, float]:
		"""Apply brokerage and slippage; return (fill_price, fee_paid)."""
		slip_bps = self._slippage_bps_for_time(minute_index)
		direction = 1.0 if desired_units > 0 else -1.0
		fill_price = price * (1.0 + direction * (slip_bps / 1e4))
		fee = abs(desired_units) * fill_price * self.cfg.transaction_brokerage_pct
		return fill_price, fee

	def reset(self):
		self.t = self.window
		self.cash = float(self.cfg.starting_capital)
		self.units = 0.0
		self.avg_price = 0.0
		self.trailing_stop = None
		self.nav_start_of_day = self.cash
		self.done = False
		self.day_index = 0
		self.nav_history = [self.cash]  # Track NAV for risk metrics
		self.reward_history = []  # Track rewards for Sharpe calculation
		
		# Reset technical indicators
		if self.cfg.use_technical_indicators:
			self.indicators = TechnicalIndicators(window=self.cfg.indicator_window)
			# Initialize with historical data
			for i in range(min(self.t, len(self.prices))):
				self.indicators.update(self.prices[i], self.volumes[i])
		
		# Reset sentiment analyzer
		if self.cfg.use_sentiment and self.sentiment_analyzer:
			self.sentiment_analyzer.sentiment_history.clear()
		
		# Episode tracking
		self.episode_start_time = datetime.now()
		self.episode_trades = []
		self.trade_counter = 0
		
		return self._get_obs()

	def _mark_to_market(self) -> float:
		price = self.prices[self.t]
		position_value = self.units * price
		return self.cash + position_value

	def _get_obs(self):
		"""Get multi-stream observation with price, indicators, position, and sentiment data."""
		start = self.t - self.window
		px = self.prices[start:self.t + 1]
		rets = np.diff(px) / px[:-1]
		if rets.size < self.window:
			rets = np.pad(rets, (self.window - rets.size, 0))
		
		vol = self._calc_vol(self.t)
		nav = self._mark_to_market()
		pos_frac = 0.0 if nav <= 0 else (self.units * self.prices[self.t]) / nav
		cash_frac = 0.0 if nav <= 0 else self.cash / nav
		
		# Base features: price returns, volatility, position, cash
		feat_streams = [
			rets[-self.window:],  # Price returns
			np.full(self.window, vol, dtype=np.float64),  # Volatility
			np.full(self.window, pos_frac, dtype=np.float64),  # Position fraction
			np.full(self.window, cash_frac, dtype=np.float64),  # Cash fraction
		]
		
		# Add technical indicators if enabled
		if self.cfg.use_technical_indicators:
			indicators = self.indicators.get_all_indicators()
			for indicator_name, value in indicators.items():
				feat_streams.append(np.full(self.window, value, dtype=np.float64))
		
		# Add sentiment if enabled
		if self.cfg.use_sentiment and self.sentiment_analyzer:
			# Get current sentiment score
			current_news = ""
			if self.t < len(self.news_data):
				current_news = self.news_data[self.t]
			
			sentiment_score = self.sentiment_analyzer.analyze_sentiment(current_news)
			self.sentiment_analyzer.update_sentiment(sentiment_score)
			
			# Add current sentiment and moving average
			feat_streams.append(np.full(self.window, sentiment_score, dtype=np.float64))
			
			recent_sentiment = self.sentiment_analyzer.get_recent_sentiment(self.cfg.sentiment_window)
			sentiment_ma = np.mean(recent_sentiment) if recent_sentiment else 0.0
			feat_streams.append(np.full(self.window, sentiment_ma, dtype=np.float64))
		
		# Stack all feature streams
		feat = np.stack(feat_streams, axis=1)
		return feat.astype(np.float32)

	def _log_timestep(self, step: int, action: float, fill_price: float, units: float, 
					 raw_reward: float, risk_adjusted_reward: float):
		"""Log detailed timestep information."""
		if not self.cfg.log_timestep_details:
			return
		
		# Generate timestamp
		current_time = datetime.now()
		if self.current_timestamp is None:
			self.current_timestamp = current_time
		
		# Calculate current metrics
		nav = self._mark_to_market()
		position_fraction = 0.0 if nav <= 0 else (self.units * self.prices[self.t]) / nav
		cash_fraction = 0.0 if nav <= 0 else self.cash / nav
		vol = self._calc_vol(self.t)
		pnl = nav - self.cfg.starting_capital
		
		# Prepare log row
		log_row = [
			current_time.strftime('%Y-%m-%d %H:%M:%S'),
			step,
			self.prices[self.t],
			action,
			fill_price,
			self.units,
			self.cash,
			nav,
			pnl,
			position_fraction,
			cash_fraction,
			vol,
			raw_reward,
			risk_adjusted_reward
		]
		
		# Add technical indicators
		if self.cfg.use_technical_indicators:
			indicators = self.indicators.get_all_indicators()
			log_row.extend([indicators[name] for name in [
				'ema_20', 'sma_50', 'rsi_14', 'macd_line', 'macd_signal', 'roc_10',
				'atr_14', 'bb_upper', 'bb_lower', 'adx_14', 'supertrend', 'obv', 'vwap'
			]])
		
		# Add sentiment
		if self.cfg.use_sentiment and self.sentiment_analyzer:
			current_sentiment = 0.0
			sentiment_ma = 0.0
			if self.sentiment_analyzer.sentiment_history:
				current_sentiment = self.sentiment_analyzer.sentiment_history[-1]
				recent_sentiment = self.sentiment_analyzer.get_recent_sentiment(self.cfg.sentiment_window)
				sentiment_ma = np.mean(recent_sentiment) if recent_sentiment else 0.0
			
			log_row.extend([current_sentiment, sentiment_ma])
		
		# Write to CSV
		with open(self.timestep_log_file, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(log_row)

	def _log_trade(self, action_type: str, entry_price: float, exit_price: float, 
				  size: float, direction: str, pnl: float, fees: float, slippage: float,
				  reason_exit: str = "normal"):
		"""Log detailed trade information."""
		if not self.cfg.log_trade_decisions:
			return
		
		self.trade_counter += 1
		current_time = datetime.now()
		
		# Calculate trade duration
		duration_minutes = 0
		if self.episode_trades:
			last_trade_time = self.episode_trades[-1].get('timestamp', current_time)
			duration_minutes = (current_time - last_trade_time).total_seconds() / 60
		
		# Prepare trade log row
		log_row = [
			current_time.strftime('%Y-%m-%d %H:%M:%S'),
			self.day_index,  # episode
			self.trade_counter,
			action_type,
			entry_price,
			exit_price,
			size,
			direction,
			duration_minutes,
			pnl,
			fees,
			slippage,
			0.0,  # stop_loss (simplified)
			0.0,  # take_profit (simplified)
			reason_exit
		]
		
		# Add technical indicators at trade time
		if self.cfg.use_technical_indicators:
			indicators = self.indicators.get_all_indicators()
			log_row.extend([indicators[name] for name in [
				'ema_20', 'sma_50', 'rsi_14', 'macd_line', 'macd_signal', 'roc_10',
				'atr_14', 'bb_upper', 'bb_lower', 'adx_14', 'supertrend', 'obv', 'vwap'
			]])
		
		# Add sentiment at trade time
		if self.cfg.use_sentiment and self.sentiment_analyzer:
			current_sentiment = 0.0
			sentiment_ma = 0.0
			if self.sentiment_analyzer.sentiment_history:
				current_sentiment = self.sentiment_analyzer.sentiment_history[-1]
				recent_sentiment = self.sentiment_analyzer.get_recent_sentiment(self.cfg.sentiment_window)
				sentiment_ma = np.mean(recent_sentiment) if recent_sentiment else 0.0
			
			log_row.extend([current_sentiment, sentiment_ma])
		
		# Write to CSV
		with open(self.trade_log_file, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(log_row)
		
		# Store trade info for episode summary
		trade_info = {
			'timestamp': current_time,
			'action_type': action_type,
			'entry_price': entry_price,
			'exit_price': exit_price,
			'size': size,
			'direction': direction,
			'pnl': pnl,
			'fees': fees,
			'slippage': slippage,
			'reason_exit': reason_exit
		}
		self.episode_trades.append(trade_info)

	def _log_episode_summary(self, episode_num: int, final_nav: float):
		"""Log episode summary statistics."""
		if not self.cfg.log_episode_summaries:
			return
		
		end_time = datetime.now()
		duration_minutes = (end_time - self.episode_start_time).total_seconds() / 60
		
		# Calculate episode metrics
		total_return_pct = (final_nav - self.cfg.starting_capital) / self.cfg.starting_capital * 100
		max_nav = max(self.nav_history) if self.nav_history else final_nav
		
		# Calculate drawdown
		navs = np.array(self.nav_history)
		peak_nav = np.maximum.accumulate(navs)
		drawdowns = (peak_nav - navs) / peak_nav
		max_drawdown_pct = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0
		
		# Calculate Sharpe and Sortino ratios
		if len(self.reward_history) > 1:
			rewards = np.array(self.reward_history)
			sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252 * 390)
			negative_rewards = rewards[rewards < 0]
			sortino_ratio = np.mean(rewards) / (np.std(negative_rewards) + 1e-8) * np.sqrt(252 * 390) if len(negative_rewards) > 0 else 0.0
		else:
			sharpe_ratio = sortino_ratio = 0.0
		
		# Position statistics
		position_fractions = []
		for i, nav in enumerate(self.nav_history):
			if nav > 0:
				pos_frac = abs(self.units * self.prices[min(i, len(self.prices)-1)]) / nav
				position_fractions.append(pos_frac)
		
		avg_position_pct = np.mean(position_fractions) * 100 if position_fractions else 0.0
		max_position_pct = np.max(position_fractions) * 100 if position_fractions else 0.0
		
		# Trade statistics
		num_trades = len(self.episode_trades)
		winning_trades = [t for t in self.episode_trades if t['pnl'] > 0]
		win_rate_pct = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0.0
		avg_trade_pnl = np.mean([t['pnl'] for t in self.episode_trades]) if self.episode_trades else 0.0
		total_fees = sum(t['fees'] for t in self.episode_trades)
		total_slippage = sum(t['slippage'] for t in self.episode_trades)
		
		# Prepare episode log row
		log_row = [
			episode_num,
			self.episode_start_time.strftime('%Y-%m-%d %H:%M:%S'),
			end_time.strftime('%Y-%m-%d %H:%M:%S'),
			duration_minutes,
			total_return_pct,
			final_nav,
			max_nav,
			max_drawdown_pct,
			sharpe_ratio,
			sortino_ratio,
			avg_position_pct,
			max_position_pct,
			num_trades,
			win_rate_pct,
			avg_trade_pnl,
			total_fees,
			total_slippage
		]
		
		# Write to CSV
		with open(self.episode_log_file, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(log_row)

	def _eod(self, next_t: int) -> bool:
		# EOD when times resets or when next index belongs to a new day. Here assume synthetic day every 390 minutes.
		if next_t >= len(self.times):
			return True
		return self.times[next_t] < self.times[self.t]

	def _calculate_risk_adjusted_reward(self, raw_reward: float, nav: float) -> float:
		"""Calculate risk-adjusted reward with drawdown penalties and Sharpe bonuses."""
		if not self.cfg.use_risk_adjusted_rewards:
			return raw_reward
		
		# Update history
		self.nav_history.append(nav)
		self.reward_history.append(raw_reward)
		
		# Keep only recent history for efficiency (last 1000 steps)
		if len(self.nav_history) > 1000:
			self.nav_history = self.nav_history[-1000:]
		if len(self.reward_history) > 1000:
			self.reward_history = self.reward_history[-1000:]
		
		risk_adjusted_reward = raw_reward
		
		# Drawdown penalty
		if len(self.nav_history) >= 2:
			navs = np.array(self.nav_history)
			peak_nav = np.maximum.accumulate(navs)
			current_drawdown = (peak_nav[-1] - navs[-1]) / peak_nav[-1] if peak_nav[-1] > 0 else 0.0
			
			# Penalty increases quadratically with drawdown
			drawdown_penalty = self.cfg.risk_penalty_lambda * (current_drawdown ** 2)
			risk_adjusted_reward -= drawdown_penalty
		
		# Sharpe ratio bonus (rolling)
		if len(self.reward_history) >= 20:  # Need enough history
			rewards = np.array(self.reward_history[-20:])  # Last 20 rewards
			if len(rewards) > 1 and np.std(rewards) > 1e-8:
				rolling_sharpe = np.mean(rewards) / np.std(rewards)
				# Bonus for positive Sharpe, penalty for negative
				sharpe_bonus = self.cfg.sharpe_bonus_weight * rolling_sharpe
				risk_adjusted_reward += sharpe_bonus
		
		return risk_adjusted_reward

	def _enforce_trailing_stop(self, price: float) -> Optional[Tuple[float, float]]:
		if self.units == 0:
			self.trailing_stop = None
			return None
		if self.units > 0:
			# Long: update trailing stop on favorable move
			peak_price = max(price, self.avg_price if self.trailing_stop is None else price)
			trail = peak_price * (1.0 - self.cfg.trailing_stop_pct)
			self.trailing_stop = trail if self.trailing_stop is None else max(self.trailing_stop, trail)
			if price <= self.trailing_stop:
				return (-self.units, price)
		else:
			# Short: symmetric trailing stop
			trough_price = min(price, self.avg_price if self.trailing_stop is None else price)
			trail = trough_price * (1.0 + self.cfg.trailing_stop_pct)
			self.trailing_stop = trail if self.trailing_stop is None else min(self.trailing_stop, trail)
			if price >= self.trailing_stop:
				return (-self.units, price)
		return None

	def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
		if self.done:
			return self._get_obs(), 0.0, True, {}
		
		action = float(np.clip(action, -1.0, 1.0))
		price = self.prices[self.t]
		nav = self._mark_to_market()
		
		# Update technical indicators with current price and volume
		if self.cfg.use_technical_indicators:
			self.indicators.update(price, self.volumes[self.t])

		# Track trade execution for logging
		trade_executed = False
		trade_action_type = ""
		trade_fill_price = price
		trade_units = 0.0
		trade_fees = 0.0
		trade_slippage = 0.0

		# Trailing stop check
		forced = self._enforce_trailing_stop(price)
		if forced is not None:
			delta_units, p = forced
			fill_price, fee = self._apply_fills(delta_units, p, self.t)
			self.cash += (-delta_units) * -fill_price - fee  # sell long or buy to cover short
			self.units += delta_units
			
			# Log trailing stop trade
			trade_executed = True
			trade_action_type = "trailing_stop"
			trade_fill_price = fill_price
			trade_units = delta_units
			trade_fees = fee
			trade_slippage = abs(fill_price - p)
			
			if self.units == 0:
				self.avg_price = 0.0
				self.trailing_stop = None

		# Target scaling using volatility; smaller steps when vol high
		vol = max(self._calc_vol(self.t), 1e-6)
		max_pos_frac = self.cfg.max_position_fraction
		target_frac = float(np.clip(action * max_pos_frac, -max_pos_frac, max_pos_frac))
		current_frac = 0.0 if nav <= 0 else (self.units * price) / nav
		# step size inversely proportional to vol
		step_frac = float(np.clip(0.2 / (1.0 + 50.0 * vol), 0.02, 0.2))
		desired_frac = current_frac + np.clip(target_frac - current_frac, -step_frac, step_frac)
		desired_value = desired_frac * nav
		desired_units = 0.0 if price <= 0 else (desired_value / price) - self.units

		# Enforce no leverage: ensure resulting position value <= NAV
		post_units = self.units + desired_units
		post_value = abs(post_units * price)
		allowed_value = nav  # no leverage
		if post_value > allowed_value and post_value > 0:
			scale = allowed_value / post_value
			desired_units *= scale

		if abs(desired_units) > 0:
			fill_price, fee = self._apply_fills(desired_units, price, self.t)
			cost = desired_units * fill_price + fee
			if self.cash - cost < 0 and desired_units > 0:
				# not enough cash to buy; scale down
				max_buy_units = max(0.0, (self.cash - fee) / fill_price)
				desired_units = min(desired_units, max_buy_units)
				fill_price, fee = self._apply_fills(desired_units, price, self.t)
				cost = desired_units * fill_price + fee
			
			self.cash -= cost
			self.units += desired_units
			
			# Log regular trade
			if not trade_executed:
				trade_executed = True
				trade_action_type = "buy" if desired_units > 0 else "sell"
				trade_fill_price = fill_price
				trade_units = desired_units
				trade_fees = fee
				trade_slippage = abs(fill_price - price)
			
			if self.units != 0:
				# update average price
				if desired_units > 0:
					self.avg_price = (self.avg_price * (self.units - desired_units) + desired_units * fill_price) / max(1e-9, self.units)
				else:
					# selling reduces position; if cross zero, reset avg
					if np.sign(self.units) != np.sign(self.units - desired_units):
						self.avg_price = fill_price
			else:
				self.avg_price = 0.0
				self.trailing_stop = None

		# Advance time
		next_t = self.t + 1
		# EOD square-off
		if self._eod(next_t):
			if self.units != 0:
				fill_price, fee = self._apply_fills(-self.units, price, self.t)
				self.cash += self.units * fill_price - fee
				
				# Log EOD trade
				if not trade_executed:
					trade_executed = True
					trade_action_type = "eod_close"
					trade_fill_price = fill_price
					trade_units = -self.units
					trade_fees = fee
					trade_slippage = abs(fill_price - price)
				
				self.units = 0.0
				self.avg_price = 0.0
				self.trailing_stop = None
			self.nav_start_of_day = self.cash
			self.day_index += 1

		old_nav = nav
		self.t = next_t
		if self.t >= len(self.prices):
			self.done = True
			return self._get_obs(), 0.0, True, {"nav": old_nav}
		
		new_nav = self._mark_to_market()
		raw_reward = float(new_nav - old_nav)
		reward = self._calculate_risk_adjusted_reward(raw_reward, new_nav)
		
		# Log trade if executed
		if trade_executed:
			direction = "long" if trade_units > 0 else "short"
			pnl = trade_units * (trade_fill_price - self.avg_price) if self.avg_price > 0 else 0.0
			self._log_trade(
				action_type=trade_action_type,
				entry_price=self.avg_price if self.avg_price > 0 else trade_fill_price,
				exit_price=trade_fill_price,
				size=abs(trade_units),
				direction=direction,
				pnl=pnl,
				fees=trade_fees,
				slippage=trade_slippage,
				reason_exit=trade_action_type
			)
		
		# Log timestep details
		self._log_timestep(
			step=self.t,
			action=action,
			fill_price=trade_fill_price if trade_executed else price,
			units=self.units,
			raw_reward=raw_reward,
			risk_adjusted_reward=reward
		)
		
		return self._get_obs(), reward, False, {"nav": new_nav, "raw_reward": raw_reward}


class ReplayBuffer:
	def __init__(self, capacity: int):
		self.capacity = capacity
		self.storage: Deque[Tuple[np.ndarray, float, float, np.ndarray, float]] = deque(maxlen=capacity)

	def add(self, s: np.ndarray, a: float, r: float, s2: np.ndarray, d: float):
		self.storage.append((s.copy(), float(a), float(r), s2.copy(), float(d)))

	def sample(self, batch_size: int):
		batch = random.sample(self.storage, batch_size)
		s, a, r, s2, d = zip(*batch)
		return (
			torch.tensor(np.stack(s), dtype=torch.float32),
			torch.tensor(np.array(a), dtype=torch.float32).unsqueeze(-1),
			torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(-1),
			torch.tensor(np.stack(s2), dtype=torch.float32),
			torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(-1),
		)

	def __len__(self):
		return len(self.storage)


class MultiStreamTransformerEncoder(nn.Module):
	"""Multi-stream transformer encoder for different data types."""
	
	def __init__(self, config: Config):
		super().__init__()
		self.cfg = config
		d_model = config.transformer_d_model
		nhead = config.transformer_nhead
		num_layers = config.transformer_num_layers
		
		# Separate encoders for different data streams
		self.price_encoder = self._create_encoder(4, d_model, nhead, num_layers)  # returns, vol, pos, cash
		
		if config.use_technical_indicators:
			# Technical indicators stream (13 indicators)
			self.indicator_encoder = self._create_encoder(13, d_model, nhead, num_layers)
		
		if config.use_sentiment:
			# Sentiment stream (2 features: current sentiment, sentiment MA)
			self.sentiment_encoder = self._create_encoder(2, d_model, nhead, num_layers)
		
		# Fusion layer will be created dynamically in forward pass
		self.fusion_layer = None
		
		self.output_dim = d_model
	
	def _create_encoder(self, input_dim: int, d_model: int, nhead: int, num_layers: int):
		"""Create a transformer encoder for a specific data stream."""
		return nn.Sequential(
			nn.Linear(input_dim, d_model),
			nn.ReLU(),
			nn.TransformerEncoder(
				nn.TransformerEncoderLayer(
					d_model=d_model, 
					nhead=nhead, 
					batch_first=True,
					dropout=0.1
				), 
				num_layers=num_layers
			),
			nn.AdaptiveAvgPool1d(1),
			nn.Flatten()
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through multi-stream transformer.
		x: [B, T, F] where F includes all feature streams
		"""
		batch_size, seq_len, total_features = x.shape
		
		# Split features into streams
		feature_idx = 0
		
		# Price stream: returns, volatility, position, cash (4 features)
		price_features = x[:, :, feature_idx:feature_idx+4]
		feature_idx += 4
		
		# Process price stream
		price_encoded = self.price_encoder(price_features)
		
		# Collect all encoded streams
		encoded_streams = [price_encoded]
		
		# Technical indicators stream
		if self.cfg.use_technical_indicators and hasattr(self, 'indicator_encoder'):
			indicator_features = x[:, :, feature_idx:feature_idx+13]
			feature_idx += 13
			indicator_encoded = self.indicator_encoder(indicator_features)
			encoded_streams.append(indicator_encoded)
		
		# Sentiment stream
		if self.cfg.use_sentiment and hasattr(self, 'sentiment_encoder'):
			sentiment_features = x[:, :, feature_idx:feature_idx+2]
			feature_idx += 2
			sentiment_encoded = self.sentiment_encoder(sentiment_features)
			encoded_streams.append(sentiment_encoded)
		
		# Concatenate all streams
		combined_features = torch.cat(encoded_streams, dim=-1)
		
		# Create fusion layer dynamically if not exists
		if self.fusion_layer is None:
			actual_input_dim = combined_features.shape[-1]
			self.fusion_layer = nn.Sequential(
				nn.Linear(actual_input_dim, self.cfg.transformer_d_model * 2),
				nn.ReLU(),
				nn.Linear(self.cfg.transformer_d_model * 2, self.cfg.transformer_d_model),
				nn.ReLU()
			).to(combined_features.device)
		
		# Fusion layer
		output = self.fusion_layer(combined_features)
		
		return output


class TimeSeriesEncoder(nn.Module):
	"""Legacy single-stream encoder for backward compatibility."""
	def __init__(self, feature_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
		super().__init__()
		self.input_proj = nn.Linear(feature_dim, d_model)
		layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
		self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
		self.pool = nn.AdaptiveAvgPool1d(1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, T, F]
		h = self.input_proj(x)
		h = self.encoder(h)
		# pool over time
		h = h.transpose(1, 2)
		h = self.pool(h).squeeze(-1)
		return h


class Actor(nn.Module):
	def __init__(self, config: Config, hidden: int = 128):
		super().__init__()
		self.cfg = config
		
		# Use multi-stream transformer if enabled, otherwise legacy encoder
		if config.use_multi_transformer:
			self.backbone = MultiStreamTransformerEncoder(config)
			backbone_output_dim = self.backbone.output_dim
		else:
			# Calculate feature dimension based on enabled features
			feature_dim = 4  # base features: returns, vol, pos, cash
			if config.use_technical_indicators:
				feature_dim += 13  # technical indicators
			if config.use_sentiment:
				feature_dim += 2  # sentiment features
			
			self.backbone = TimeSeriesEncoder(feature_dim, config.transformer_d_model, 
											config.transformer_nhead, config.transformer_num_layers)
			backbone_output_dim = config.transformer_d_model
		
		self.mlp = nn.Sequential(
			nn.Linear(backbone_output_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, 2),  # mean, log_std
		)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		z = self.backbone(x)
		out = self.mlp(z)
		mean, log_std = out[:, :1], out[:, 1:2]
		log_std = torch.clamp(log_std, -5, 1)
		return mean, log_std

	def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		mean, log_std = self(x)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		z = normal.rsample()
		a = torch.tanh(z)  # bound to [-1, 1]
		log_prob = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
		return a, log_prob


class Critic(nn.Module):
	def __init__(self, config: Config, hidden: int = 128):
		super().__init__()
		self.cfg = config
		
		# Use multi-stream transformer if enabled, otherwise legacy encoder
		if config.use_multi_transformer:
			self.backbone = MultiStreamTransformerEncoder(config)
			backbone_output_dim = self.backbone.output_dim
		else:
			# Calculate feature dimension based on enabled features
			feature_dim = 4  # base features: returns, vol, pos, cash
			if config.use_technical_indicators:
				feature_dim += 13  # technical indicators
			if config.use_sentiment:
				feature_dim += 2  # sentiment features
			
			self.backbone = TimeSeriesEncoder(feature_dim, config.transformer_d_model, 
											config.transformer_nhead, config.transformer_num_layers)
			backbone_output_dim = config.transformer_d_model
		
		self.q = nn.Sequential(
			nn.Linear(backbone_output_dim + 1, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, 1),
		)

	def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
		z = self.backbone(x)
		qa = self.q(torch.cat([z, a], dim=-1))
		return qa


class SACAgent:
	def __init__(self, obs_shape: Tuple[int, int], cfg: Config):
		self.cfg = cfg
		self.device = cfg.device
		
		# Initialize actor and critics with config-based architecture
		self.actor = Actor(cfg).to(self.device)
		self.critics = nn.ModuleList([Critic(cfg).to(self.device) for _ in range(cfg.critic_ensemble)])
		self.critics_target = nn.ModuleList([Critic(cfg).to(self.device) for _ in range(cfg.critic_ensemble)])
		
		# Copy initial weights to target networks
		for t, s in zip(self.critics_target, self.critics):
			t.load_state_dict(s.state_dict())
		
		# Optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
		self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.critic_lr) for c in self.critics]
		
		# SAC++ automatic entropy tuning
		if cfg.auto_entropy:
			self.log_alpha = torch.tensor(np.log(cfg.alpha), device=self.device, requires_grad=True)
			self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
			self.target_entropy = cfg.target_entropy
		else:
			self.log_alpha = torch.tensor(np.log(cfg.alpha), device=self.device, requires_grad=False)
		
		self.alpha = cfg.alpha

	def act(self, obs: np.ndarray, explore: bool = True) -> float:
		self.actor.eval()
		with torch.no_grad():
			x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
			if explore:
				a, _ = self.actor.sample(x)
			else:
				mean, _ = self.actor(x)
				a = torch.tanh(mean)
			return float(a.squeeze().cpu().numpy())

	def update(self, batch, step: int):
		s, a, r, s2, d = batch
		s = s.to(self.device)
		a = a.to(self.device)
		r = r.to(self.device)
		s2 = s2.to(self.device)
		d = d.to(self.device)

		# Update alpha (entropy coefficient) if using SAC++
		if self.cfg.auto_entropy:
			with torch.no_grad():
				a_sample, logp_sample = self.actor.sample(s)
			alpha_loss = -(self.log_alpha * (logp_sample + self.target_entropy).detach()).mean()
			self.alpha_opt.zero_grad(set_to_none=True)
			alpha_loss.backward()
			self.alpha_opt.step()
			self.alpha = self.log_alpha.exp().item()

		# Critic targets
		with torch.no_grad():
			a2, logp2 = self.actor.sample(s2)
			q_targets = []
			for ct in self.critics_target:
				q_targets.append(ct(s2, a2))
			q_target = torch.min(torch.stack(q_targets, dim=0), dim=0).values
			y = r + (1 - d) * self.cfg.gamma * (q_target - self.alpha * logp2)

		# Critic update (ensemble)
		critic_losses = []
		for c, opt in zip(self.critics, self.critic_opts):
			q = c(s, a)
			loss = F.mse_loss(q, y)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			critic_losses.append(loss.detach())

		# Actor update
		a_new, logp = self.actor.sample(s)
		q_new_list = [c(s, a_new) for c in self.critics]
		q_new = torch.min(torch.stack(q_new_list, dim=0), dim=0).values
		actor_loss = (self.alpha * logp - q_new).mean()
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		# Soft update targets
		with torch.no_grad():
			for ct, c in zip(self.critics_target, self.critics):
				for tp, p in zip(ct.parameters(), c.parameters()):
					tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

		alpha_loss_val = alpha_loss.detach().cpu().numpy() if self.cfg.auto_entropy else 0.0
		return float(torch.mean(torch.stack(critic_losses)).cpu().numpy()), float(actor_loss.detach().cpu().numpy()), alpha_loss_val


def generate_synthetic_intraday(days: int = 10, minutes_per_day: int = 390, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
	"""Generate synthetic intraday data with prices, times, news, and volumes."""
	rng = np.random.default_rng(seed)
	prices = []
	times = []
	volumes = []
	news_data = []
	
	# Sample financial news/tweets for sentiment analysis
	sample_news = [
		"Market shows strong bullish momentum with positive earnings reports",
		"Investors remain cautious amid economic uncertainty",
		"Stock prices surge on positive economic data",
		"Market volatility increases due to geopolitical tensions",
		"Trading volume spikes as institutional investors enter",
		"Price correction expected after recent rally",
		"Technical indicators suggest oversold conditions",
		"Market sentiment turns bearish on negative news",
		"Strong buying pressure drives prices higher",
		"Profit-taking causes temporary price decline",
		"Market consolidates after recent gains",
		"Breakout above resistance levels signals continuation",
		"Support levels hold as buyers step in",
		"Volume confirms the price movement",
		"Market shows mixed signals with indecision"
	]
	
	price = 100.0
	for day in range(days):
		minute = 0
		for _ in range(minutes_per_day):
			# Heteroskedastic: higher vol at open/close
			vol = 0.0008 if (minute < 10 or minute > minutes_per_day - 10) else 0.0003
			ret = rng.normal(0.0, vol)
			price = max(0.5, price * (1.0 + ret))
			
			# Generate volume (higher at open/close, correlated with price movement)
			base_volume = 1000.0
			volume_multiplier = 1.0 + abs(ret) * 10  # Higher volume on larger moves
			if minute < 10 or minute > minutes_per_day - 10:
				volume_multiplier *= 2.0  # Higher volume at open/close
			volume = base_volume * volume_multiplier * rng.uniform(0.5, 1.5)
			
			# Generate news (more frequent during high volatility periods)
			news = ""
			if rng.random() < 0.1 or abs(ret) > 2 * vol:  # 10% chance or during large moves
				news = rng.choice(sample_news)
			
			prices.append(price)
			times.append(minute)
			volumes.append(volume)
			news_data.append(news)
			minute += 1
	
	return (np.array(prices, dtype=np.float64), 
			np.array(times, dtype=np.int32),
			news_data,
			np.array(volumes, dtype=np.float64))


class MetricsTracker:
	def __init__(self):
		self.nav_history: List[float] = []
		self.reward_history: List[float] = []
		self.action_history: List[float] = []
		self.position_history: List[float] = []
		self.episode_returns: List[float] = []
		self.start_time = time.time()

	def update(self, nav: float, reward: float, action: float, position_frac: float):
		self.nav_history.append(nav)
		self.reward_history.append(reward)
		self.action_history.append(action)
		self.position_history.append(position_frac)

	def end_episode(self, final_nav: float):
		if self.nav_history:
			episode_return = (final_nav - self.nav_history[0]) / self.nav_history[0]
			self.episode_returns.append(episode_return)

	def get_stats(self) -> Dict[str, float]:
		if not self.nav_history:
			return {}
		
		navs = np.array(self.nav_history)
		rewards = np.array(self.reward_history)
		
		# Basic stats
		total_return = (navs[-1] - navs[0]) / navs[0] if navs[0] > 0 else 0.0
		max_nav = np.max(navs)
		max_drawdown = np.max(np.maximum.accumulate(navs) - navs) / np.maximum.accumulate(navs)
		max_dd = np.max(max_drawdown) if len(max_drawdown) > 0 else 0.0
		
		# Risk metrics
		if len(rewards) > 1:
			sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252 * 390)  # annualized
			sortino = np.mean(rewards) / (np.std(rewards[rewards < 0]) + 1e-8) * np.sqrt(252 * 390)
		else:
			sharpe = sortino = 0.0
		
		# Position stats
		avg_position = np.mean(np.abs(self.position_history))
		max_position = np.max(np.abs(self.position_history))
		
		return {
			"total_return_pct": total_return * 100,
			"max_nav": max_nav,
			"max_drawdown_pct": max_dd * 100,
			"sharpe_ratio": sharpe,
			"sortino_ratio": sortino,
			"avg_position_pct": avg_position * 100,
			"max_position_pct": max_position * 100,
			"episodes_completed": len(self.episode_returns),
			"avg_episode_return_pct": np.mean(self.episode_returns) * 100 if self.episode_returns else 0.0,
			"training_time_min": (time.time() - self.start_time) / 60,
		}


def train(cfg: Config, prices: np.ndarray, times: np.ndarray, news_data: List[str], volumes: np.ndarray):
	print(f"🚀 Starting Enhanced Intraday SAC+Multi-Transformer Agent")
	print(f"📊 Config: {json.dumps(asdict(cfg), indent=2)}")
	print(f"💰 Starting Capital: ₹{cfg.starting_capital:,.2f}")
	print(f"📈 Data: {len(prices)} price points, {len(times)} time points")
	print(f"📰 News: {len([n for n in news_data if n])} news items")
	print(f"📊 Volumes: {len(volumes)} volume points")
	print(f"🎯 Device: {cfg.device}")
	print(f"🏗️ Multi-Transformer: {cfg.use_multi_transformer}")
	print(f"📈 Technical Indicators: {cfg.use_technical_indicators}")
	print(f"😊 Sentiment Analysis: {cfg.use_sentiment}")
	print(f"📝 Logging: {cfg.log_dir}")
	print("-" * 60)
	
	env = IntradayTradingEnv(prices, times, cfg, news_data, volumes)
	obs = env.reset()
	obs_shape = obs.shape  # [T, F]
	agent = SACAgent(obs_shape, cfg)
	replay = ReplayBuffer(cfg.replay_capacity)
	metrics = MetricsTracker()

	step = 0
	best_nav = cfg.starting_capital
	episode_count = 0
	
	while step < cfg.max_steps:
		action = agent.act(obs, explore=True)
		next_obs, reward, done, info = env.step(action)
		done_flag = 1.0 if done else 0.0
		
		# Track metrics
		nav = info.get("nav", env._mark_to_market()) if info is not None else env._mark_to_market()
		position_frac = 0.0 if nav <= 0 else (env.units * env.prices[env.t]) / nav
		metrics.update(nav, reward, action, position_frac)
		
		replay.add(obs, action, reward, next_obs, done_flag)
		obs = next_obs

		if len(replay) >= cfg.start_training_after:
			for _ in range(cfg.train_iterations_per_step):
				batch = replay.sample(cfg.batch_size)
				critic_loss, actor_loss, alpha_loss = agent.update(batch, step)

		if done:
			# Log episode summary
			env._log_episode_summary(episode_count, nav)
			metrics.end_episode(nav)
			obs = env.reset()
			episode_count += 1

		if (step + 1) % cfg.print_every == 0:
			best_nav = max(best_nav, nav)
			stats = metrics.get_stats()
			alpha_info = f"α={agent.alpha:.3f}" if cfg.auto_entropy else f"α={cfg.alpha:.3f}"
			print(f"Step {step+1:6d} | NAV: ₹{nav:8,.2f} | Best: ₹{best_nav:8,.2f} | "
				  f"Return: {stats.get('total_return_pct', 0):+6.2f}% | "
				  f"Sharpe: {stats.get('sharpe_ratio', 0):5.2f} | "
				  f"Episodes: {episode_count:3d} | Buffer: {len(replay):6d} | {alpha_info}")

		step += 1
	
	# Final stats
	final_stats = metrics.get_stats()
	print("\n" + "="*60)
	print("📊 FINAL PERFORMANCE METRICS")
	print("="*60)
	for key, value in final_stats.items():
		if "pct" in key:
			print(f"{key:25s}: {value:8.2f}%")
		elif "ratio" in key:
			print(f"{key:25s}: {value:8.2f}")
		elif "time" in key:
			print(f"{key:25s}: {value:8.1f} min")
		else:
			print(f"{key:25s}: {value:8.2f}")
	print("="*60)
	print(f"📝 Logs saved to: {cfg.log_dir}")
	print("   - timestep_logs.csv: Per-timestep details")
	print("   - episode_logs.csv: Episode summaries")
	print("   - trade_logs.csv: Individual trade details")


def validate_config(cfg: Config) -> List[str]:
	"""Validate configuration parameters and return list of issues."""
	issues = []
	
	if cfg.starting_capital <= 0:
		issues.append("starting_capital must be positive")
	if cfg.trailing_stop_pct <= 0 or cfg.trailing_stop_pct >= 1:
		issues.append("trailing_stop_pct must be in (0, 1)")
	if cfg.transaction_brokerage_pct < 0:
		issues.append("transaction_brokerage_pct must be non-negative")
	if cfg.transaction_slippage_bps_open_close < 0 or cfg.transaction_slippage_bps_mid < 0:
		issues.append("slippage values must be non-negative")
	if cfg.rolling_vol_window < 2:
		issues.append("rolling_vol_window must be at least 2")
	if cfg.obs_lookback < 1:
		issues.append("obs_lookback must be at least 1")
	if cfg.max_position_fraction <= 0 or cfg.max_position_fraction > 1:
		issues.append("max_position_fraction must be in (0, 1]")
	if cfg.gamma <= 0 or cfg.gamma >= 1:
		issues.append("gamma must be in (0, 1)")
	if cfg.tau <= 0 or cfg.tau >= 1:
		issues.append("tau must be in (0, 1)")
	if cfg.alpha <= 0:
		issues.append("alpha must be positive")
	if cfg.auto_entropy and cfg.target_entropy >= 0:
		issues.append("target_entropy must be negative for SAC++")
	if cfg.auto_entropy and cfg.alpha_lr <= 0:
		issues.append("alpha_lr must be positive when auto_entropy is enabled")
	if cfg.actor_lr <= 0 or cfg.critic_lr <= 0:
		issues.append("learning rates must be positive")
	if cfg.risk_penalty_lambda < 0:
		issues.append("risk_penalty_lambda must be non-negative")
	if cfg.sharpe_bonus_weight < 0:
		issues.append("sharpe_bonus_weight must be non-negative")
	if cfg.batch_size <= 0:
		issues.append("batch_size must be positive")
	if cfg.replay_capacity <= 0:
		issues.append("replay_capacity must be positive")
	if cfg.start_training_after < 0:
		issues.append("start_training_after must be non-negative")
	if cfg.train_iterations_per_step <= 0:
		issues.append("train_iterations_per_step must be positive")
	if cfg.critic_ensemble < 1:
		issues.append("critic_ensemble must be at least 1")
	if cfg.max_steps <= 0:
		issues.append("max_steps must be positive")
	if cfg.print_every <= 0:
		issues.append("print_every must be positive")
	
	return issues


def main():
	if except_import_error is not None:
		print("❌ PyTorch is required to run this script. Install torch first.")
		print(f"Import error: {except_import_error}")
		return

	parser = argparse.ArgumentParser(
		description="Single-file Intraday SAC+Transformer Agent",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	
	# Trading parameters
	parser.add_argument("--capital", type=float, default=100000.0,
						help="Starting capital in ₹")
	parser.add_argument("--trailing_stop_pct", type=float, default=0.01,
						help="Trailing stop percentage (0.01 = 1%%)")
	parser.add_argument("--brokerage_pct", type=float, default=0.0005,
						help="Brokerage fee percentage (0.0005 = 5 bps)")
	parser.add_argument("--slippage_open_close", type=float, default=4.0,
						help="Slippage in bps during open/close")
	parser.add_argument("--slippage_mid", type=float, default=1.5,
						help="Slippage in bps during mid-day")
	
	# Data parameters
	parser.add_argument("--days", type=int, default=20,
						help="Number of trading days to simulate")
	parser.add_argument("--minutes_per_day", type=int, default=390,
						help="Minutes per trading day")
	parser.add_argument("--vol_window", type=int, default=20,
						help="Rolling volatility window")
	parser.add_argument("--lookback", type=int, default=60,
						help="Observation lookback window")
	
	# Training parameters
	parser.add_argument("--max_steps", type=int, default=50000,
						help="Maximum training steps")
	parser.add_argument("--batch_size", type=int, default=128,
						help="Training batch size")
	parser.add_argument("--replay_capacity", type=int, default=200000,
						help="Replay buffer capacity")
	parser.add_argument("--start_training_after", type=int, default=2000,
						help="Start training after this many steps")
	parser.add_argument("--print_every", type=int, default=1000,
						help="Print progress every N steps")
	
	# Model parameters
	parser.add_argument("--critic_ensemble", type=int, default=5,
						help="Number of critic networks")
	parser.add_argument("--gamma", type=float, default=0.99,
						help="Discount factor")
	parser.add_argument("--tau", type=float, default=0.005,
						help="Soft update coefficient")
	parser.add_argument("--alpha", type=float, default=0.2,
						help="Initial entropy coefficient")
	parser.add_argument("--auto_entropy", action="store_true", default=True,
						help="Enable SAC++ automatic entropy tuning")
	parser.add_argument("--no_auto_entropy", dest="auto_entropy", action="store_false",
						help="Disable automatic entropy tuning")
	parser.add_argument("--target_entropy", type=float, default=-1.0,
						help="Target entropy for automatic tuning")
	parser.add_argument("--alpha_lr", type=float, default=3e-4,
						help="Learning rate for entropy coefficient")
	parser.add_argument("--actor_lr", type=float, default=3e-4,
						help="Actor learning rate")
	parser.add_argument("--critic_lr", type=float, default=3e-4,
						help="Critic learning rate")
	
	# Risk-adjusted reward parameters
	parser.add_argument("--risk_penalty_lambda", type=float, default=0.1,
						help="Penalty weight for drawdowns")
	parser.add_argument("--sharpe_bonus_weight", type=float, default=0.05,
						help="Bonus weight for Sharpe ratio")
	parser.add_argument("--use_risk_adjusted_rewards", action="store_true", default=True,
						help="Enable risk-adjusted reward shaping")
	parser.add_argument("--no_risk_adjusted_rewards", dest="use_risk_adjusted_rewards", action="store_false",
						help="Disable risk-adjusted reward shaping")
	
	# Multi-transformer architecture parameters
	parser.add_argument("--use_multi_transformer", action="store_true", default=True,
						help="Enable multi-stream transformer architecture")
	parser.add_argument("--no_multi_transformer", dest="use_multi_transformer", action="store_false",
						help="Disable multi-stream transformer architecture")
	parser.add_argument("--transformer_d_model", type=int, default=64,
						help="Transformer model dimension")
	parser.add_argument("--transformer_nhead", type=int, default=4,
						help="Number of attention heads")
	parser.add_argument("--transformer_num_layers", type=int, default=2,
						help="Number of transformer layers")
	
	# Technical indicators parameters
	parser.add_argument("--use_technical_indicators", action="store_true", default=True,
						help="Enable technical indicators")
	parser.add_argument("--no_technical_indicators", dest="use_technical_indicators", action="store_false",
						help="Disable technical indicators")
	parser.add_argument("--indicator_window", type=int, default=20,
						help="Window size for technical indicators")
	
	# Sentiment analysis parameters
	parser.add_argument("--use_sentiment", action="store_true", default=True,
						help="Enable sentiment analysis")
	parser.add_argument("--no_sentiment", dest="use_sentiment", action="store_false",
						help="Disable sentiment analysis")
	parser.add_argument("--sentiment_model_name", type=str, default="ProsusAI/finbert",
						help="Sentiment analysis model name")
	parser.add_argument("--sentiment_window", type=int, default=10,
						help="Window size for sentiment moving average")
	
	# Logging parameters
	parser.add_argument("--log_dir", type=str, default="trading_logs",
						help="Directory for logging files")
	parser.add_argument("--log_timestep_details", action="store_true", default=True,
						help="Enable detailed timestep logging")
	parser.add_argument("--no_log_timestep_details", dest="log_timestep_details", action="store_false",
						help="Disable detailed timestep logging")
	parser.add_argument("--log_episode_summaries", action="store_true", default=True,
						help="Enable episode summary logging")
	parser.add_argument("--no_log_episode_summaries", dest="log_episode_summaries", action="store_false",
						help="Disable episode summary logging")
	parser.add_argument("--log_trade_decisions", action="store_true", default=True,
						help="Enable trade decision logging")
	parser.add_argument("--no_log_trade_decisions", dest="log_trade_decisions", action="store_false",
						help="Disable trade decision logging")
	
	# System parameters
	parser.add_argument("--seed", type=int, default=42,
						help="Random seed")
	parser.add_argument("--device", type=str, default="auto",
						help="Device (auto/cpu/cuda)")
	
	args = parser.parse_args()
	
	# Set device
	if args.device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device
	
	cfg = Config(
		starting_capital=args.capital,
		trailing_stop_pct=args.trailing_stop_pct,
		transaction_brokerage_pct=args.brokerage_pct,
		transaction_slippage_bps_open_close=args.slippage_open_close,
		transaction_slippage_bps_mid=args.slippage_mid,
		rolling_vol_window=args.vol_window,
		obs_lookback=args.lookback,
		seed=args.seed,
		device=device,
		gamma=args.gamma,
		tau=args.tau,
		alpha=args.alpha,
		auto_entropy=args.auto_entropy,
		target_entropy=args.target_entropy,
		alpha_lr=args.alpha_lr,
		actor_lr=args.actor_lr,
		critic_lr=args.critic_lr,
		batch_size=args.batch_size,
		replay_capacity=args.replay_capacity,
		start_training_after=args.start_training_after,
		train_iterations_per_step=1,
		critic_ensemble=args.critic_ensemble,
		max_steps=args.max_steps,
		print_every=args.print_every,
		risk_penalty_lambda=args.risk_penalty_lambda,
		sharpe_bonus_weight=args.sharpe_bonus_weight,
		use_risk_adjusted_rewards=args.use_risk_adjusted_rewards,
		# Multi-transformer architecture
		use_multi_transformer=args.use_multi_transformer,
		transformer_d_model=args.transformer_d_model,
		transformer_nhead=args.transformer_nhead,
		transformer_num_layers=args.transformer_num_layers,
		# Technical indicators
		use_technical_indicators=args.use_technical_indicators,
		indicator_window=args.indicator_window,
		# Sentiment analysis
		use_sentiment=args.use_sentiment,
		sentiment_model_name=args.sentiment_model_name,
		sentiment_window=args.sentiment_window,
		# Logging
		log_dir=args.log_dir,
		log_timestep_details=args.log_timestep_details,
		log_episode_summaries=args.log_episode_summaries,
		log_trade_decisions=args.log_trade_decisions,
	)
	
	# Validate configuration
	issues = validate_config(cfg)
	if issues:
		print("❌ Configuration validation failed:")
		for issue in issues:
			print(f"  - {issue}")
		return
	
	# Set seeds
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	
	# Generate data and train
	prices, times, news_data, volumes = generate_synthetic_intraday(
		days=args.days, 
		minutes_per_day=args.minutes_per_day, 
		seed=args.seed
	)
	train(cfg, prices, times, news_data, volumes)


if __name__ == "__main__":
	main()


