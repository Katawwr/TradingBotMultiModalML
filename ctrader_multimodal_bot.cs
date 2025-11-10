using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class MultiModalAIBot : Robot
    {
        #region Parameters
        
        [Parameter("Python API URL", DefaultValue = "http://localhost:5000")]
        public string ApiUrl { get; set; }
        
        [Parameter("Symbol to Trade", DefaultValue = "US100" )]
        public string TradingSymbol { get; set; }
        
        [Parameter("Base Risk %", DefaultValue = 1.0, MinValue = 0.5, MaxValue = 3.0)]
        public double BaseRiskPercent { get; set; }
        
        [Parameter("Max Positions", DefaultValue = 3, MinValue = 1)]
        public int MaxPositions { get; set; }
        
        [Parameter("Min Confidence", DefaultValue = 0.70, MinValue = 0.50)]
        public double MinConfidence { get; set; }
        
        [Parameter("Query Interval (Seconds)", DefaultValue = 30, MinValue = 10)]
        public int QueryIntervalSeconds { get; set; }
        
        [Parameter("Enable Auto Trading", DefaultValue = false)]
        public bool EnableAutoTrading { get; set; }
        
        [Parameter("Max Daily Loss %", DefaultValue = 3.0)]
        public double MaxDailyLossPercent { get; set; }
        
        [Parameter("Debug Mode", DefaultValue = true)]
        public bool DebugMode { get; set; }
        
        #endregion
        
        #region Private Variables
        
        private HttpClient _httpClient;
        private DateTime _lastQueryTime;
        private double _dailyStartBalance;
        private Dictionary<int, AITradeContext> _tradeContexts;
        
        // Performance tracking
        private int _totalSignals = 0;
        private int _tradesExecuted = 0;
        private int _winningTrades = 0;
        private Dictionary<string, int> _layerSuccessCount;
        
        // Indicators for fallback
        private ExponentialMovingAverage _ema9;
        private ExponentialMovingAverage _ema21;
        private RelativeStrengthIndex _rsi;
        private AverageTrueRange _atr;
        
        #endregion
        
        #region Data Classes
        
        private class AISignalResponse
        {
            public string Symbol { get; set; }
            public string Signal { get; set; }
            public double Confidence { get; set; }
            public Dictionary<string, double> LayerScores { get; set; }
            public Dictionary<string, double> LayerWeights { get; set; }
            public RiskParameters RiskParams { get; set; }
            public string Reasoning { get; set; }
            public DateTime Timestamp { get; set; }
        }
        
        private class RiskParameters
        {
            public double PositionSizePct { get; set; }
            public double StopLossAtrMultiple { get; set; }
            public double TakeProfitAtrMultiple { get; set; }
            public int MaxHoldMinutes { get; set; }
            public bool TrailingStopEnabled { get; set; }
        }
        
        private class AITradeContext
        {
            public DateTime EntryTime { get; set; }
            public double Confidence { get; set; }
            public Dictionary<string, double> LayerScores { get; set; }
            public string Reasoning { get; set; }
            public double EntryPrice { get; set; }
            public int MaxHoldMinutes { get; set; }
        }
        
        #endregion
        
        protected override void OnStart()
        {
            // Initialize HTTP client
            _httpClient = new HttpClient();
            _httpClient.Timeout = TimeSpan.FromSeconds(10);
            
            // Initialize tracking
            _lastQueryTime = Server.Time.AddSeconds(-QueryIntervalSeconds);
            _dailyStartBalance = Account.Balance;
            _tradeContexts = new Dictionary<int, AITradeContext>();
            _layerSuccessCount = new Dictionary<string, int>
            {
                {"social_sentiment", 0},
                {"news_nlp", 0},
                {"lstm_prediction", 0},
                {"ensemble_ml", 0},
                {"technical", 0}
            };
            
            // Initialize fallback indicators
            _ema9 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 9);
            _ema21 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 21);
            _rsi = Indicators.RelativeStrengthIndex(Bars.ClosePrices, 14);
            _atr = Indicators.AverageTrueRange(14, MovingAverageType.Exponential);
            
            // Event subscriptions
            Positions.Opened += OnPositionOpened;
            Positions.Closed += OnPositionClosed;
            
            Print("═══════════════════════════════════════════════════════════");
            Print($"  MULTI-MODAL AI TRADING BOT");
            Print("═══════════════════════════════════════════════════════════");
            Print($"  Python API: {ApiUrl}");
            Print($"  Symbol: {TradingSymbol}");
            Print($"  Auto Trading: {(EnableAutoTrading ? "✓ ENABLED" : "✗ DISABLED (Manual Mode)")}");
            Print($"  Min Confidence: {MinConfidence:P0}");
            Print($"  Query Interval: {QueryIntervalSeconds}s");
            Print($"  Max Positions: {MaxPositions}");
            Print("═══════════════════════════════════════════════════════════\n");
            
            // Test API connection
            TestAPIConnection();
        }
        
        protected override void OnTick()
        {
            // Manage existing positions
            ManagePositions();
        }
        
        protected override void OnBar()
        {
            // Daily reset
            if (Server.Time.Date != _lastQueryTime.Date)
            {
                ResetDaily();
            }
            
            // Check if it's time to query API
            var timeSinceLastQuery = (Server.Time - _lastQueryTime).TotalSeconds;
            if (timeSinceLastQuery >= QueryIntervalSeconds)
            {
                QueryAISignal();
                _lastQueryTime = Server.Time;
            }
            
            // Performance reporting
            if (DebugMode && Server.Time.Minute % 15 == 0 && Server.Time.Second == 0)
            {
                PrintPerformance();
            }
        }
        
        private async void TestAPIConnection()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{ApiUrl}/api/health");
                if (response.IsSuccessStatusCode)
                {
                    Print("✅ Python API connection successful");
                }
                else
                {
                    Print($"⚠️ API returned status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Print($"❌ Failed to connect to Python API: {ex.Message}");
                Print("   Make sure Python backend is running at " + ApiUrl);
            }
        }
        
        private async void QueryAISignal()
        {
            try
            {
                // Query Python API for signal
                var url = $"{ApiUrl}/api/signal/{TradingSymbol}";
                var response = await _httpClient.GetAsync(url);
                
                if (!response.IsSuccessStatusCode)
                {
                    if (DebugMode)
                        Print($"⚠️ API query failed: {response.StatusCode}");
                    return;
                }
                
                var json = await response.Content.ReadAsStringAsync();
                var signal = JsonSerializer.Deserialize<AISignalResponse>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
                
                _totalSignals++;
                
                // Process signal
                ProcessAISignal(signal);
            }
            catch (HttpRequestException ex)
            {
                if (DebugMode)
                    Print($"⚠️ Network error querying API: {ex.Message}");
            }
            catch (Exception ex)
            {
                Print($"❌ Error processing AI signal: {ex.Message}");
            }
        }
        
        private void ProcessAISignal(AISignalResponse signal)
        {
            if (signal == null) return;
            
            // Check confidence threshold
            if (signal.Confidence < MinConfidence)
            {
                if (DebugMode)
                    Print($"⏸️ Signal ignored - Low confidence: {signal.Confidence:P0} < {MinConfidence:P0}");
                return;
            }
            
            // Check if we can trade
            if (!CanTrade())
            {
                if (DebugMode)
                    Print($"⏸️ Cannot trade - Max positions: {Positions.Count}/{MaxPositions}");
                return;
            }
            
            // Check risk limits
            if (!CheckRiskLimits())
            {
                Print("🛑 Daily risk limit reached - No new trades");
                return;
            }
            
            // Log signal
            if (DebugMode)
            {
                Print($"\n{'─',60}");
                Print($"🤖 AI SIGNAL RECEIVED");
                Print($"{'─',60}");
                Print($"  Direction: {signal.Signal}");
                Print($"  Confidence: {signal.Confidence:P1}");
                Print($"  Timestamp: {signal.Timestamp:HH:mm:ss}");
                Print($"\n  Layer Scores:");
                foreach (var layer in signal.LayerScores.OrderByDescending(x => Math.Abs(x.Value)))
                {
                    var arrow = layer.Value > 0 ? "↑" : "↓";
                    Print($"    {arrow} {layer.Key}: {layer.Value:+0.00;-0.00}");
                }
                Print($"\n  Reasoning:\n{signal.Reasoning}");
                Print($"{'─',60}\n");
            }
            
            // Execute trade if auto trading enabled
            if (EnableAutoTrading && (signal.Signal == "BUY" || signal.Signal == "SELL"))
            {
                ExecuteAITrade(signal);
            }
            else if (!EnableAutoTrading)
            {
                Print($"ℹ️ Manual mode - Signal: {signal.Signal} @ {signal.Confidence:P0} confidence");
            }
        }
        
        private void ExecuteAITrade(AISignalResponse signal)
        {
            var direction = signal.Signal == "BUY" ? TradeType.Buy : TradeType.Sell;
            
            // Calculate position size
            var volume = CalculatePositionSize(signal.RiskParams);
            
            // Calculate SL/TP using ATR
            var atrValue = _atr.Result.LastValue;
            var stopLossPips = atrValue * signal.RiskParams.StopLossAtrMultiple / Symbol.PipSize;
            var takeProfitPips = atrValue * signal.RiskParams.TakeProfitAtrMultiple / Symbol.PipSize;
            
            // Execute order
            var label = $"AI_{signal.Signal}_{signal.Confidence:F2}";
            var result = ExecuteMarketOrder(direction, Symbol.Name, volume, label, 
                (int)stopLossPips, (int)takeProfitPips);
            
            if (result.IsSuccessful)
            {
                _tradesExecuted++;
                
                // Store context for tracking
                _tradeContexts[result.Position.Id] = new AITradeContext
                {
                    EntryTime = Server.Time,
                    Confidence = signal.Confidence,
                    LayerScores = signal.LayerScores,
                    Reasoning = signal.Reasoning,
                    EntryPrice = result.Position.EntryPrice,
                    MaxHoldMinutes = signal.RiskParams.MaxHoldMinutes
                };
                
                Print($"✅ {direction} executed: {volume} units @ {result.Position.EntryPrice}");
                Print($"   SL: {stopLossPips:F1} pips | TP: {takeProfitPips:F1} pips");
            }
            else
            {
                Print($"❌ Order failed: {result.Error}");
            }
        }
        
        private void ManagePositions()
        {
            foreach (var position in Positions)
            {
                if (!position.Label.StartsWith("AI_")) continue;
                
                var context = _tradeContexts.ContainsKey(position.Id) ? _tradeContexts[position.Id] : null;
                if (context == null) continue;
                
                // Time-based exit
                var holdingMinutes = (Server.Time - context.EntryTime).TotalMinutes;
                if (holdingMinutes > context.MaxHoldMinutes)
                {
                    ClosePosition(position);
                    Print($"⏰ Time exit: {holdingMinutes:F0}min > {context.MaxHoldMinutes}min");
                    continue;
                }
                
                // Trailing stop (if enabled in risk params)
                // Would need to parse from label or store in context
                var profitPips = position.Pips;
                if (profitPips > 15)
                {
                    var currentPrice = position.TradeType == TradeType.Buy ? Symbol.Bid : Symbol.Ask;
                    var atrValue = _atr.Result.LastValue;
                    var trailingDistance = atrValue * 1.5;
                    
                    var newStop = position.TradeType == TradeType.Buy ?
                        currentPrice - trailingDistance :
                        currentPrice + trailingDistance;
                    
                    if (position.StopLoss.HasValue)
                    {
                        if ((position.TradeType == TradeType.Buy && newStop > position.StopLoss) ||
                            (position.TradeType == TradeType.Sell && newStop < position.StopLoss))
                        {
                            ModifyPosition(position, newStop, position.TakeProfit);
                        }
                    }
                }
            }
        }
        
        private double CalculatePositionSize(RiskParameters riskParams)
        {
            // Use AI-suggested position size
            var riskAmount = Account.Balance * (riskParams.PositionSizePct);
            var atrValue = _atr.Result.LastValue;
            var stopDistance = atrValue * riskParams.StopLossAtrMultiple;
            
            if (stopDistance <= 0) return Symbol.VolumeInUnitsMin;
            
            var size = riskAmount / (stopDistance / Symbol.TickSize * Symbol.TickValue);
            var volume = Symbol.NormalizeVolumeInUnits(size);
            
            return Math.Max(Symbol.VolumeInUnitsMin, Math.Min(Symbol.VolumeInUnitsMax, volume));
        }
        
        private bool CheckRiskLimits()
        {
            var dailyPnL = Account.Balance - _dailyStartBalance;
            var maxLoss = _dailyStartBalance * (MaxDailyLossPercent / 100);
            
            return dailyPnL > -maxLoss;
        }
        
        private bool CanTrade()
        {
            return Positions.Count < MaxPositions;
        }
        
        private void ResetDaily()
        {
            _dailyStartBalance = Account.Balance;
            
            if (DebugMode)
            {
                Print($"\n{'═',60}");
                Print($"  📅 NEW TRADING DAY - {Server.Time:yyyy-MM-dd}");
                Print($"  Starting Balance: {_dailyStartBalance:C}");
                if (_totalSignals > 0)
                {
                    var executionRate = (_tradesExecuted / (double)_totalSignals) * 100;
                    var winRate = _tradesExecuted > 0 ? (_winningTrades / (double)_tradesExecuted) * 100 : 0;
                    Print($"  Yesterday: {_totalSignals} signals, {_tradesExecuted} trades ({executionRate:F0}% execution)");
                    Print($"  Win Rate: {winRate:F0}% ({_winningTrades}W/{_tradesExecuted-_winningTrades}L)");
                }
                Print($"{'═',60}\n");
            }
            
            _totalSignals = 0;
            _tradesExecuted = 0;
            _winningTrades = 0;
        }
        
        private void OnPositionOpened(PositionOpenedEventArgs args)
        {
            if (!args.Position.Label.StartsWith("AI_")) return;
            
            if (DebugMode)
            {
                var context = _tradeContexts.ContainsKey(args.Position.Id) ? _tradeContexts[args.Position.Id] : null;
                if (context != null)
                {
                    Print($"📈 Position opened: {args.Position.TradeType} | Confidence: {context.Confidence:P0}");
                }
            }
        }
        
        private void OnPositionClosed(PositionClosedEventArgs args)
        {
            if (!args.Position.Label.StartsWith("AI_")) return;
            
            var position = args.Position;
            var isWin = position.NetProfit > 0;
            
            if (isWin) _winningTrades++;
            
            // Update layer success tracking
            if (_tradeContexts.ContainsKey(position.Id))
            {
                var context = _tradeContexts[position.Id];
                
                // Find strongest layer signal
                if (context.LayerScores != null && context.LayerScores.Count > 0)
                {
                    var strongestLayer = context.LayerScores.OrderByDescending(x => Math.Abs(x.Value)).First();
                    if (isWin && _layerSuccessCount.ContainsKey(strongestLayer.Key))
                    {
                        _layerSuccessCount[strongestLayer.Key]++;
                    }
                }
                
                // Send feedback to Python API (async, don't wait)
                SendFeedbackToAPI(position, context, isWin);
                
                _tradeContexts.Remove(position.Id);
            }
            
            if (DebugMode)
            {
                var icon = isWin ? "✅" : "❌";
                var duration = (Server.Time - position.EntryTime).TotalMinutes;
                Print($"{icon} Closed: {position.NetProfit:C} ({position.Pips:F1}pips) | {duration:F0}min");
            }
        }
        
        private async void SendFeedbackToAPI(Position position, AITradeContext context, bool isWin)
        {
            try
            {
                var feedback = new
                {
                    position_id = position.Id,
                    symbol = position.SymbolName,
                    direction = position.TradeType.ToString(),
                    entry_price = position.EntryPrice,
                    exit_price = position.ClosePrice,
                    profit = position.NetProfit,
                    pips = position.Pips,
                    duration_minutes = (Server.Time - position.EntryTime).TotalMinutes,
                    was_win = isWin,
                    confidence = context.Confidence,
                    layer_scores = context.LayerScores,
                    timestamp = Server.Time
                };
                
                var json = JsonSerializer.Serialize(feedback);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
                
                await _httpClient.PostAsync($"{ApiUrl}/api/feedback", content);
            }
            catch (Exception ex)
            {
                // Silently fail - feedback is nice to have but not critical
                if (DebugMode)
                    Print($"⚠️ Failed to send feedback: {ex.Message}");
            }
        }
        
        private void PrintPerformance()
        {
            var dailyPnL = Account.Balance - _dailyStartBalance;
            var executionRate = _totalSignals > 0 ? (_tradesExecuted / (double)_totalSignals) * 100 : 0;
            var winRate = _tradesExecuted > 0 ? (_winningTrades / (double)_tradesExecuted) * 100 : 0;
            
            Print($"\n{'─',60}");
            Print($"  📊 PERFORMANCE ({Server.Time:HH:mm})");
            Print($"{'─',60}");
            Print($"  Balance: {Account.Balance:C} ({dailyPnL:+0.00;-0.00;0})");
            Print($"  Signals Received: {_totalSignals}");
            Print($"  Trades Executed: {_tradesExecuted} ({executionRate:F0}%)");
            Print($"  Win Rate: {winRate:F0}% ({_winningTrades}W/{_tradesExecuted-_winningTrades}L)");
            Print($"  Active Positions: {Positions.Count}");
            
            if (_layerSuccessCount.Values.Sum() > 0)
            {
                Print($"\n  Best Performing Layers:");
                foreach (var layer in _layerSuccessCount.OrderByDescending(x => x.Value).Take(3))
                {
                    if (layer.Value > 0)
                        Print($"    • {layer.Key}: {layer.Value} wins");
                }
            }
            
            Print($"{'─',60}\n");
        }
        
        protected override void OnStop()
        {
            Print($"\n{'═',70}");
            Print($"  MULTI-MODAL AI BOT - SESSION END");
            Print($"{'═',70}");
            Print($"  Final Balance: {Account.Balance:C}");
            Print($"  Session P&L: {Account.Balance - _dailyStartBalance:+0.00;-0.00;0}");
            Print($"  Total Signals: {_totalSignals}");
            Print($"  Trades Executed: {_tradesExecuted}");
            if (_tradesExecuted > 0)
            {
                var winRate = (_winningTrades / (double)_tradesExecuted) * 100;
                Print($"  Win Rate: {winRate:F1}%");
            }
            Print($"{'═',70}\n");
            
            _httpClient?.Dispose();
        }
    }
}
