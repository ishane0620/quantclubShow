from datetime import datetime 
import backtrader as bt 
import backtrader.analyzers as btanalyzers
import numpy as np
import pandas as pd
# import pyfolio as pf
from joblib import Parallel, delayed
import logging
from backtesting import Backtest, Strategy 

def run_parallel(sma_period, rsi_buy, rsi_sell):
    return runtest(sma_period=sma_period, rsi_buy=rsi_buy, rsi_sell=rsi_sell)



class SortinoRatio(bt.Analyzer):
    def create_analysis(self):
        self.rets = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.rets.append(trade.pnlcomm)

    def get_analysis(self):
        returns = np.array(self.rets)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns)
        avg_return = np.mean(returns)
        sortino_ratio = avg_return / downside_deviation if downside_deviation else None
        return sortino_ratio
    
class SmaCross(bt.SignalStrategy):
    def __init__(self, period=20): 
        # when simple moving average crosses the price, can change the number
        sma = bt.ind.SMA(period=period)
        # this grabs the price data from the excel
        price = self.data
        # this defines the cross over.. price and sma
        crossover = bt.ind.CrossOver(price, sma)
        # this tells the code to LONG when it crossover, which is defined above
        self.signal_add(bt.SIGNAL_LONG, crossover)

class RsiCross(bt.SignalStrategy):
    def __init__(self, period=14, buy=30, sell=70): 
        self.rsi = bt.indicators.RSI(self.data, period=period)
        
        self.buy_level = buy
        self.sell_level = sell

    def next(self):
        if self.rsi[0] < self.buy_level and not self.position:
            self.buy()
        elif self.rsi[0] > self.sell_level and self.position:
            self.sell()

# class MixedIndicatorStrategy1(bt.SignalStrategy):
#     def __init__(self, sma_period=20, rsi_period=14, rsi_buy=30, rsi_sell=70):
#         self.sma = bt.indicators.SMA(period=sma_period)
#         price = self.data
#         self.crossover = bt.ind.CrossOver(price, self.sma) #if sma crossed (1, 0, or -1 value)
#         # self.signal_add(bt.SIGNAL_LONG, crossover)

#         self.rsi = bt.indicators.RSI(price, period=rsi_period)
#         self.buy_level = rsi_buy
#         self.sell_level = rsi_sell        
    
#     def next(self):
#         # Buy signal: RSI < rsi_buy and price above SMA
#         if self.rsi[0] < self.params.rsi_buy and self.data.close > self.sma[0]:
#             self.buy()

#         # Sell signal: RSI > rsi_sell and price below SMA
#         elif self.rsi[0] > self.params.rsi_sell or self.crossover[0] == -1:
#             self.sell()

class MixedIndicatorStrategy(bt.SignalStrategy):
    def __init__(self, sma_period=20, rsi_period=14, rsi_buy=30, rsi_sell=70):
        self.sma = bt.indicators.SMA(self.data.close, period=sma_period)
        self.crossover = bt.ind.CrossOver(self.data.close, self.sma)  # CrossOver indicator

        self.rsi = bt.indicators.RSI(self.data.close, period=rsi_period)
        self.buy_level = rsi_buy
        self.sell_level = rsi_sell        
    
    def next(self):
        # Buy signal: RSI < rsi_buy and price above SMA
        if (self.rsi[0] < self.buy_level or self.data.close[0] > self.sma[0]) and not self.position:
            self.buy()

        # Sell signal: RSI > rsi_sell or a bearish crossover occurs
        elif (self.rsi[0] > self.sell_level or self.crossover[0] == -1) and self.position:
            self.sell()



# Load data
data = bt.feeds.YahooFinanceCSVData(

    dataname = '/Users/isaac/Desktop/FTC_1/BTC-USD.csv',
   
    # do not pass values before this date
    # this is when we want to start the date
    fromdate=datetime(2019, 7, 29),
    # do not passs values after this date
    todate = datetime(2024, 7, 29), 
    reverse = False
)

results = []

def runtest(sma_period, rsi_period=14, rsi_buy=30, rsi_sell=70):
    cerebro = bt.Cerebro()

    cerebro.adddata(data)

    # Set initial capital
    cerebro.broker.set_cash(1000)
    cerebro.addsizer(bt.sizers.AllInSizer, percents=95)

    # Set commission
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.01, annualize=True)
    cerebro.addanalyzer(SortinoRatio, _name='sortino')
    cerebro.addanalyzer(btanalyzers.Calmar, _name='calmar')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.Transactions, _name = 'tx')
    # cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name = 'trades')

    # cerebro.addanalyzer(bt.analyzers.PyFolio)

    # Add strategy to cerebro with current parameters
    cerebro.addstrategy(MixedIndicatorStrategy, sma_period=sma_period, rsi_period=rsi_period, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
    
    
    # Run the backtest
    result = cerebro.run()

    # Extract performance metrics
    final_value = cerebro.broker.getvalue()
    sharpe = result[0].analyzers.sharpe.get_analysis()['sharperatio']
    sortino = result[0].analyzers.sortino.get_analysis()
    max_drawdown = result[0].analyzers.drawdown.get_analysis().max.drawdown
    txs = result[0].analyzers.tx.get_analysis() 
    tx_amt = len(txs)
    # pyfolio = result[0].analyzers.getbyname('pyfolio')
    # returns, positions, transactions, gross_lev = pyfolio.get_pf_items()
    # pf.create_full_tear_sheet(
    #     returns,
    #     positions=positions,
    #     transactions=transactions,
    #     gross_lev=gross_lev,
    #     live_start_date='2005-05-01',  # This date is sample specific
    #     round_trips=True)

    # print('SMA Period',sma_period,
    #     'RSI Buy',rsi_buy,
    #     'RSI Sell',rsi_sell,
    #     'Final Value',final_value,
    #     'Sharpe Ratio',sharpe,
    #     'Sortino Ratio',sortino,
    #     'Max Drawdown',max_drawdown,
    #     'Txs',tx_amt, flush=True)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logging.info(f"SMA Period {sma_period} RSI Buy {rsi_buy} RSI Sell {rsi_sell} Final Value {final_value} Sharpe Ratio {sharpe} Sortino Ratio {sortino} Max Drawdown {max_drawdown} Txs {tx_amt}")

    # Store the results
    results.append({
        'SMA Period': sma_period,
        'RSI Buy': rsi_buy,
        'RSI Sell': rsi_sell,
        'Final Value': final_value,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Txs': tx_amt
    })
    return {
        'SMA Period': sma_period,
        'RSI Buy': rsi_buy,
        'RSI Sell': rsi_sell,
        'Final Value': final_value,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Txs': tx_amt
    }

# # runtest(20)
# # cerebro.plot()

# sma_periods = range(18, 23)
# rsi_buy_levels = range(20, 31)
# rsi_sell_levels = range(70, 81)
# # Loop through all combinations of parameters
# # for sma_period in sma_periods:
# #     for rsi_buy in rsi_buy_levels:
# #         for rsi_sell in rsi_sell_levels:
# #             runtest(sma_period=sma_period, rsi_buy=rsi_buy, rsi_sell=rsi_sell)

# results = Parallel(n_jobs=-1)(
#     delayed(run_parallel)(sma_period, rsi_buy, rsi_sell)
#     for sma_period in sma_periods
#     for rsi_buy in rsi_buy_levels
#     for rsi_sell in rsi_sell_levels
# )


# # Convert the results to a pandas DataFrame
# df = pd.DataFrame(results)
# # Save the DataFrame to a CSV file for further analysis
# df.to_csv('backtest_results.csv', index=False)

# # Optionally, display the DataFrame
# print(df)

# best_result = df.loc[df['Final Value'].idxmax()]
# print(f"Best parameters: {best_result}")


data_df = pd.read_csv('/Users/isaac/Desktop/FTC_1/Data/BTC-USD.csv')
print(data_df)


import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover
import warnings 
warnings.filterwarnings('ignore')

class MixedStrat(Strategy):
    sma_period = 20
    rsi_period = 14
    rsi_buy = 30
    rsi_sell = 70

    def init(self):
        # Calculate indicators and ensure they return numpy arrays
        self.sma = self.I(lambda x: ta.sma(pd.Series(x), self.sma_period).to_numpy(), self.data.Close)
        self.rsi = self.I(lambda x: ta.rsi(pd.Series(x), self.rsi_period).to_numpy(), self.data.Close)

    def next(self):
        # Buy signal: RSI < rsi_buy and price above SMA
        if (self.rsi[-1] < self.rsi_buy or self.data.Close[-1] > self.sma[-1]) and not self.position:
            self.buy()

        # Sell signal: RSI > rsi_sell or a bearish crossover occurs
        elif (self.rsi[-1] > self.rsi_sell or crossover(self.data.Close, self.sma)) and self.position:
            self.sell()


data_df.index = pd.DatetimeIndex(data_df.index)
bt = Backtest(data_df.dropna(), MixedStrat , cash=1000000, commission=0.001)
stats = bt.run()
# bt.plot()

print(stats)