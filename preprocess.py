import pandas as pd
import time
import ta.momentum
import ta.trend
import numpy as np
import pandas_ta
import warnings

warnings.filterwarnings("ignore")  # to Ignore the deprication Warnings

start = time.time()
nasdaq = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                       "Learning/Project/Data/NASDAQ (Jan 2003 - Apr 2022).xlsx")
nasdaq = nasdaq.reindex(index=nasdaq.index[::-1])
nasdaq.reset_index(inplace=True, drop=True)

dow = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/Project/Data/DOW 30 "
                    "(Jan 2003 - Apr 2022).xlsx")
dow = dow.reindex(index=dow.index[::-1])
dow.reset_index(inplace=True, drop=True)

sp500 = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/Project/Data/S_P "
                      "500 (Jan 2003 - Apr 2022).xlsx")
sp500 = sp500.reindex(index=sp500.index[::-1])
sp500.reset_index(inplace=True, drop=True)

one_year_treasury = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                  "Learning/Project/Data/1-year treasury constant maturity rate.xlsx")

libor = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                      "Learning/Project/Data/3-Month London Interbank Offered Rate (LIBOR).xlsx")

libor = libor.reindex(index=libor.index[::-1])
libor.reset_index(inplace=True, drop=True)

five_year_inflation = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                    "Learning/Project/Data/5-Year, 5-Year forward inflation expectation rate.xlsx")

ten_year_breakeven_inflation = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                             "Learning/Project/Data/10-year breakeven inflation rate.xlsx")

ten_year_treasury_rate = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                       "Learning/Project/Data/10-year treasury constant maturity rate.xlsx")

aaa = pd.read_excel(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/Project/Data/AAA.xlsx")

baa = pd.read_excel(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/Project/Data/BAA.xlsx")

china_us_exchange = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                  "Learning/Project/Data/China-U.S. foreign exchange rate.xlsx")

crude_oil_price = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                "Learning/Project/Data/Crude oil prices.xlsx")

usd_index_2006 = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                               "Learning/Project/Data/DTWEXBGS.xlsx")

usd_index_2003 = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                               "Learning/Project/Data/FRED-DTWEXM.xlsx")

usd_index_2003 = usd_index_2003.reindex(index=usd_index_2003.index[::-1])
usd_index_2003.reset_index(inplace=True, drop=True)
usd_index = pd.concat([usd_index_2003, usd_index_2006])
usd_index.reset_index(inplace=True, drop=True)

effective_federal_rate = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                       "Learning/Project/Data/Effective federal funds rate.xlsx")

gold_fixing_price = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                  "Learning/Project/Data/Gold fixing price.xlsx")

gold_fixing_price = gold_fixing_price.reindex(index=gold_fixing_price.index[::-1])
gold_fixing_price.reset_index(inplace=True, drop=True)

japan_us_exchange = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                  "Learning/Project/Data/Japan-U.S. foreign exchange rate.xlsx")

ted_rate = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                         "Learning/Project/Data/TEDRATE.xlsx")

euro_us_exchange = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                                 "Learning/Project/Data/U.S.- Euro foreign exchange rate.xlsx")

vix = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                    "Learning/Project/Data/^VIX.xlsx")

date_temp = pd.read_excel(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/Project/Data/date-temp.xlsx")


def rm_nan(d):
    """
        Returns a DataFrame Without Nan Values
    """
    if d.isnull().values.any():
        # print("we have Nan values")
        d = d.dropna()
    d.reset_index(inplace=True, drop=True)
    return d


def exponential_smoothing(d, alpha=0.095):
    """
        calculates the Exponential smoothing to eliminate noises.
    """

    if d.shape[1] == 2:
        # print("dim is 2")

        if d.iloc[0, 0] == pd.Timestamp(2003, 1, 2, 0) or d.iloc[0, 0] == pd.Timestamp(2003, 1, 1, 0):
            new_values = pd.DataFrame({'Close': [d.iloc[0, 1]]})
            # print("date is ascending")
            for i in range(1, d.shape[0]):
                new_values.at[i, 'Close'] = (alpha * d.iloc[i, 1]) + (
                        (1 - alpha) * d.iloc[i - 1, 1])
            return pd.concat([d['Date'], new_values], axis=1).rename(columns={'Close': d.columns.values[1]})
            # return new_values

        else:
            new_values = pd.DataFrame({'Close': [d.iloc[d.shape[0] - 1, 1]]})
            # print("date is descending")
            j = 1
            for i in range(d.shape[0] - 2, 0, -1):
                new_values.at[j, 'Close'] = (alpha * d.iloc[i, 1]) + (
                        (1 - alpha) * d.iloc[i + 1, 1])
                j += 1
            b = new_values.reindex(index=new_values.index[::-1])
            b.reset_index(inplace=True, drop=True)
            return pd.concat([d['Date'], b], axis=1).rename(columns={'Close': d.columns.values[1]})
            # return b

    else:
        # print("dim is bigger than 2")
        if d.iloc[0, 0] == pd.Timestamp(2003, 1, 2, 0) or d.iloc[0, 0] == pd.Timestamp(2003, 1, 1, 0):
            new_values = pd.DataFrame({'Open': [d.iloc[0, 1]],
                                       'High': [d.iloc[0, 2]],
                                       'Low': [d.iloc[0, 3]],
                                       'Close': [d.iloc[0, 4]]})
            # print("date is ascending")
            for i in range(1, d.shape[0]):
                new_values.at[i, 'Open'] = (alpha * d.iloc[i, 1]) + (
                        (1 - alpha) * d.iloc[i - 1, 1])
                new_values.at[i, 'High'] = (alpha * d.iloc[i, 2]) + (
                        (1 - alpha) * d.iloc[i - 1, 2])
                new_values.at[i, 'Low'] = (alpha * d.iloc[i, 3]) + (
                        (1 - alpha) * d.iloc[i - 1, 3])
                new_values.at[i, 'Close'] = (alpha * d.iloc[i, 4]) + (
                        (1 - alpha) * d.iloc[i - 1, 4])
            return pd.concat([d['Date'], new_values], axis=1)
            # return new_values
        else:
            new_values = pd.DataFrame({'Open': [d.iloc[d.shape[0] - 1, 1]],
                                       'High': [d.iloc[d.shape[0] - 1, 2]],
                                       'Low': [d.iloc[d.shape[0] - 1, 3]],
                                       'Close': [d.iloc[d.shape[0] - 1, 4]]})
            # print("date is descending")
            j = 1
            for i in range(d.shape[0] - 2, -1, -1):
                new_values.at[j, 'Open'] = (alpha * d.iloc[i, 1]) + (
                        (1 - alpha) * d.iloc[i + 1, 1])
                new_values.at[j, 'High'] = (alpha * d.iloc[i, 2]) + (
                        (1 - alpha) * d.iloc[i + 1, 2])
                new_values.at[j, 'Low'] = (alpha * d.iloc[i, 3]) + (
                        (1 - alpha) * d.iloc[i + 1, 3])
                new_values.at[j, 'Close'] = (alpha * d.iloc[i, 4]) + (
                        (1 - alpha) * d.iloc[i + 1, 4])
                j += 1
            b = new_values.reindex(index=new_values.index[::-1])
            b.reset_index(inplace=True, drop=True)
            return pd.concat([d['Date'], b], axis=1)
            # return b


def rsi(df, periods=14, ema=True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    rs = rsi.to_frame()
    rs = rs.rename(columns={'Close': 'rsi'})
    return pd.concat([df['Date'], rs], axis=1)


def stc(df):
    df.ta.stoch(high='high', low='low', k=14, d=3, append=True)
    aux = df[["Date", "STOCHk_14_3_3", "STOCHd_14_3_3"]]
    df.drop(df.iloc[:, 6:8], inplace=True, axis=1)

    return aux


def william_r(df):
    aux = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
    aux = aux.to_frame()
    aux = aux.rename(columns={'wr': 'williamr'})
    return pd.concat([df['Date'], aux], axis=1)


def macd(df):
    aux = ta.trend.macd(df['Close'])
    aux = aux.to_frame()
    aux = aux.rename(columns={'MACD_12_26': 'MACD'})
    return pd.concat([df['Date'], aux], axis=1)


def roc(df):
    aux = ta.momentum.roc(df['Close'], 20)
    aux = aux.to_frame()
    return pd.concat([df['Date'], aux], axis=1)


def cci(df):
    aux = ta.trend.cci(df['High'], df['Low'], df['Close'])
    aux = aux.to_frame()
    return pd.concat([df['Date'], aux], axis=1)


def month_to_day(df, temp=date_temp):
    rtn = df[0: 0]
    cnt = 0

    for i in range(2003, 2023):
        for j in range(1, 13):
            if cnt < df.shape[0]:
                year = temp.loc[temp['Date'].dt.year == i]
                month = year.loc[year['Date'].dt.month == j]
                month[df.columns.values[1]] = df.iloc[cnt, 1]
                rtn = pd.concat([rtn, month])
                cnt += 1
    rtn.reset_index(inplace=True, drop=True)
    return rtn


def label(df):
    lbl = pd.DataFrame(columns=['label'])
    for i in range(0, df.shape[0]):
        if i + 20 >= df.shape[0]:
            aux = None
        else:
            aux = np.sign((df.iloc[i + 20, 4] - df.iloc[i, 4]))
            if aux == -1:
                aux = 0
        lbl = lbl.append({'label': aux}, ignore_index=True)
    return pd.concat([df, lbl], axis=1)


nasdaq = rm_nan(nasdaq)
dow = rm_nan(dow)
sp500 = rm_nan(sp500)
one_year_treasury = rm_nan(one_year_treasury)
libor = rm_nan(libor)
five_year_inflation = rm_nan(five_year_inflation)
ten_year_breakeven_inflation = rm_nan(ten_year_breakeven_inflation)
ten_year_treasury_rate = rm_nan(ten_year_treasury_rate)
aaa = rm_nan(aaa)
baa = rm_nan(baa)
china_us_exchange = rm_nan(china_us_exchange)
crude_oil_price = rm_nan(crude_oil_price)
usd_index = rm_nan(usd_index)
effective_federal_rate = rm_nan(effective_federal_rate)
gold_fixing_price = rm_nan(gold_fixing_price)
japan_us_exchange = rm_nan(japan_us_exchange)
ted_rate = rm_nan(ted_rate)
euro_us_exchange = rm_nan(euro_us_exchange)
vix = rm_nan(vix)

nasdaq = exponential_smoothing(nasdaq)
nasdaq = label(nasdaq)
dow = exponential_smoothing(dow)
dow = label(dow)
sp500 = exponential_smoothing(sp500)
sp500 = label(sp500)
usd_index = exponential_smoothing(usd_index)
one_year_treasury = exponential_smoothing(one_year_treasury)
libor = exponential_smoothing(libor)
five_year_inflation = exponential_smoothing(five_year_inflation)
ten_year_breakeven_inflation = exponential_smoothing(ten_year_breakeven_inflation)
ten_year_treasury_rate = exponential_smoothing(ten_year_treasury_rate)
aaa = exponential_smoothing(aaa)
baa = exponential_smoothing(baa)
china_us_exchange = exponential_smoothing(china_us_exchange)
crude_oil_price = exponential_smoothing(crude_oil_price)
effective_federal_rate = exponential_smoothing(effective_federal_rate)
gold_fixing_price = exponential_smoothing(gold_fixing_price)
japan_us_exchange = exponential_smoothing(japan_us_exchange)
ted_rate = exponential_smoothing(ted_rate)
euro_us_exchange = exponential_smoothing(euro_us_exchange)
vix = exponential_smoothing(vix)

aaa = month_to_day(aaa)
baa = month_to_day(baa)
effective_federal_rate = month_to_day(effective_federal_rate)

rsi_nasdaq = rsi(nasdaq)
rsi_dow = rsi(dow)
rsi_sp500 = rsi(sp500)

rsi_nasdaq = rm_nan(rsi_nasdaq)
rsi_dow = rm_nan(rsi_dow)
rsi_sp500 = rm_nan(rsi_sp500)

stochastic_nasdaq = stc(nasdaq)
stochastic_dow = stc(dow)
stochastic_sp500 = stc(sp500)

stochastic_nasdaq = rm_nan(stochastic_nasdaq)
stochastic_dow = rm_nan(stochastic_dow)
stochastic_sp500 = rm_nan(stochastic_sp500)

wil_nasdaq = william_r(nasdaq)
wil_dow = william_r(dow)
wil_sp500 = william_r(sp500)

wil_nasdaq = rm_nan(wil_nasdaq)
wil_dow = rm_nan(wil_dow)
wil_sp500 = rm_nan(wil_sp500)

macd_nasdaq = macd(nasdaq)
macd_dow = macd(dow)
macd_sp500 = macd(sp500)

macd_nasdaq = rm_nan(macd_nasdaq)
macd_dow = rm_nan(macd_dow)
macd_sp500 = rm_nan(macd_sp500)

roc_nasdaq = roc(nasdaq)
roc_dow = roc(dow)
roc_sp500 = roc(sp500)

roc_nasdaq = rm_nan(roc_nasdaq)
roc_dow = rm_nan(roc_dow)
roc_sp500 = rm_nan(roc_sp500)

cci_nasdaq = cci(nasdaq)
cci_dow = cci(dow)
cci_sp500 = cci(sp500)

cci_nasdaq = rm_nan(cci_nasdaq)
cci_dow = rm_nan(cci_dow)
cci_sp500 = rm_nan(cci_sp500)

nasdaq_full = pd.merge(nasdaq, rsi_nasdaq, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, wil_nasdaq, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, macd_nasdaq, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, roc_nasdaq, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, cci_nasdaq, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, stochastic_nasdaq, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, vix, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, euro_us_exchange, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, ted_rate, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, japan_us_exchange, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, gold_fixing_price, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, crude_oil_price, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, china_us_exchange, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, ten_year_treasury_rate, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, ten_year_breakeven_inflation, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, five_year_inflation, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, libor, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, one_year_treasury, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, usd_index, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, effective_federal_rate, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, aaa, how='outer', on='Date')
nasdaq_full = pd.merge(nasdaq_full, baa, how='outer', on='Date')

dow_full = pd.merge(dow, rsi_nasdaq, how='outer', on='Date')
dow_full = pd.merge(dow_full, wil_nasdaq, how='outer', on='Date')
dow_full = pd.merge(dow_full, macd_nasdaq, how='outer', on='Date')
dow_full = pd.merge(dow_full, roc_nasdaq, how='outer', on='Date')
dow_full = pd.merge(dow_full, cci_nasdaq, how='outer', on='Date')
dow_full = pd.merge(dow_full, stochastic_nasdaq, how='outer', on='Date')
dow_full = pd.merge(dow_full, vix, how='outer', on='Date')
dow_full = pd.merge(dow_full, euro_us_exchange, how='outer', on='Date')
dow_full = pd.merge(dow_full, ted_rate, how='outer', on='Date')
dow_full = pd.merge(dow_full, japan_us_exchange, how='outer', on='Date')
dow_full = pd.merge(dow_full, gold_fixing_price, how='outer', on='Date')
dow_full = pd.merge(dow_full, crude_oil_price, how='outer', on='Date')
dow_full = pd.merge(dow_full, china_us_exchange, how='outer', on='Date')
dow_full = pd.merge(dow_full, ten_year_treasury_rate, how='outer', on='Date')
dow_full = pd.merge(dow_full, ten_year_breakeven_inflation, how='outer', on='Date')
dow_full = pd.merge(dow_full, five_year_inflation, how='outer', on='Date')
dow_full = pd.merge(dow_full, libor, how='outer', on='Date')
dow_full = pd.merge(dow_full, one_year_treasury, how='outer', on='Date')
dow_full = pd.merge(dow_full, usd_index, how='outer', on='Date')
dow_full = pd.merge(dow_full, effective_federal_rate, how='outer', on='Date')
dow_full = pd.merge(dow_full, aaa, how='outer', on='Date')
dow_full = pd.merge(dow_full, baa, how='outer', on='Date')

sp500_full = pd.merge(sp500, rsi_nasdaq, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, wil_nasdaq, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, macd_nasdaq, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, roc_nasdaq, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, cci_nasdaq, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, stochastic_nasdaq, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, vix, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, euro_us_exchange, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, ted_rate, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, japan_us_exchange, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, gold_fixing_price, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, crude_oil_price, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, china_us_exchange, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, ten_year_treasury_rate, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, ten_year_breakeven_inflation, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, five_year_inflation, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, libor, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, one_year_treasury, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, usd_index, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, effective_federal_rate, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, aaa, how='outer', on='Date')
sp500_full = pd.merge(sp500_full, baa, how='outer', on='Date')

nasdaq_full = rm_nan(nasdaq_full)

dow_full = rm_nan(dow_full)

sp500_full = rm_nan(sp500_full)

nasdaq_full.to_excel("nasdaq_full.xlsx")
dow_full.to_excel("dow_full.xlsx")
sp500_full.to_excel("sp500_full.xlsx")

end = time.time()
print(end - start)
