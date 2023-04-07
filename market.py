import numpy as np
import pandas as pd
import itertools


def get_stock_data(companies, start_row=None, end_row=None):
    output = pd.DataFrame(columns=['Date'])
    is_first = True
    for company in companies:
        df = pd.read_csv(f'data/{company}.csv', usecols=['Date', 'Close'], nrows=end_row)
        if start_row is not None:
            df = df.iloc[start_row:]
        df = df.rename(columns = {'Close': company})
        if is_first:
            output = df.copy()
            is_first = False
        else:
            output = output.merge(df, on='Date')
    output = output.drop(columns = ['Date'])
    return output.values

class market:
    def __init__(self, companies, budget=1e4, start_row=None, end_row=None):
        self.data = get_stock_data(companies, start_row=start_row, end_row=end_row)
        self.budget = budget
        self.total_days = self.data.shape[0]
        self.total_companies = self.data.shape[1]
        self.index_actions = np.arange(self.total_companies**3) # 3 ações para cada empresa
        self.action_list = list(map(list,itertools.product([0,1,2],repeat=self.total_companies)))
        self.state_size = self.total_companies * 2 + 1
        self.start()

    def get_episode_value(self):
        return self._get_eval()

    def start(self):
        self.today = 0
        self.stocks = np.zeros(self.total_companies)
        self.stock_price = self.data[self.today]
        self.money_available = self.budget
        return self._get_state()

    def new_day(self, action):

        previous_val = self._get_eval()
        self.today += 1
        self.stock_price = self.data[self.today]
        self._exchange(action)
        current_val = self._get_eval()
        reward = current_val - previous_val
        done = self.today == (self.total_days - 1)
        return self._get_state(), reward, done

    def _exchange(self, action):
        actions = self.action_list[action] # [0,1,0]

        sell_list = []
        buy_list = []

        for i, action in enumerate(actions):
            if action == 0:
                sell_list.append(i)
            elif action ==2:
                buy_list.append(i)

        if sell_list:
            for i in sell_list:
                self.money_available += self.stock_price[i] * self.stocks[i]
                self.stocks[i] = 0

        if buy_list:
            broke = False
            cur_stock_prices = [self.stock_price[i] for i in buy_list]
            while not broke:
                for i in buy_list:
                    if self.money_available > self.stock_price[i]:
                        self.money_available -= self.stock_price[i]
                        self.stocks[i] += 1
                    elif all(self.money_available < cur_stock_prices):
                        broke = True

    
    def _get_eval(self):
        return self.stocks.dot(self.stock_price) + self.money_available



    def _get_state(self):
        state = np.zeros(self.state_size) #[0,0,0,0,0,0,0]
        state[:self.total_companies] = self.stocks
        state[self.total_companies:self.total_companies*2] = self.stock_price
        state[-1] = self.money_available
        return state
