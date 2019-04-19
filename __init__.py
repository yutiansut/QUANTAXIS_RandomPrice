#
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def get_random_price(price, code='rb1905', tradingDay='20181119', mu=0, sigma=0.2, theta=0.15, dt=1e-2):
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.array(mu))
    plt.figure('data')
    y = []
    data = pd.read_csv('tick_temple.csv')

    for _, item in data.iterrows():
        item['InstrumentID'] = str(code)
        item['LastPrice'] = (ou_noise()+1)*0.2*price + 0.8*price

    return data


if __name__ == '__main__':
    print(get_random_price(3600))
