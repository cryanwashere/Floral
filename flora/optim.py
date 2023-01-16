class StochasticGradientDescent(object):
    def __init__(self, lr):
        self.lr = lr
    def optimize(self, param, grad):
        return param - (grad * self.lr)