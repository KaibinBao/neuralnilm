from __future__ import division
from copy import copy
import numpy as np


class Processor(object):
    def report(self):
        report = copy(self.__dict__)
        report['name'] = self.__class__.__name__
        return report

    def inverse(self, data):
        raise NotImplementedError("To be implemented by subclass.")

    def __call__(self, data):
        raise NotImplementedError("To be implemented by subclass.")


class DivideBy(Processor):
    def __init__(self, divisor):
        self.divisor = divisor

    def __call__(self, dividend):
        return dividend / self.divisor

    def inverse(self, quotient):
        return quotient * self.divisor


class Add(Processor):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, data):
        return data + self.amount

    def inverse(self, data):
        return data - self.amount

class Clip(Processor):
    def __init__(self, left_border, right_border):
        self.left_border = left_border
        self.right_border = right_border

    def __call__(self, data):
        return data.clip(self.left_border, self.right_border)

    def inverse(self, data):
        return data

class SelectCentralPoint(Processor):
    def __call__(self, data):
        center = int(data.shape[1] / 2)
        return data[:,center,:]

    def inverse(self, data):
        return data

class SubSequence(Processor):
    def __init__(self, from_i, to_i):
        self.from_i = from_i
        self.to_i = to_i

    def __call__(self, data):
        return data[:,self.from_i:self.to_i,:]

    def inverse(self, data):
        return data

class IndependentlyCenter(Processor):
    def __call__(self, data):
        means = data.mean(axis=1, keepdims=True)
        self.metadata = {'IndependentlyCentre': {'means': means}}
        return data - means


class AlignToBaseline(Processor):
    def __call__(self, data):
        baseline = data.min(axis=1, keepdims=True)
        return data - baseline


class AddNoise(Processor):
    def __init__(self, amount, rng):
        self.amount = amount
        self.rng = rng

    def __call__(self, data):
        noise = self.rng.randn(*data.shape) * self.amount
        return data + noise


class Transpose(Processor):
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, data):
        return data.transpose(self.axes)

    def inverse(self, data):
        return data.transpose(self.axes)


class DownSample(Processor):
    def __init__(self, downsample_factor, rng):
        self.downsample_factor = downsample_factor
        self.rng = rng

    def __call__(self, data):
        batch_size = data.shape[0]
        seq_length = data.shape[1]
        nc = data.shape[2]
        offset = self.rng.randint(low=0, high=self.downsample_factor)
        n_samples_new = int(np.ceil((seq_length+offset)/self.downsample_factor))
        left_pad, right_pad = offset, (n_samples_new*self.downsample_factor) - (offset+seq_length)
        resampled_data = np.pad(data, pad_width=[(0,0), (left_pad,right_pad), (0,0)], mode='edge')
        resampled_data = resampled_data.reshape(batch_size, n_samples_new, self.downsample_factor, nc)
        resampled_data[:, :, :, :] = resampled_data.mean(axis=2)[:, :, np.newaxis, :]
        resampled_data = resampled_data.reshape(
            batch_size, n_samples_new*self.downsample_factor, nc)[:,offset:offset+seq_length,:]
        return resampled_data

    def inverse(self, data):
        return data
