import math
import numpy as np

class Gaussian_Distribution:
    def __init__(self, mean, std):
        '''
            This class represents a Gaussian distribution, also known as the normal distribution, characterized by a mean and standard deviation.

            @Arguments:
                
                mean [float]: The mean of the Gaussian distribution.
                
                std [float]: The standard deviation of the Gaussian distribution.

            @Public Methods:

                probability(x):
                    Calculate the probability density function (PDF) of the Gaussian distribution at a given point x.

        '''
        self.mean = mean
        self.std = std

    def probability(self, x):
        exponent = -((x - self.mean) ** 2) / (2 * self.std ** 2)
        prob = math.exp(exponent) # if * 1 / (\sqrt{2pi} * std) when std -> 0 prob -> inf
        return prob
    
class N_Gaussian_Distribution:
    def __init__(self, dim):
        '''
            N_Gaussian_Distribution - Class for handling multiple Gaussian distributions.

            @Arguments:
                dim [int]: Dimensionality of the Gaussian distributions.

            @Public Methods:
                probability(x): Calculates the joint probability of a data point under the multiple Gaussian distributions.
                    Arguments:
                        x [array-like]: Input data point.
                    Returns:
                        float: Joint probability of the data point under the Gaussian distributions.
                
                set_data_point(data_point, regular): Sets the data points and updates the parameters of the Gaussian distributions accordingly.
                    Arguments:
                        data_point [array-like]: Input data points.
                        regular [float]: Regularization threshold for standard deviation.
                    Returns:
                        None
        '''
        self.dim = dim
        self.gaussian_dist = [Gaussian_Distribution(0, 1) for _ in range(dim)]
        self.data_point = []
        self.N = 0
    
    def probability(self, x):
        if x.shape[0] != self.dim:
            raise "dimension not match."
        return math.prod([self.gaussian_dist[i].probability(x[i]) for i in range(self.dim)])
    
    def set_data_point(self, data_point, regular):
        self.data_point = data_point
        self.N = len(data_point)

        for i in range(self.dim):
            mean = np.mean(data_point[:, i])
            std = np.std(data_point[:, i])
            if std < regular:
                std = regular
            self.gaussian_dist[i].mean = mean
            self.gaussian_dist[i].std = std



        