import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, C):
        self.X = X
        self.C = C
        weights = np.zeros((3,3))
        weights.fill(1)
        
        ones=np.ones((len(X),1))
        data=np.concatenate((ones,X),axis=1)
        
        ## Make one-hot matrix
        n_labels = 3
        targets = C
        #create an empty one-hot matrix
        ohm = np.zeros((targets.shape[0], n_labels))
        #set target idx to 1
        ohm[np.arange(targets.shape[0]), targets] = 1
        loss=[]
        weights_new=weights
        for i in xrange(500000):
            weights=weights_new
            sm_matrix = np.exp(np.dot(data,weights.T))
             ##test is the final softmax matrix
            test=sm_matrix/sm_matrix.sum(axis=1)[:,None]
            ## multiply softmax by ohm to calculate loss.
            loss.append(-sum(sum(np.log(test)*ohm)+self.lambda_parameter*sum(sum(weights**2))))
            gradient=np.dot((test-ohm).T,data)
            weights_new = weights-self.eta*(gradient+2*self.lambda_parameter*weights)

        self.weights=weights_new
        self.loss=loss
        return self.loss[-1]

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        ones=np.ones((len(X_to_predict),1))
        newX = np.concatenate((ones,X_to_predict),axis=1)
        
        return np.argmax(np.dot(newX,self.weights.T),axis=1)

    def plot(self):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        plt.figure()
        plt.plot(self.loss)
        plt.ylabel("Loss")
        plt.xlabel("Number of Iterations")
        plt.suptitle("Loss over iterations")

    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))
        
        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
