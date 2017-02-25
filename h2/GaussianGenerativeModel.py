from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X=X
        self.Y=Y
        prop1=len(X[Y==0])/float(len(X))
        prop2=len(X[Y==1])/float(len(X))
        prop3=len(X[Y==2])/float(len(X))
        
        mu1=np.mean(X[Y==0],axis=0)
        mu2=np.mean(X[Y==1],axis=0)
        mu3=np.mean(X[Y==2],axis=0)
        
        if self.isSharedCovariance==True:
            covmatrix = prop1*(np.cov(X[Y==0].T))+prop2*(np.cov(X[Y==1].T))+prop3*(np.cov(X[Y==2].T))
            self.covmatrix=covmatrix
        else:
            self.covmatrix1=np.cov(X[Y==0].T)
            self.covmatrix2=np.cov(X[Y==1].T)
            self.covmatrix3=np.cov(X[Y==2].T)
            
        self.mu1=mu1
        self.mu2=mu2
        self.mu3=mu3
        self.prop1=prop1
        self.prop2=prop2
        self.prop3=prop3

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        if self.isSharedCovariance==True:   
            likelihoods=[]
            likelihoods.append(np.log(self.prop1)+np.log(multivariate_normal.pdf(X_to_predict,self.mu1,self.covmatrix)))
            likelihoods.append(np.log(self.prop2)+np.log(multivariate_normal.pdf(X_to_predict,self.mu2,self.covmatrix)))
            likelihoods.append(np.log(self.prop3)+np.log(multivariate_normal.pdf(X_to_predict,self.mu3,self.covmatrix)))
            #self.likelihoods=likelihoods
            return np.argmax(np.array(likelihoods).T,axis=1)
        else:
            likelihoods=[]
            likelihoods.append(np.log(self.prop1)+np.log(multivariate_normal.pdf(X_to_predict,self.mu1,self.covmatrix1)))
            likelihoods.append(np.log(self.prop2)+np.log(multivariate_normal.pdf(X_to_predict,self.mu2,self.covmatrix2)))
            likelihoods.append(np.log(self.prop3)+np.log(multivariate_normal.pdf(X_to_predict,self.mu3,self.covmatrix3)))
            #self.likelihoods=likelihoods
            return np.argmax(np.array(likelihoods).T,axis=1)
            

    def returnlikelihoods(self):
        ## Make one-hot matrix
        n_labels = 3
        targets = self.Y
        #create an empty one-hot matrix
        ohm = np.zeros((targets.shape[0], n_labels))
        #set target idx to 1
        ohm[np.arange(targets.shape[0]), targets] = 1
        if self.isSharedCovariance==True:   
            likelihoods=[]
            likelihoods.append(np.log(self.prop1)+np.log(multivariate_normal.pdf(self.X,self.mu1,self.covmatrix)))
            likelihoods.append(np.log(self.prop2)+np.log(multivariate_normal.pdf(self.X,self.mu2,self.covmatrix)))
            likelihoods.append(np.log(self.prop3)+np.log(multivariate_normal.pdf(self.X,self.mu3,self.covmatrix)))
            self.likelihoods=likelihoods
        else:
            likelihoods=[]
            likelihoods.append(np.log(self.prop1)+np.log(multivariate_normal.pdf(self.X,self.mu1,self.covmatrix1)))
            likelihoods.append(np.log(self.prop2)+np.log(multivariate_normal.pdf(self.X,self.mu2,self.covmatrix2)))
            likelihoods.append(np.log(self.prop3)+np.log(multivariate_normal.pdf(self.X,self.mu3,self.covmatrix3)))
            self.likelihoods=likelihoods
        
        temp=sum(sum(np.array(self.likelihoods).T*ohm))
        return temp
        

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .005), np.arange(y_min,
            y_max, .005))

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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
