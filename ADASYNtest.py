from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as pl
import ADASYN


# Function that creates 2 clusters that partially overlap
# @return: X The datapoints e.g.: [f1, f2, ... ,fn]
# @return: y the classlabels e.g: [0,1,1,1,0,...,Cn]
def createCluster():
    # TODO: For the example to work properly with the new changes for multi-class data support
    #       this method should change so it creates multi-class data. I'm not getting into that
    #       since I don't need it. Simplest way: just use your own multi-class data.
    X1, y1 = make_blobs(n_samples=50, centers=1, n_features=2,random_state=0,center_box = (-5.0,5.0))
    X2, y2 = make_blobs(n_samples=200, centers=1, n_features=2,random_state=0,center_box = (-4.0,6.0))
    
    X = np.concatenate((X1,X2),axis=0)
    y = np.concatenate((y1,[1]*len(y2)),axis=0)
    
    return X.tolist(),y.tolist()


def main():

    # Create 2 artificial clusters that partially overlap
    # TODO: For the example to work properly with the new changes for multi-class data support
    #       this method should change so it creates multi-class data. I'm not getting into that
    #       since I don't need it. Simplest way: just use your own multi-class data.
    X,y = createCluster()
    # You probably need to modify the createCluster() so that you can pass the labels
    # or the number of them in order to produce data according to your needs.
    labels = [0, 1, 2, 3, 4]
    
    
    # Plot the clusters
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    pl.scatter(np.array(X)[:, 0], np.array(X)[:, 1], color=colors[y].tolist(), s=10)
    pl.show()
    
    a = ADASYN.Adasyn(X, y, labels)
    X, y = a.balance_all()
    # # Get the minority and majority count
    # ms,ml = ADASYN.getClassCount(X,y)
    # d = ADASYN.getd(X,y,ms,ml)
    # G = ADASYN.getG(X,y,ms,ml,1)

    # # Get the list of r values, which indicate how many samples will be made per data point in the minority dataset
    # rlist = ADASYN.getRis(X,y,0,5)

    # # Generate the synthetic data
    # newX,newy = ADASYN.generateSamples(rlist,X,y,G,0,5)
    
    # # Plot the dataset again
    # pl.scatter(np.array(X)[:, 0], np.array(X)[:, 1], color=colors[y].tolist(), s=10)
    # pl.scatter(np.array(newX)[:, 0], np.array(newX)[:, 1], color='red', s=10)
    # pl.show()
    
    # X,y = ADASYN.joinwithmajorityClass(X,y,newX,newy,1)

    print 'test'

if  __name__ =='__main__':main()
