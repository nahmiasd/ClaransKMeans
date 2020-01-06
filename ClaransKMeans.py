import numpy as np
from sklearn.cluster import KMeans
import random
from sklearn import datasets


class ClaransKMeans:
    def __init__(self,k,nr):
        #constructor, k is number of clusters, nr is number of iteration for clarans algorithm
        self.k=k
        self.NR=nr
        self.km=None

    def calcEnergy(self,centers,X):
        # calculates the energy of the model given the train set and the cluster centers indecies.
        en=0
        for x in X:
            en+=min([np.linalg.norm(x-X[c])**2 for c in centers]) #this is the calculation described in the paper
        return en

    def clarans(self,X):
        # the clarans initialization algorithm as described in the paper
        if len(X)<self.k:
            raise ValueError('k is smaller than train set size')
        centersIndecies=set([x[0] for x in random.sample(list(enumerate(X)),self.k)]) #we sample cluster centers randomly
        en_=self.calcEnergy(centersIndecies,X) #current energy given our samples
        nr=0
        while(nr<=self.NR):
            i_=random.sample(centersIndecies,1)[0] #sample a center to replace
            ip=random.sample([i for i in range(len(X)) if i not in centersIndecies],1)[0] #sample a replacement from the train set
            newCenters=set([c for c in centersIndecies if c!=i_]+[ip])
            enp=self.calcEnergy(newCenters,X) #calculate energy after replacement
            if enp<en_: #if the energy is lower, update the chosen centers and restart iterations counting
                centersIndecies=newCenters
                en_=enp
                nr=0
            else:
                nr+=1
        return np.array([X[c] for c in centersIndecies]) #return chosen centers

    def fit(self,X): # fit the model
        X=[np.array(x) for x in X]
        centers=self.clarans(X) #initialize centers using clarans
        self.km=KMeans(n_clusters=self.k,init=centers)
        self.km.fit(X) # fit k-means

    def predict(self,y):
        if self.km==None:
            raise ValueError('classifier not fitted')
        return self.km.predict(y)

    def get_params(self):
        return {'k':self.k,'nr':self.nr}

    def set_params(self,**params):
        self.k=params[0]
        self.nr=params[1]


ckm=ClaransKMeans(5,10)
iris=datasets.load_iris()
X=iris.data
ckm.fit(X)

