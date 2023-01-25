class GUClustering():
  def __init__(self):
    self.Y = None 
    self.X = None 
    
  def _cluster(self, Y, quantiles):
    quantiles = quantiles.reshape(-1,1)
    return np.argmin((abs(quantiles - Y)).T,axis=1) + 1 # starting from 1

  def fit(self,X):
    self.X = X
    self.Y = ((self.X[:,0] - np.mean(self.X[:,0])) ** 2) 
    for j in range(1,len(X[0])):
      self.Y += ((self.X[:,j] - np.mean(self.X[:,j])) ** 2) 
    self.Y = np.power(self.Y,0.5)
    self.Y = self.Y / self.Y.max() # convert between [0,1] (optional)

  def global_clustering(self,K=4):
    quantiles = np.quantile(self.Y,[k/K for k in range(K)])
    return self._cluster(self.Y,quantiles)
    
  def outlier_detection(self, q=0.95):
    threshold = np.quantile(self.Y,q)
    return ((self.Y > threshold)).astype(int)
