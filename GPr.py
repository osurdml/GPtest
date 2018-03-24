import numpy as np

# Define squared distance calculation function
def squared_distance(A,B):
    A = np.reshape(A,(len(A),1))
    B = np.reshape(B,(len(B),1))
    A2_sum = A*A
    B2_sum = B*B
    AB = 2*np.dot(A,B.T)
    A2 = np.tile(A2_sum,(1,np.size(B2_sum,axis=0)))
    B2 = np.tile(B2_sum.T,(np.size(A2_sum,axis=0),1))
    sqDist = A2 + B2 - AB
    return sqDist

# Define GP class
class GaussianProcess(object):
    def __init__(self,log_hyp,mean_hyp,like_hyp,covFunName,meanFunName,likeFunName, \
        trainInput,trainTarget):
        self.log_hyp = log_hyp
        self.mean_hyp = mean_hyp
        self.like_hyp = like_hyp
        self.covFunName = covFunName
        self.meanFunName = meanFunName
        self.likeFunName = likeFunName
        self.trainInput = trainInput
        self.trainTarget = trainTarget
        
        # Can you pass class handles instead of doing it the dumb way below?
        if covFunName == "SE":
            self.covFun = SquaredExponential(log_hyp,trainInput)
        else:
            self.covFun = []
        
        if meanFunName == "zero":
            self.meanFun = MeanFunction(mean_hyp)
        else:
            self.meanFun = []
        
        if likeFunName == "zero":
            self.likeFun = LikelihoodFunction(like_hyp)
        else:
            self.likeFun = []
        
    # Define GP prediction function
    def compute_prediction(self,testInput):
        Kxz = self.covFun.compute_Kxz_matrix(testInput)
        Kxx = self.covFun.compute_Kxx_matrix()
        iKxx = np.linalg.inv(Kxx)
        Kzx = Kxz.T
        K_diag = np.dot(np.dot(Kzx,iKxx),Kxz)
        K_noise = self.covFun.sn2*np.eye(np.size(testInput,axis=0))
        fz = np.dot(np.dot(Kzx,iKxx),self.trainTarget.T)
        cov_fz = K_noise + K_diag
        return fz, cov_fz
	
    # Define GP negative log marginal likelihood function
    def compute_likelihood(self,hyp):
        n = np.size(self.trainInput,axis=0)
        covSE = SquaredExponential(hyp,self.trainInput)
        Kxx = covSE.compute_Kxx_matrix()
        m = self.meanFun.y
        L = np.linalg.cholesky(Kxx)
        iKxx = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)))
        y = np.reshape(self.trainTarget,(len(self.trainTarget),1))
        err_y = np.dot(np.dot((y-m).T,iKxx),(y-m))/2
        det_Kxx = np.sum(np.log(np.diag(L)))
        occams_razor = n*np.log(2*np.pi)/2
        nlml = err_y + det_Kxx + occams_razor
        return nlml

# Define GP mean function class
class MeanFunction(object):
    def __init__(self,x):
        self.x = x
        self.y = np.zeros_like(x)

# Define GP likelihood function class
class LikelihoodFunction(object):
    def __init__(self,x):
        self.x = x
        self.y = np.exp(2*x)	

# Define covariance function class
class CovarianceFunction(object):
    def __init__(self,logHyp,x):
        self.logHyp = logHyp         	# log hyperparameters
        self.x = x                        # training inputs

# Define squared exponential CovarianceFunction function
class SquaredExponential(CovarianceFunction):
    def __init__(self,logHyp,x):
        CovarianceFunction.__init__(self,logHyp,x)
        self.hyp = np.exp(self.logHyp)	# hyperparameters
        n = len(self.hyp)
        self.M = self.hyp[:n-2]        	# length scales
        self.sf2 = self.hyp[n-2]**2        # sigma_f variance
        self.sn2 = self.hyp[n-1]**2        # noise variance
	
    def compute_Kxx_matrix(self):
        scaledX = self.x/self.M
        sqDist = squared_distance(scaledX,scaledX)
        Kxx = self.sn2*np.eye(np.size(self.x,axis=0))+self.sf2*np.exp(-0.5*sqDist)
        return Kxx
	
    def compute_Kxz_matrix(self,z):
        scaledX = self.x/self.M
        scaledZ = z/self.M
        sqDist = squared_distance(scaledX,scaledZ)
        Kxz = self.sf2*np.exp(-0.5*sqDist)
        return Kxz