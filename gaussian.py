import torch as tr


class Gaussian(object):
	def __init__(self, D,  log_sigma, dtype = tr.float32, device = 'cpu'):
		self.D =D
		self.params  = log_sigma
		self.dtype = dtype
		self.device = device
		self.adaptive=  False
		self.params_0 = log_sigma  

	def get_exp_params(self):
		return pow_10(self.params, dtype= self.dtype, device = self.device)
	def update_params(self,log_sigma):
		self.params = log_sigma


	def square_dist(self, X, Y):
		# Squared distance matrix of pariwise elements in X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._square_dist( X, Y)

	def kernel(self, X,Y):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._kernel(self.params,X, Y)

	def dkdxdy(self,X,Y,mask=None):
		return self._dkdxdy(self.params,X,Y,mask=mask)
# Private functions 

	def _square_dist(self,X, Y):
		n_x,d = X.shape
		n_y,d = Y.shape
		dist = -2*tr.einsum('mr,nr->mn',X,Y) + tr.sum(X**2,1).unsqueeze(-1).repeat(1,n_y) +  tr.sum(Y**2,1).unsqueeze(0).repeat(n_x,1) #  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 

		return dist 

	def _kernel(self,log_sigma,X,Y):
		N,d = X.shape
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		tmp = self._square_dist( X, Y)
		dist = tr.max(tmp,tr.zeros_like(tmp))
		if self.adaptive:
			ss = tr.mean(dist).clone().detach()
			dist = dist/(ss+1e-5)
		return  tr.exp(-0.5*dist/sigma)


	def _dkdxdy(self,log_sigma,X,Y,mask=None):
		# X : [M,T]
		# Y : [N,R]

		# dkdxdy ,   dkdxdy2  = [M,N,T,R]  
		# dkdx = [M,N,T]
		N,d = X.shape
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		gram = self._kernel(log_sigma,X, Y)

		D = (X.unsqueeze(1) - Y.unsqueeze(0))/sigma
		 
		I  = tr.ones( D.shape[-1],dtype=self.dtype, device = self.device)/sigma

		dkdy = tr.einsum('mn,mnr->mnr', gram,D)
		dkdx = -dkdy



		if mask is None:
			D2 = tr.einsum('mnt,mnr->mntr', D, D)
			I  = tr.eye( D.shape[-1],dtype=self.dtype, device = self.device)/sigma
			dkdxdy = I - D2
			dkdxdy = tr.einsum('mn, mntr->mntr', gram, dkdxdy)
		else:
			D_masked = tr.einsum('mnt,mt->mn', D, mask)
			D2 = tr.einsum('mn,mnr->mnr', D_masked, D)

			dkdxdy =  tr.einsum('mn,mr->mnr', gram, mask)/sigma  -tr.einsum('mn, mnr->mnr', gram, D2)
			dkdx = tr.einsum('mnt,mt->mn',dkdx,mask)

		return dkdxdy, dkdx, gram




def pow_10(x, dtype=tr.float32,device = 'cpu'): 

	return tr.pow(tr.tensor(10., dtype=dtype, device = device),x)

























