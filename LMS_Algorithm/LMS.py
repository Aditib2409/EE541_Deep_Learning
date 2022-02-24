import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import lfilter
f = h5py.File('lms_fun_v3.hdf5', 'r') ## Group - dictionary

"""
Defining the LMS parameters as follows - 
    L: no. of filter taps
    Learning_rate: for updating the weights
    N: Total no. of Input Sequences
    n: no. of data samples in each input sequence    
"""
L = 3 # numbe of filter taps in Weiner filter
N = 600 # sequence length
n = 501 # number of samples in each sequence
learning_rate = [0.05, 0.1, 0.12,0.15, 0.175,0.2]
#learning_rate = [0.05, 0.15]

# --------------- CREATING DATASETS -------------------- #

## for snr = 3dB
SNR = [3,10]
matched_v = np.zeros([len(SNR),N,n,L]) # regressor
matched_x = np.zeros([len(SNR),N,n]) # inputs
matched_y = np.zeros([len(SNR),N,n]) # ground truth outputs
matched_z = np.zeros([len(SNR),N,n]) # noisy target

matched_v[0,:,:,:] =  f['matched_3_v']
matched_v[1,:,:,:] =  f['matched_10_v']

matched_y[0,:,:] = f['matched_3_y']
matched_y[1,:,:] = f['matched_10_y']

matched_x[0,:,:] = f['matched_3_x']
matched_x[1,:,:] = f['matched_10_x']

matched_z[0,:,:] = f['matched_3_z']
matched_z[1,:,:] = f['matched_10_z']

"""
    Creating an LMS class as follows -
    The adapt_filter function calculates three terms - 
        1. filter output 'y'.
        2. Weights - (600, 501, 3) sized 3D array having the coefficients.
        3. w - (3,) sized 1D array calculating coefficients for each data sample in each input sequence.
        4. e - error between the original labeled output and the predicted output from the filter.
        
    The function 'adapt_filter' returns "y, e, Weights" as outputs.
    
    Additionally I have considered an error tolerance - 'epsilon' to avoid division by zero while normalization.

"""

class LMS():
    def __init__(self, L=L, learning_rate=learning_rate):
        self.L = L # no. of filter taps
        self.learning_rate = learning_rate # Eeta
        self.w = np.zeros(self.L) # initialiing the weight vector
        self.epsilon = 1e-4 # error tolerance
    
    """
    BUILDING THE WEINER FILTER 
    """
    
    def adapt_filter(self, v, x, zn, yn, N, n, normalize):
        Weights = np.zeros([N,n,3]) # N = 600, n = 501
        Y = np.zeros([N,n])
        E = np.zeros([N,n])
        for i in range(N): # for each input sequence
            self.w = np.zeros(self.L)
            for j in range(n): # for each data sample in one input sequence
                prod = np.inner(self.w, v[i,j,:]) # prediction 
                if normalize:
                    self.w += self.learning_rate*(zn[i,j]-prod)*v[i,j]/(np.inner(v[i,j],v[i,j]) + self.epsilon)
                else:
                    self.w += self.learning_rate*(zn[i,j]-prod)*v[i,j]  # each data point
                Weights[i,j,:] = self.w
                #print(Weights)
                
                y = np.inner(self.w, v[i,j]) # filter output with noise w^{T}x # scalar
                
                e = (yn[i,j] - y)**2 # error in the prediction # scalar
            
                Y[i,j] = y
                E[i,j] = e
        
        return E, Y, Weights    
    
# creating an LMS class instance - 
def Compute_LMS(v, x, z, y, snr):
    
    #fig1,axs = plt.subplots(2,2, figsize = (10,8),sharex=True, sharey=False)
    fig2,axes = plt.subplots(1,6, figsize = (21,4), sharex = True, sharey = False)
    for i in range(len(learning_rate)):
        mse = np.zeros([n])
        lr = learning_rate[i]
        lms = LMS(L=3, learning_rate=lr)


        square_error, predicted_output, weights = lms.adapt_filter(v, x, z, y, N=N, n=n, normalize=False)
        #print(square_error)
        for m in range(n): # n= 501 (600,501)
            mse[m] = np.mean(square_error[:,m], axis=0) # (501,)
        
        ## coefficient averages
        weights_average = np.zeros([n,3])
        for k in range(n):
            for j in range(L):
                weights_average[k,j] = np.sum(weights[:,k,j])/N

        original_weights = [1, 0.5, 0.25]
        labels = ["h0", "h1", "h2"]
        colors = ["c","orange","y","hotpink","m","#4CAF50"]
        #fig1.suptitle(f'Coefficients for 1st input sequence (top) & Coefficient averages over all sequences (bottom)')
    
        for j in range(L):
            axs[0,i].axhline(y=original_weights[j], color='black', linestyle='--')
            axs[0,i].plot(weights[0,:,j], color=colors[j], label=labels[j])
            axs[0,i].legend(loc="lower right")
            axs[0,i].set_xlabel('Updates')
            axs[0,i].set_ylabel('Coefficients')
            axs[0,i].set_title(f'$\eta$={lr}, SNR={snr}')
            axs[1,i].plot(original_weights[j], color=colors[j])
            axs[1,i].plot(weights_average[:,j], color=colors[j], label=labels[j])
            axs[1,i].legend(loc="lower right")
            axs[1,i].set_xlabel('Updates')
            axs[1,i].set_ylabel('Coefficients')
            axs[1,i].set_title(f'$\eta$={lr}, SNR={snr}')
        
        axes[i].plot(mse[:], color=colors[i])
        axes[i].set_title(f'$\eta$={lr}, SNR={snr}')
        axes[i].set_xlabel('Updates')
        #axes[i].set_ylabel('MSE')
        fig2.suptitle(f'Divergence of the MSE')
        #plt.savefig('Result_figures/Problem1(a-ii)_'+str(SNR[i])+'dB.png')
        
print("LMS successfully adapted!")
print("Weights, error and Output signal successfully predicted!")            
    
"""
    CAOMPUTATIONS AND PLOTS FOR THE 'MATCHED' FILTER
"""

for i in range(len(SNR)):
    Compute_LMS(matched_v[i,:,:,:], matched_x[i,:,:], matched_z[i,:,:], matched_y[i,:,:], snr=SNR[i])
    #plt.savefig('Result_figures/Problem1(a-iii)_'+str(SNR[i])+'dB.png')
