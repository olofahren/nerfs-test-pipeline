import numpy as np
import random
import HDRutils


class CameraSim:
  def __init__(self, type = 0):

    #init
    print('Initializing camera simulator...\n')

    self.n = 0.9
    self.sigma = 0.6
    self.gamma = 2.0
    
    # Subset of K-means clustered    (the cluster closest to the overall mean CRF)
    subs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 66, 67,
            72, 73, 76, 77, 85, 86, 88, 142, 143, 144, 145, 146, 147, 149, 150, 160, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 182, 184, 185, 186, 187, 190, 191, 192, 194, 195, 198, 199, 200]
    
    crf_pth = 'camera_simulation/dorfCurves.bin'
    data = np.fromfile(crf_pth, 'float32')

    ss = data[0:2].astype('int32')
    I = data[2:ss[0]*ss[1]+2]
    B = data[ss[0]*ss[1]+2:2*ss[0]*ss[1]+2]

    I = np.reshape(I, [ss[0],ss[1]]);
    B = np.reshape(B, [ss[0],ss[1]]);

    sigma, n = 0.6, 0.9
    ma, mi = 0, 1e10
    for i in range(ss[0]):
        ms = np.sum(B[i,:]-I[i,:])
        if ms > ma:
            ma = ms
            ma_ind = i

        xp = np.power(I[i,:], n)
        y = (1+sigma) * xp / (xp + sigma)
        ms = np.sum(np.abs(B[i,:]-y))
        if ms < mi and ms > 0:
            mi = ms
            mi_ind = i
    self.crf1 = (I[mi_ind,:],B[mi_ind,:])
    self.crf2 = (I[ma_ind,:],B[ma_ind,:])
    
    self.I = I[subs,:]
    self.B = B[subs,:]
    self.N = self.I.shape[0]
    
    self.model = HDRutils.NormalNoise()

  def capture(self, img, meth, crf=0, noise=False, exp=1.0/30.0, iso=800):
    x = img
    
    if noise:
      self.model.set_profile(make_str='Canon', model_str='EOS-1Ds', iso=iso)
      x = self.model.simulate(x, exp=exp, iso=iso)
    
    x = np.maximum(np.minimum(x,1.0),0.0)
    
    x, crf = getattr(self, 'crf_%s'%meth)(x, crf)

    return x, crf

  def crf_sigmoid(self, x):
    xp = np.power(x, self.n)
    y = (1+self.sigma) * xp / (xp + self.sigma)
    return y

  def crf_gn1(self, x):
    y = np.interp(x, self.crf1[0], self.crf1[1])
    return y

  def crf_gn2(self, x):
    y = np.interp(x, self.crf2[0], self.crf2[1])
    return y

  def crf_rand(self, x, sind=0):
    #print('CRF = ', sind)
    
    y = np.interp(x, self.I[sind,:], self.B[sind,:])
    
    crf = np.zeros((2,self.I.shape[1]))
    crf[0,:] = self.I[sind,:]
    crf[1,:] = self.B[sind,:]
    
    return y, crf
