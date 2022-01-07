import tensorflow as tf
import numpy as np 
from astropy.io import fits
import os

#----------------
# Architectures
#----------------

# for this application cpu is enough, if your batch of images is huge and want to run the script using GPUs, comment this line 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 


class BaseModel(tf.keras.Model):

    def __init__(self,dropout=0.1):
        super(BaseModel, self).__init__()
        self.C1 = tf.keras.layers.Conv2D(16, 3,activation='relu',input_shape=(96, 96, 1))
        self.C2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.C3 = tf.keras.layers.Conv2D(32, 3,activation='relu')
        self.C4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        

        self.C5 = tf.keras.layers.Conv2D(64, 3,activation='relu')
        self.C6 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))


        self.C7 = tf.keras.layers.GlobalAveragePooling2D()
        self.C8 = tf.keras.layers.Dropout(dropout)
        
        self.C9 = tf.keras.layers.Dense(200,activation='relu')
        self.C10 = tf.keras.layers.Dropout(dropout)
        self.C11 = tf.keras.layers.Dense(100,activation='relu')
        self.C12 = tf.keras.layers.Dense(20,activation='relu')
        
        self.Cout = tf.keras.layers.Dense(1,activation='linear')
    
    def call(self,inputs):
        # in this method we define the forward passing of the CNN
        x = self.C1(inputs)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        x = self.C7(x)
        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)
        x = self.C11(x)
        x = self.C12(x)
        x = self.Cout(x)
        return x
    
    
    
    def PredictMass(self,X,z,weights_path = "./weights/"):
        '''
        return an array containing the CNN masses given the image X and the redshift z as an input.
        
        X: shape has to be  (batch,96,96,1), for an image with no color (channels=1)
        '''
        h=0.678 # the hubble parameter
        z = np.array(z) # z has to be a list
        X = X.reshape((len(z),96,96,1))
        preds = np.zeros(len(z))
        
        redshift_ranges = (0,0.1,0.2,0.4,1)
        mu = [1.8827347385422218e-06,2.2296471704066543e-07,8.347489023971544e-08,3.14216726766519e-08]
        sigma = [7.136290778515806e-06,2.036420289662876e-06,1.4639878011606421e-06,1.300205829438648e-06] 
        
        
        for i in range(len(redshift_ranges)-1):
            mask = (z>redshift_ranges[i])&(z<=redshift_ranges[i+1])
            if len(z[mask])>0:
                img = X[mask]
                img = (img-mu[i])/sigma[i]
                self.build(img.shape)
                self.load_weights(weights_path+ 'bestmodel-%i.h5'%(i+1))
                preds[mask] = self.predict(img)[:,0]
                
                
        preds = preds-np.log10(h)

        return preds
    
    def ReadFits(self,img_file="PSZ2G359.07-32.12.fits"):
        hdu = fits.open(img_file)
        header = hdu[0].header
        X = hdu[0].data
        z = header['Z']
        M500 = header['M500']*1e14
        return X,z,M500
        