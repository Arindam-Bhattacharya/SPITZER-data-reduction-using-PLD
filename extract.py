import numpy
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pyfits
import scipy
import random
import os.path
from scipy import signal

# The average function is used for determining the Poissonian nature of residual noise.

def average(array,answer):
    newarray=[]
    answer.append(array)
    if len(array)>1:
        for i in range(0,len(array)/2):
            newarray.append((array[2*i]+array[2*i+1])/2)
        average(newarray,answer)

# The polynomial fit is used to fit a 17th degree polynomial to the data to account for intrinsic stellar variability

def polynomial_fit(p,j,c,d,e,f,g,h,i,k,l,m,n,o,q,r,s,t,u):
    return j+c*p+d*p**(2.0)+e*p**(3.0)+f*p**(4.0)+g*p**(5.0)+h*p**(6.0)+i*p**(7.0)+k*p**(8.0)+l*p**(9.0)+m*p**(10.0)+n*p**(11.0)+o*p**(12.0)+q*p**(13.0)+r*p**(14.0)+s*p**(15.0)+t*p**(16.0)+u*p**(17.0)

# The transit function is used to generate fake transits for the data. The function currently doesn't account for limb darkening. The input parameters are as below:
# k = Radius of planet/Radius of Star
# l = Radius of star/Semi-major axis of planet's orbit
# P = Orbital period of planet in days
# array1 = Flux time series
# array2 = Time series of observations
# m = Fraction which indicates where the midtransit should be at

def transit(k,l,P,array1,array2,m):
    tfull=P*math.asin((1-abs(k))*l)/math.pi
    ttot=P*math.asin((1+abs(k))*l)/math.pi
    v=ttot/(array2[len(array2)-1]-array2[0])
    o=tfull/(array2[len(array2)-1]-array2[0])
    a=v*len(array2)
    h=o*len(array2)
    s=int(m*len(array2))
    c=numpy.zeros(len(array2))
    e=numpy.average(array1)
    if k>0:
        for i in range(0,s-int(a/2)):
            c[i]=e
        for i in range(s-int(a/2),s-int(h/2)):
            c[i]=e-k**2*e*(i-(s-int(a/2)))/(int(a/2)-int(h/2))
        for i in range(s-int(h/2),s+int(h/2)):
            c[i]=(1-k**2)*e
        for i in range(s+int(h/2),s+int(a/2)):
            c[i]=(1-k**2)*e+k**2*e*(i-(s+int(h/2)))/(int(a/2)-int(h/2))
        for i in range(s+int(a/2),len(array2)):
            c[i]=e
    else:
        for i in range(0,s-int(a/2)):
            c[i]=e
        for i in range(s-int(a/2),s-int(h/2)):
            c[i]=e+k**2*e*(i-(s-int(a/2)))/(int(a/2)-int(h/2))
        for i in range(s-int(h/2),s+int(h/2)):
            c[i]=(1+k**2)*e
        for i in range(s+int(h/2),s+int(a/2)):
            c[i]=(1+k**2)*e-k**2*e*(i-(s+int(h/2)))/(int(a/2)-int(h/2))
        for i in range(s+int(a/2),len(array2)):
            c[i]=e
    return c

# The chi-squared function is the standard chi-squared parameter that determines how well the data fits the model. Here data and model are the arrays that represent the lightcurve and the model transit curve respectively.

def chisquared(data,model):
    sigma=numpy.std(data)
    val=numpy.sum((data-model)**(2.0))/sigma**(2.0)
    return val

# The PLD function is the one that models the instrumental response in warm Spitzer data. The inputs to the function are:
# array = Raw flux
# matrix = Matrix that stores individual pixel flux. For this particular code we use 9 pixels. 

def PLD(array,matrix):
    l=numpy.transpose(array)
    normp=numpy.zeros((len(array),9))
    u=matrix.sum(axis=1)
    for i in range(len(array)):
        for j in range(9):
            normp[i,j]=matrix[i,j]/u[i]
    c=numpy.dot(numpy.linalg.pinv(normp,rcond=1e-25),l)
    n=numpy.dot(normp,c)
    PLD=numpy.transpose(n)
    return PLD

# head is the header file for the object. Insert the appropriate file location. 
head=pyfits.getheader('C:/Users/Arindam/Desktop/UROP/J213926+022023/r46321152/ch2/bcd/SPITZER_I2_46321152_0001_0000_2_bcd.fits')
name=head['OBJECT']

# f stores the filelist of the object's observations. Insert the appropriate .txt file that stores the names of the files.
f = open(os.path.normpath('C:/Users/Arindam/Desktop/UROP/J213926+022023/r46321152/ch2/bcd/filename.txt'))
fr = []
# stej is the location where the observation data is stored
stej = "C:/Users/Arindam/Desktop/UROP/J213926+022023/r46321152/ch2/bcd/"

# The following loop selects for BCD(Basic Calibrated Data) files in the filelist.
for line in f:
    if "_bcd.fits" in line:fr.append(stej+line[:-1])

flux=numpy.zeros(len(fr)) # flux is the raw flux time series
time=numpy.zeros(len(fr)) # time is the array that records the time of observation for each element in flux
p=numpy.zeros((len(flux),9)) # p is the matrix that stores individual pixel flux data

for i in range(len(fr)):
    newimage=pyfits.getdata(fr[i]) # Obtain the image
    header=pyfits.getheader(fr[i])
    time[i]=header['MJD_OBS']
    image=numpy.transpose(newimage) # Traspose the image in order to match the coordinates in a fits liberator
    value = 0
    for k in range(3):
          for j in range(3):
              value = value+image[22+j,230+k] # Obtain the raw total flux from 9 pixels
    flux[i]= value
    p[i,0]=image[22,230] #Store the pixel flux in the p matrix. Insert the coordinates as desired, preferably using a 3x3 grid centred on the object of interest
    p[i,1]=image[22,231]
    p[i,2]=image[22,232]
    p[i,3]=image[23,230]
    p[i,4]=image[23,231]
    p[i,5]=image[23,232]
    p[i,6]=image[24,230]
    p[i,7]=image[24,231]
    p[i,8]=image[24,232]
    
# Implement a moving median filter of desired kernel size in order to obtain outliers. The tolerance limit for deviation is 4*standard deviation
    
filtered_flux=scipy.signal.medfilt(flux,kernel_size=11) 
noise=flux-filtered_flux
stdev=numpy.std(noise)

mask=numpy.zeros(len(flux))
a=[]

for  i in range(len(flux)):
    if abs(noise[i])<(4*stdev):
        mask[i]=0
    else:
        mask[i]=1
        a.append(i)

# Returns the flux time series with the outliers removed
newflux=flux[numpy.where(mask==0)] 
newtime=time[numpy.where(mask==0)]              
p=numpy.delete(p,a,axis=0)

correction=PLD(newflux,p) # Apply PLD to obtain the instrumental response.  

plt.scatter(newtime,newflux)   # Produce a plot showing the raw flux versus time with the correction overplotted on the same.
plt.plot(newtime,correction,'ro')
plt.xlabel('Time',fontsize=26)
plt.ylabel('Uncorrected Flux',fontsize=26)
plt.title(name,fontsize=26)
plt.show()

fluxcorrected=newflux-correction # Subtract the PLD correction 

#Set a scale in order to obtain a polynomial fit for the intrinsic stellar variation in the reduced data 
scale=newtime[len(newtime)-1]-newtime[0]
scaledtime=(newtime-newtime[0])/scale
osc=opt.curve_fit(polynomial_fit,scaledtime,fluxcorrected)[0]
bn=polynomial_fit(scaledtime,osc[0],osc[1],osc[2],osc[3],osc[4],osc[5],osc[6],osc[7],osc[8],osc[9],osc[10],osc[11],osc[12],osc[13],osc[14],osc[15],osc[16],osc[17])

#The following commented region can be decommmented if one observes that the PLD correction process trades off between the stellar variation and the instrumental variation.
#It just iterates on a first order guess and reimplements PLD. In order to better model the instrumental response, it is recommended that one iterate it quite some times.
 
##normastro=(bn+numpy.average(newflux))/numpy.average(bn+newflux)
##it1=newflux/normastro
##pd=numpy.zeros((len(newflux),9))
##for i in range(9):
##    pd[:,i]=p[:,i]/normastro
##itcorr1=instrument(it1,pd)
##fluxcorrected=newflux-itcorr1
##osc=opt.curve_fit(polynomial_fit,scaledtime,fluxcorrected)[0]
##bn=polynomial_fit(scaledtime,osc[0],osc[1],osc[2],osc[3],osc[4],osc[5],osc[6],osc[7],osc[8],osc[9],osc[10],osc[11],osc[12],osc[13],osc[14],osc[15],osc[16],osc[17])


expectedtransit=fluxcorrected-bn # Obtain the residuals after having subtracted the stellar variation

# Plot the stellar variation along with the polynomial fit
plt.xlabel('Time',fontsize=26)
plt.ylabel('Residuals',fontsize=26)
plt.scatter(newtime,fluxcorrected) 
plt.scatter(newtime,bn)
plt.title(name,fontsize=26)
plt.show()

# Plot the remaining residuals versus time
plt.xlabel('Time',fontsize=26)
plt.ylabel('Remaining Residuals',fontsize=26)
plt.title(name,fontsize=26)
plt.scatter(newtime,expectedtransit)
plt.show()

# This part of the code determines the Poissonian nature of the noise by inspecting if the standard deviation goes as square root bin-size. 
s=math.floor(math.log(len(expectedtransit),2))
g=2**s
noisefilter=expectedtransit[:g]
m = []
average(noisefilter,m)
sg=numpy.zeros(len(m))
pred=numpy.zeros(len(m))
for i in range(len(m)):
    sg[i]=numpy.std(m[i])
for i in range(len(m)):
    pred[i]=sg[0]/2**(i/2.0)    
plt.plot(range(len(m)),pred)
plt.scatter(range(len(m)),sg)
plt.xlabel('Bin Size',fontsize=24)
plt.ylabel('Standard Dev',fontsize=24)
plt.title(name,fontsize=24)
plt.show()

# Normalise the remaining residuals
intensity=(expectedtransit+numpy.average(newflux))/numpy.average(newflux+expectedtransit)

# This part of the code does Monte-Carlo analysis on the normalised lightcurve by blindly injecting transits and recovering them
orbitalperiod=1
rad=numpy.arange(0.1,0.12,0.001) #Setup the range of radii which have to be injected

prob=numpy.zeros(len(rad)) #Setup an array that records the probability of recovery of the transit
sigr=numpy.zeros(len(rad)) #Setup an array recording the standard deviation of the radii recovered for 100 trials. We blindly inject 100 transits for each radius in rad
obsr=numpy.zeros(len(rad)) #Setup an array recording the average rdaius for 100 trials

#This part generates model lightcurves which will be used for implementing a chi-squared search on the fake lightcurves
light=numpy.zeros((170,1000))
light=light.astype(numpy.object)
radius=numpy.arange(-0.05,0.12,0.001) #Setup the radius ratio range that you will work on. Throughout this code,we assume a primary that is the size of Jupiter and fix the orbital period at 1 day.
u=(orbitalperiod*numpy.arcsin((1+abs(radius))*0.0565)/numpy.pi)/(newtime[-1]-newtime[0])
for i in range(170):
    for j in range(1000):
        light[i][j]=transit(radius[i],0.0565,orbitalperiod,intensity,newtime,u[i]/(2.0)+j*(1-u[i])/1000)
chi=numpy.zeros((100,170,1000))
observedphase=numpy.zeros(100)
observedradius=numpy.zeros(100)
injectedphase=numpy.zeros(100)
detection=numpy.zeros(100)

#This loop runs 100 trials for each radius in the rad array. We first generate a random phase and create a fake lightcurve based on that. Thereafter
#we implement a chi-squared search by generating chi-squared values for each of the models that we created in the previous section. We then recover
#the radius and phase of mid-transit for the model-lightcurve that has the lowest chi-squared value. A 'detection' occurs only if the calculated phase is within
#half the  transit time of the injected phase AND if the calculated radius ratio is within 0.005 of the injected radius ratio. Thereafter we obtain the total
#number of detections in the 100 trials and record the probability of detection,the average radius and the standard deviation for each 100 trials.
for i in range(len(rad)):
    fake=[]
    radiusratio=rad[i]
    transittime=orbitalperiod*math.asin((1+radiusratio)*0.0565)/math.pi
    scaledtransittime=transittime/(newtime[-1]-newtime[0])
    for p in range(100):
        phi=random.uniform(scaledtransittime/(2.0),1-scaledtransittime/(2.0))
        injectedphase[p]=phi
        trans=transit(radiusratio,0.0565,orbitalperiod,intensity,newtime,phi)
        lightcurve=trans*intensity/(numpy.average(intensity))**(2.0)
        fake.append(lightcurve)
    for k in range(100):
        for e in range(170):
            for j in range(1000):
                chi[k,e,j]=chisquared(fake[k],light[e][j])
        l,m=numpy.unravel_index(chi[k,:,:].argmin(),chi[k,:,:].shape)
        r=(-0.05)+0.001*l
        o=(orbitalperiod*math.asin((1+abs(r))*0.0565)/math.pi)/(newtime[-1]-newtime[0])
        observedradius[k]=r
        observedphase[k]=o/(2.0)+(1-o)*m/1000
        if (abs(observedphase[k]-injectedphase[k])<=(0.5*scaledtransittime)) and (abs(observedradius[k]-radiusratio)<=0.005):
            detection[k]=1
        else: detection[k]=0
    prob[i]=numpy.sum(detection)/100.0
    obs=observedradius[numpy.where(detection==1)]
    obsr[i]=numpy.average(obs)
    sigr[i]=numpy.std(obs)

# Plot a Probability of detection vs Radius plot
plt.scatter(rad,prob)
plt.ylabel('Probability of detection',fontsize=40)
plt.xlabel('Radius Ratio',fontsize=40)
plt.title('Probability for '+name,fontsize=40)
plt.tick_params(labelsize=34)
plt.grid()
plt.show()

# Plot an Observed radius vs Injected radius plot
plt.errorbar(rad,obsr,xerr=0,yerr=err,ls='none')
plt.scatter(rad,rad)
plt.ylabel('Radius ratio calculated',fontsize=40)
plt.xlabel('Radius Ratio',fontsize=40)
plt.title('Rad vs Rad for '+name,fontsize=40)
plt.tick_params(labelsize=34)
plt.grid()
plt.show()
