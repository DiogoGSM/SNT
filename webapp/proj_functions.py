import numpy as np
import math
from scipy import interpolate
import scipy.constants as constant
from scipy.signal import find_peaks
from collections import deque


def b_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2 
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1

class smooth:
    @staticmethod
    def normalize (f,minl, maxl, minf, maxf, stretch):
        delta_lambda=maxl-minl
        delta_f=maxf-minf
        q=(delta_lambda/delta_f)
        fn= (f*q*(1/stretch))  
        return fn
    
    @staticmethod                           #sigma clipping using a rolling window of size w_size (asymmetric upper bound only)
    def rolling_sigma_clip (y, x, w_size):  #w_size number of elements in window, returns new list without outliers, uses median as cent function and sigma=1.5*iqr
        y_clipped=[]
        x_clipped=[]
        nwindows=len(x)//w_size  #number of windows
        rem=len(x)%w_size        #remaining elements
        w_start=0
        w_end=0
        for i in range(0,nwindows):
            w_end+=w_size
            w_elements=[]
            for j in range(w_start, w_end):
                w_elements.append((y[j], x[j]))
            q75, q25 = np.percentile([x[0] for x in w_elements], [75 ,25])
            iqr = q75 - q25
            upper= q75 + 1.5*iqr
            #lower= q25 - 1.5*iqr for symmetric clip
            for w in w_elements:                           
                if(not(w[0] > upper)): #or (w[0] < lower)): for a symmetric clip
                    y_clipped.append(w[0])
                    x_clipped.append(w[1]) 
            w_start=w_end
        if(rem!=0):         #calculate for remaining elements
            w_elements=[]                   
            for j in range(w_end, w_end+rem):
                w_elements.append((y[j], x[j]))
            q75, q25 = np.percentile([x[0] for x in w_elements], [75 ,25])
            iqr = q75 - q25
            upper= q75 + (1.5*iqr)
            #lower= q25 - 1.5*iqr
            for w in w_elements:                           
                if(not(w[0] > upper)): #or (w[0] < lower)): 
                 y_clipped.append(w[0])
                 x_clipped.append(w[1])          
        return y_clipped, x_clipped        
    
    @staticmethod
    def sigma_clip_iqr(distance, anchors_y, anchors_x):      #(asymetric lower bound only) sigma clipping for points that are too close, uses median as cent function and sigma=1.5*iqr 
        l=len(distance)                                                   
        q75, q25 = np.percentile(distance, [75 ,25])
        iqr = q75 - q25
        lower= q25 - 1.5*iqr 
        for i in range(1,l-1):         #for each pair keeps the one that maximises equidistance to neighbours
            if(distance[i] < lower): 
                if(distance[i-1]<distance[i+1]):
                    print("removed close points:",anchors_x[i])
                    anchors_x.pop(i)
                    anchors_y.pop(i)
                else:
                    print("removed close points", anchors_x[i])
                    anchors_x.pop(i+1)
                    anchors_y.pop(i+1)
        return                      

    @staticmethod
    def abs_rl_slope(a, b, c):     #calculates the sum of the absolute value of right and left derivative if they have different signs
        left=(b[1]-a[1])/(b[0]-a[0])
        right=(c[1]-b[1])/(c[0]-b[0])
        if(left*right<0):
            v=abs(left)+abs(right)
            return v
        else:
            return 0         #returns 0 if they have the same sign   
      
    @staticmethod    
    def remove_peaks(anchors_y, anchors_x, anchors_idx, ntimes):      #removes the sharpest peaks, alters the original lists, iterates ntimes
        derivatives=[]                                        #derivatives[0]: point , derivatives[1]: sum of abs(left) and abs(right) derivatives                         
        for i in range(1, len(anchors_y)-1):                 #calculate in groups of 3  until second last                                 
            deriv=smooth.abs_rl_slope((anchors_x[i-1], anchors_y[i-1]), (anchors_x[i], anchors_y[i]), (anchors_x[i+1], anchors_y[i+1]))
            if(deriv!=0):
                derivatives.append((anchors_x[i], anchors_y[i], deriv))
        dvalues = [dvalue[2] for dvalue in derivatives]           
        for j in range(0, ntimes):           
            percentile_value=np.percentile(dvalues,99.5)
            for k,d in enumerate (derivatives):
                if(d[2]>percentile_value):
                    dvalues.pop(k)
                    anchors_idx.pop(anchors_x.index(d[0]))
                    anchors_x.remove(d[0])
                    anchors_y.remove(d[1]) 
                    derivatives.remove(d)
        return  

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))

    @staticmethod
    def remove_close(anchors_y, anchors_x):
        l=len(anchors_x)
        dist=[]
        for i in range(0, l-1):
            d=smooth.distance(anchors_x[i], anchors_y[i], anchors_x[i+1], anchors_y[i+1])
            dist.append(d)

        smooth.sigma_clip_iqr(dist, anchors_y, anchors_x)
        return
            

    @staticmethod
    def denoise(anchors_y, anchors_idx, spectra, window_size):    #changes each maxima to the average in the window_size, changes list passed as function parameter
        l=len(spectra)
        for i, anchors_idx in enumerate(anchors_idx):
            window_elements=[]
            for j in range(0,window_size):
                if(anchors_idx-j)>0:
                    window_elements.append(spectra[anchors_idx-j])
                if(anchors_idx+j)<l:
                    window_elements.append(spectra[anchors_idx+j])
            anchors_y[i]=np.median(window_elements)               
        return
    
class a_shape:        #Defines methods for computing the alpha hull
    
    @staticmethod
    def angle (Cx,Cy, Px,Py,r):
        if((Cy-Py)>=0):
            return -math.acos((Cx-Px)/r)+math.pi
        else:
            return -math.asin((Cy-Py)/r)+math.pi
    
    @staticmethod
    def anchors(max_index, ys, xs, p_ys, p_xs,min_lambda , r_min, r_max, nu, use_pmap, global_stretch):    #Calculates the anchor points in the alpha hull             
        w_stretch=global_stretch                                                                          
        max_lambda = max(xs)
        min_flux=min(ys)
        max_flux=max(ys)
        furthest_point=math.sqrt(pow(max_lambda,2) + pow(max_flux,2))*global_stretch   #<--adjusted to reflect stretching
        
        P=np.array([xs[max_index[0]],smooth.normalize (ys[max_index[0]],min_lambda, max_lambda, min_flux, max_flux, w_stretch)])                                                      
        Pidx=0                              #index in the max_index array of current anchor
        l=len(max_index)                      #total number of points
        anchors_x=[]  #list of anchor points 
        anchors_y=[] #list of values   
        anchors_index=[] #index of anchor points in original arrays
       
        anchors_x.append(xs[max_index[0]])
        anchors_y.append(ys[max_index[0]]) 
        anchors_index.append(max_index[0])

        if(use_pmap):                                    
            r=p_map.r_map(xs[max_index[0]], p_ys, p_xs, min_lambda, r_min, r_max, nu)   #radius adjusted acording to penalty map
        else:
            r=r_min    
        while(True):
            M=deque()        #list of index of candidate points
            A=[]        #list of angles and index of candidate points
            while(not M):
                for i in range(Pidx+1, l):       #test all points to the right of P
                    Nx=xs[max_index[i]]
                    Ny=smooth.normalize (ys[max_index[i]],min_lambda, max_lambda, min_flux, max_flux, w_stretch)
                    if(Nx>P[0]+(2*r)):                   #if x>Px+(2*r) further points are outside
                        break
                    d= np.linalg.norm(P-np.array([Nx,Ny]))  
                    if(d<2*r and (P[0]!=Nx or P[1]!=Ny)):        #second condition to avoid duplicate points
                        M.append(i)           #save index of those inside the circ
                r=1.5*r             
                if(P[0]+(2*r)>furthest_point):       #stop searching for points and return anchors list
                    return anchors_x, anchors_y, anchors_index
            r=r/1.5
            while(M):                        #for all points in M, compute the angle
                Nidx=M.popleft()
                delta=np.array([xs[max_index[Nidx]]-P[0], smooth.normalize (ys[max_index[Nidx]],min_lambda, max_lambda, min_flux, max_flux, w_stretch)-P[1]])
                delta_norm=math.sqrt((delta[0]**2) + (delta[1]**2)) 
                delta_inv=np.array([-delta[1], delta[0]])
                h=math.sqrt((r**2)- ((delta_norm**2)/4))
                C=P+(0.5*delta)+((h/delta_norm)*delta_inv)
                A.append([Nidx, a_shape.angle(C[0], C[1], P[0], P[1], r)])  #save index and angle in A
        
            min_val = min(A, key=lambda v: v[1])                    #select the min angle
            min_idx=min_val[0]
            #print(P[0], P[1], max_pos[min_idx], max_heights[min_idx], "radius", r)
            P[0]= xs[max_index[min_idx]]                                          #update P to be the newly selected point
            P[1]= smooth.normalize (ys[max_index[min_idx]],min_lambda, max_lambda, min_flux, max_flux, w_stretch)
            Pidx=min_idx
            if(use_pmap):                                                 #update radius
                r=p_map.r_map(P[0], p_ys, p_xs, min_lambda, r_min, r_max, nu) 
            else:
                r=r_min       
            anchors_x.append(P[0])                                         #save point coordinates in anchors list
            anchors_y.append(ys[max_index[min_idx]])
            anchors_index.append(max_index[min_idx])

class continuum:     
    @staticmethod
    def interpolate(xs, ys, type):          #interpolate on x and y, acordding to type
        if(type=="cubic"):
            cs = interpolate.CubicSpline(xs, ys, bc_type='not-a-knot')
            return cs                       #returns cubic function
        elif(type=='linear'):
            ls= interpolate.interp1d(xs, ys, kind='linear', axis=-1, copy=True, bounds_error=False,fill_value="nan", assume_sorted=False)
            return ls            #returns linear function
    
class p_map:      #defines methods for computing the penalty map (to increase or decrease the radius in diferent zones)
    @staticmethod
    def rolling_max(ys, xs, w_size):   #adjust aprox continuum using rolling max (w_size=size of the window) and linear interpolation
        x_cont=[]
        y_cont=[]  
        i=0
        flag=0
        end_inter=min(xs)+w_size
        while(True):
            points_cont=[]   
            while(xs[i]<end_inter):
                points_cont.append((xs[i], ys[i]))
                i+=1
                if(i==(len(xs)-1)):
                    flag=1
                    break 
            max_p = max(points_cont, key=lambda v: v[1])      
            x_cont.append(max_p[0])
            y_cont.append(max_p[1])
            end_inter+=w_size 
            if(flag==1):
                s1=interpolate.interp1d(x_cont, y_cont, kind='linear', axis=-1, copy=True, bounds_error=False,fill_value="nan", assume_sorted=False)
                return s1
    @staticmethod
    def penalty(s1, s2, wavelengths):     #calculates the relative difference between continuums s1 and s2
            ps=[]            
            for w in wavelengths:
               if(math.isnan(s1(w)) or math.isnan(s2(w)) or s2(w)==0):
                    ps.append(0)
               else:  
                    ps.append(s2(w)-s1(w))
            minp=min(ps)   
            maxp=max(ps)
            for idx, p in enumerate(ps):
                ps[idx] =  (p - minp) / (maxp - minp)
            return ps
   
    @staticmethod
    def step_transform(ys, xs, step_size):    #transforms function into a step function, modifies original YS values
        peak_indices, peaks = find_peaks(ys, height=0, threshold=None, distance=step_size) 
        peak_heights = peaks['peak_heights']
        for i in range(1, len(peak_indices)+1):         #absolute maximums by descending order
            h2_peak_idx = peak_indices[np.argpartition(peak_heights,-i)[-i]]
            j=0
            while(h2_peak_idx+j<len(xs) and (xs[h2_peak_idx+j]<xs[h2_peak_idx]+step_size)):  #step right
                ys[h2_peak_idx+j]=ys[h2_peak_idx]
                j+=1  
            j=0    
            while(h2_peak_idx-j>=0 and (xs[h2_peak_idx-j]>xs[h2_peak_idx]-step_size)):      #step left
                ys[h2_peak_idx-j]=ys[h2_peak_idx]
                j+=1
        return ys, xs
                             
    @staticmethod
    def r_map( x ,p_y,p_x,lambda_min, r_min, r_max, nu):       #computes the radius at a given x point, p_y, p_x is the computed penalty
      p_idx=b_search(p_x, x)
      p=p_y[p_idx]
      c=x/lambda_min
      r=c*(r_min+(r_max-r_min)*(p**nu))
      return r



