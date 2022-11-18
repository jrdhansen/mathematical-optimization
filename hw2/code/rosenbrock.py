# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:41:59 2019

@author: jrdha
"""



import numpy as np
import numpy.linalg as npla


def f(x):

    return (1 - x[0])**2+ 100*((x[1]-x[0]**2)**2)

   

# first order derivatives of the function

def fdx(x):

    #df1 = 400*(x[0]**3 - x[0]*x[1])+2*(x[0]-1)

    df1 = -2*(1 - x[0]) - (400*x[0])*(x[1] - (x[0]**2))

    df2 = 200*(x[1] - (x[0]**2))

    return np.array([df1, df2])

fdx([2.0,2.0]) 




def Hessian(x):
    d2f_dx2 = 2 - 400*x[1] + 1200 * (x[0]**2)
    d2f_dyx = -400*x[0]
    d2f_dy2 = 200
    return(np.matrix([[d2f_dx2, d2f_dyx], [d2f_dyx, d2f_dy2]]))
    
Hessian([2.0,2.0])




def backtrack2(x0, f, fdx, t = 1, alpha = 0.2, beta = 0.8):

    while f(x0 - t*fdx(x0)) > f(x0) + alpha * t * np.dot(fdx(x0).T, -1*fdx(x0)):

        t *= beta

        #print(f(x0 - t*fdx(x0)) - (f(x0) + alpha * t * np.dot(fdx(x0).T, -1*fdx(x0))), t)

    return t





backtrack2([1.0,2.0], f, fdx)










def grad(point, max_iter):

    iter = 1


   # grad2 = -1*DerrivRosenbrock1(x) # Calculate Gradient at x



  

    #lrate = 0.0002

   

    while (np.linalg.norm(fdx(point)) > 0.000001):

       

        #Find t by backtracking

        t = backtrack2(point, f, fdx) #Step Size

        #Update x ---- i.e. get to new point

        #x = x + t*grad2

        #point = point - np.dot(t,fdx(point))

        point = point - np.dot(t, fdx(point))

        #Calculate New Value of Function

        print(point, f(point), fdx(point), iter)

       

        iter += 1

       

        if iter > max_iter:

            break

   

    return point, f(point), iter



grad([2.0,2.0], 10000)




def lambda_sq(fdx, Hessian, point):
    lambda_sq = np.dot(np.dot(fdx(point) , npla.pinv(Hessian(point))) , fdx(point).T)
    return(np.asscalar(lambda_sq))
    

lambda_sq(fdx, Hessian, [2.0,2.0])



def delta_x(fdx, Hessian, point):
    delta_x = np.dot(-npla.pinv(Hessian(point)) , fdx(point).T)
    return(delta_x)





def newtons_method(x, eps=0.00001, max_iters=20000):
    # Compute 
    iters = 0
    lmb_sq = lambda_sq(fdx, Hessian, x)
    while((lmb_sq/2.0) > eps):
        # Compute delta_x and lambda_sq
        dlt_x = delta_x(fdx, Hessian, x)
        #Line search for t
        t = backtrack2(x, f, fdx)

        # Update x
        x = np.array((x + np.dot(t , dlt_x)))[0]
        print("iters: ", iters)
        print("f(x): ", f(x))
        # Update lmb_sq, see if we still stay in the loop
        lmb_sq = lambda_sq(fdx, Hessian, x)
        iters += 1
        if(iters > max_iters):
            break

    return(x, f(x), iters)

#x = [2.0,2.0]

newtons_method([2.0,2.0])

newtons_method([1.0, -1.0])

newtons_method([100.0, -1.0])




 