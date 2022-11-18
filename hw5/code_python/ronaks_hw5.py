"""
Created on Thu Oct 17 21:41:50 2019

@author: ronaktali
"""


from pulp import *
import numpy as np
import matplotlib.pyplot as plt

#a handful of sites
sites = ['org','A','B','C','D','E','F','G','H','I','J','K', 'L', 'M','N','O','P','Q', 'R','S']




coordx = [np.random.randint(1,20) for b in sites]
coordy = [np.random.randint(1,20) for b in sites]


#Initially Plotting the Points
plt.plot(coordx, coordy, 'ro')
plt.axis([0, 25, 0, 25])
plt.show()


def euclid(m,n):
    return (m[0] - n[0])**2 + (m[1] - n[1])**2 

distances = {}

for i in range(0,20):
    for j in range (0, 20):
        if i != j:
            distances[(sites[i], sites[j])] = euclid([coordx[i], coordy[i]], [coordx[j], coordy[j]])
    




#non symetric distances
#distances = dict( ((a,b),np.random.randint(1,361)) for a in sites for b in sites if a!=b )

#create the problme
prob=LpProblem("salesman",LpMinimize)



#indicator variable if site i is connected to site j in the tour
x = LpVariable.dicts('x',distances, 0,1,LpBinary)

#the objective
cost = lpSum([x[(i,j)]*distances[(i,j)] for (i,j) in distances])
prob+=cost

#constraints
for k in sites:
    #every site has exactly one inbound connection
    prob+= lpSum([ x[(i,k)] for i in sites if (i,k) in x]) ==1
    #every site has exactly one outbound connection
    prob+=lpSum([ x[(k,i)] for i in sites if (k,i) in x]) ==1
    
#we need to keep track of the order in the tour to eliminate the possibility of subtours
u = LpVariable.dicts('u', sites, 0, len(sites)-1, LpInteger)

#subtour elimination
N=len(sites)
for i in sites:
    for j in sites:
        if i != j and (i != 'org' and j!= 'org') and (i,j) in x:
            prob += u[i] - u[j] <= (N)*(1-x[(i,j)]) - 1
            


prob.solve()




print(LpStatus[prob.status])

sites_left = sites.copy()
org = 'org'
tour=[]
tour.append(sites_left.pop( sites_left.index(org)))

while len(sites_left) > 0:
    
    for k in sites_left:
        if x[(org,k)].varValue ==1:
            tour.append( sites_left.pop( sites_left.index(k)))
            org=k
            break
            
tour.append('org')

tour_legs = [distances[(tour[i-1], tour[i])] for i in range(1,len(tour))]

print('Found optimal tour!')
print(' -> '.join(tour))



sum(tour_legs)




#Plotting Logic



for i in range(len(tour)-1):
    plt.plot([coordx[sites.index(tour[i])],coordx[sites.index(tour[i+1])]], [coordy[sites.index(tour[i])],coordy[sites.index(tour[i+1])]], 'ro-')

plt.show()





