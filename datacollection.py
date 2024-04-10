# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:51:26 2020

@author: Shuyu
"""

import pylab as pl
import numpy as np
from ball import Ball
from ball import Simulation

'''initialization'''


from ball import Ball
from ball import Simulation
import numpy as np
import pylab as pl
sim = Simulation(num = 100)
sim.run(1,animate = True,pause = 1e-12) #DO NOT CLOSE THE GRAPH WINDOW TO SEE UPDATE IN POSITIONS
                
'''save the velocities since it's random everytime'''
v = [] #will use in future simulations
pos = []
for ball in sim.listb[1:]:
    v.append(ball.vel())
    pos.append(ball.pos())
    
print('mean velocity (vx,vy): %r'%np.mean(v,axis= 0) )#should be 0    

'''Task 9         ~ 3 mins'''
#distance histograms
t =[]
KE_tot = []
momenx = []
momeny = []
for i in range(10):
    sim.runagain(100,animate = False,pause = 1e-11)
    print('time elapsed: %r s'%sim.timepassed)
    t.append(sim.timepassed)
    print('sum of impulse: %r kgm/s'%sim.sum_Impulse)
    print('average force on container: %r kgms^-2'%sim.F_avg())
    print('total kinetic energy of all balls: %r kgm^2s^-2'%sim.totalKE())
    print('total momenmtum of balls + container: (%r,%r) kgm/s'\
          %sim.total_momentum())
    KE_tot.append(sim.totalKE()+sim.container.ke())     
    momenx.append(sim.total_momentum()[0])
    momeny.append(sim.total_momentum()[1])
sim.runagain(1,animate = True,pause = 1e-11)   #to see update in positions
sim.dcentre()
sim.dinterball()

#also check conservations in the same run
pl.figure()
pl.plot(t,KE_tot,'x')
pl.xlabel('Time(s)')
pl.ylabel('KE (J)')
#pl.ylim(398.9,399.1)
pl.figure()
pl.plot(t,momenx,'x')
pl.ylim(-5e-4,5e-4)
pl.xlabel('Time(s)')
pl.ylabel('x momen (kgm/s)')
pl.figure()
pl.plot(t,momeny,'x')
pl.ylim(-5e-4,5e-4)
pl.ylabel('y momen (kgm/s)')
pl.xlabel('Time(s)')
#%%
'''TASK 10   ~20 mins'''
#PT and TK graphs

T = []
K = []
P = []
print('average kinetic energy of gas: %r J' %sim.avgKE())
print('Temperature: %r K'%sim.Temp())
print('Pressure: %r Pa*m' %sim.P_avg())
T.append(sim.Temp())
K.append(sim.avgKE())
P.append(sim.P_avg())

multipliers = np.arange(1.0,5.0,0.5)
for x in multipliers:
    sim1 = Simulation(pos,x*np.array(v),num = 100)
    sim1.run(1,animate = False,pause = 1e-12)
    for i in range(10):
        
        sim1.runagain(100,animate = False,pause = 1e-11)
        print('101 collisions performed')
        
    print('%r times the original speed done'%x)
    T.append(sim1.Temp())
    K.append(sim1.avgKE())
    P.append(sim1.P_avg())
    
pl.figure()
pl.plot(K,T,'x')
pl.xlabel('kinetic energy of 1 particle (J)')
pl.ylabel('Temperature (K)')
pl.figure()
pl.plot(T,P,'x')
pl.ylabel('Pressure (Pa*m)')
pl.xlabel('Temperature (K)')
#%%
''''Task 12'''
'''WARNING: TAKES HOURS'''
#ideal gas and radius of balls
#NEED T FROM PREVIOUS CELL

R = 10 #radius of container in m
V3 = np.pi*R**2
k_B = 1.38e-23
#N = sim1.numball
N = 100
gradient = N*k_B/V3
Tnew = 1.5*np.array(T) #correct to temperature&KE relation
#ideal gas equation PV = Nk_BT


def PT(r):
    '''
    finds P,T for 8 ball speeds
    WARNING : takes about an hour for each run
    '''
    
    T = []
    P = []
    multipliers = np.arange(1.0,5.0,0.5)
    for x in multipliers:
        sim1 = Simulation(pos,x*np.array(v),r,num = 100)
        sim1.run(1,animate = False,pause = 1e-12)
        for i in range(10):
            
            sim1.runagain(100,animate = False,pause = 1e-11)
            print('101 collisions performed')
            
        print('%r times the original speed done, container radius=%r,\
              ball radius = %r'%(x,sim1.container._radius,sim1.listb[20]._radius))
        T.append(sim1.Temp())
        P.append(sim1.P_avg())    
    return P,T
multipliers =np.arange(0.01,3,0.3)
r = multipliers*0.1
r = r[:3] #ball radii to be passed

T4 = []
P4 = []
for x in r:
    P_,T_ = PT(x) '''AN HOUR FOR EACH CALL'''
    P4.append(P_)
    T4.append(T_)
    print('atomic radius = %gm done')

pl.figure()
pl.plot(T4[0],P4[0],'x',label = 'r = 0.1cm',color = 'r')
pl.legend()
pl.ylabel('Pressure (Pa*m)')
pl.xlabel('Temperature (K)')
#%%
'''plotting and fitting PT graphs for different ball radii'''
multipliers =np.arange(0.01,3,0.3)
r = multipliers*0.1
r = r[:3] #ball radii to be passed
i = 0
r = multipliers*0.1*100
pl.figure()
Tnew = 1.5*np.array(T4[3])
pl.plot(Tnew,gradient*Tnew,label = 'Ideal Gas Eqn')
for x,y in zip(T4,P4):
    temp = 1.5*np.array(x)
    p = y
    line = pl.plot(temp,p,'x', label = 'r = %gcm'%r[i] )
    c = line[0].get_color()
    if i ==0:
        temp1 = temp[:-1].copy() #outlier point
        p1 = p[:-1].copy()#outlier point
        fit,cov = np.polyfit(temp1,p1,1,cov = True)
    else:
        fit,cov = np.polyfit(temp,p,1,cov = True)
    poft = np.poly1d(fit)
    pl.plot(temp,poft(temp),color = c)
    i+=1

pl.legend()
pl.ylabel('Pressure (Pa*m)')
pl.xlabel('Temperature (K)')
#%%
'''Task 14'''
# Fit PT graph to van der Waals law
# NEED T4 FROM TASK 12, OR IMPORT T4,P4 VALUES PROVIDED FROM A PREVIOUS RUN
n = 100/6.02e23 #N/N_A
N=100
k_B = (1.38e-23)
R = 10 #radius of container in m
V3 = np.pi*R**2
r = 0.031 #radius of atom in T4[1]
gradient = N*k_B/V3
Temp = 1.5*np.array(T4[1]) #correction to temperature and KE formula
Pres = np.array(P4[1])


fit, cov = np.polyfit(Temp,Pres,1,cov = True)
pp = np.poly1d(fit)
pl.figure()
pl.plot(Temp,Pres,'x')
pl.plot(Temp, pp(Temp),label = 'VdW fit')
pl.plot(Temp,gradient*Temp,label = 'Ideal Gas Eqn')
pl.legend()
pl.xlabel('Temperature (K)')
pl.ylabel('Pressure (Pa)')

grad = fit[0]
interc = fit[1]
#grad = Nk_B/(V3/n - b)
b = V3/n - N*k_B/grad
#b1=V3/N - k_B/grad
print('b = %g'%b)

a = -interc*V3**2/N**2
print('a = %g'%a)
#%%
'''Task 13 - velocity distribution & Boltzmann'''
#will need to have an instance of sim, run first cell again if necessary

from scipy.optimize import curve_fit
Temp = sim.Temp()

def boltz(v,A,vmean):
    k_B = 1.38e-23
    m = 1
    return A*v*np.exp(-(0.5*m*(v-vmean)**2)/(k_B*Temp))

v13 = []
pos13 = []
for ball in sim.listb[1:]:
    v13.append(ball._vel)
    pos13.append(ball._pos)
pl.figure()
pl.hist(np.array(v13)[:,0],bins = 20)
pl.xlabel('x velocity of balls(m/s)')
pl.ylabel('number of balls')
pl.figure()
pl.hist(np.array(v13)[:,1],bins = 20)
pl.xlabel('y velocity of balls(m/s)')
pl.ylabel('number of balls')

pl.figure()
vmag =[]
for x in v13:
    vmag.append(np.sqrt(np.dot(x,x)))
heights, edges, patches = pl.hist(vmag,bins = 20)
bin_centres = edges[:-1] + np.diff(edges) / 2
guess = [10,2]
popt, pcov = curve_fit(boltz,bin_centres,heights,p0=guess)
vpoints = np.arange(0,8,0.25)

pl.plot(vpoints,boltz(vpoints,*popt))
pl.xlabel('velocity of balls(m/s)')
pl.ylabel('number of balls')
#pl.ylabel('y velocity of balls(m/s)')