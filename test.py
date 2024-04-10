# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:49:27 2020

@author: Shuyu
"""

'''
to do
hide b1, container
#Ball.numball counts repeated instantiations        
        
BALL ESCAPES IF STARTS FROM 9
'''
#%%
import pylab as pl
import numpy as np
from ball import Ball
from ball import Simulation
#%%check that times to collision is correct

from ball import Ball
a = Ball(pos = [2,0],vel = [-1,0])
b = Ball(pos = [-2,0],vel = [1,0])

print(b.time_to_collision(a))
print(a.time_to_collision(b))
#should be equal

#%%
import pylab as pl

f = pl.figure()
patch1 = pl.Circle([0., 0.], 4, fc='r') # coords, radius, facecolour
patch2 = pl.Circle([5., 2.], 4, fc='b')
patch3 = pl.Circle([-2., 0.], 4, ec='b',fill = False, ls='solid')
ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.add_patch(patch1)
ax.add_patch(patch2)
ax.add_patch(patch3)
pl.show()

#%%

f = pl.figure()
patch = pl.Circle([-4., -4.], 3, fc='r')
ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.add_patch(patch)

pl.pause(1) #redraws and pauses, to see the previous position of the ball before updating
patch.center = [4, 4]
pl.pause(1)
pl.show()

#%%
import pylab as pl

f = pl.figure()
patch = pl.Circle([-10., -10.], 1, fc='r')
ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.add_patch(patch)

for i in range(-10, 10):
    patch.center = [i, i]
    pl.pause(0.001)
pl.show()
#%%
'''initial simulation'''


from ball import Ball
from ball import Simulation
import numpy as np
import pylab as pl
#sim = Simulation(bpos = [0,0],bvel = [2**-0.5,2**-0.5])
sim = Simulation(num = 100)
sim.run(1,animate = True,pause = 1e-12) #DO NOT CLOSE THE GRAPH WINDOW TO SEE UPDATE IN POSITIONS
print('time elapsed: %r s'%sim.timepassed)
print('sum of impulse: %r kgm/s'%sim.sum_Impulse)
print('average force on container: %r kgms^-2'%sim.F_avg())
print('total kinetic energy of all balls: %r kgm^2s^-2'%sim.totalKE())
print('total momenmtum of balls + container: (%r,%r) kgm/s'%sim.total_momentum())

'''save the velocities since it's random everytime'''
v = []
pos = []
for ball in sim.listb[1:]:
    v.append(ball._vel)
    pos.append(ball._pos)
    
print('mean velocity (vx,vy): %r'%np.mean(v,axis= 0) )#should be 0    
#%%
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
    print('total momenmtum of balls + container: (%r,%r) kgm/s'%sim.total_momentum())
    KE_tot.append(sim.totalKE()+sim.container.ke())
    momenx.append(sim.total_momentum()[0])
    momeny.append(sim.total_momentum()[1])
sim.runagain(1,animate = True,pause = 1e-11)
sim.dcentre()
sim.dinterball()
#%%
pl.figure()
pl.plot(t,KE_tot,'x')
pl.xlabel('Time(s)')
pl.ylabel('KE (J)')
pl.ylim(398.9,399.1)
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
sim.runagain(1,animate = True,pause = 1e-11)
print('time elapsed: %r s'%sim.timepassed)
print('sum of impulse: %r kgm/s'%sim.sum_Impulse)
print('average force on container: %r kgms^-2'%sim.F_avg())
print('total kinetic energy of all balls: %r kgm^2s^-2'%sim.totalKE())
sim.dcentre()

#%%
'''SIM1'''
#animation is on at the start and end only

sim1 = Simulation(pos, v,num = 100)
sim1.run(1,animate = True,pause = 1)
for i in range(10):
    sim1.runagain(100,animate = False,pause = 1e-11)
    print('100 collisions done')
sim1.runagain(1,animate = False,pause = 1e-11)
#print('time elapsed: %r s'%sim1.timepassed)
#print('sum of impulse: %r kgm/s'%sim1.sum_Impulse)
#print('average force on container: %r kgms^-2'%sim1.F_avg())
#print('total kinetic energy of all balls: %r kgm^2s^-2'%sim1.totalKE())

#%%
'''TASK 10'''
#RADIUS OF CONTAINER MIGHT BE WRONG HERE


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
#%%
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
R = 10 #radius of container in m
V3 = np.pi*R**2
#V3 = V3**(2/3)
k_B = 1.38e-23
#N = sim1.numball
N = 100
gradient = N*k_B/V3
Tnew = 1.5*np.array(T)
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
r = r[:3] 

T4 = []
P4 = []
for x in r:
    P_,T_ = PT(x)
    P4.append(P_)
    T4.append(T_)
    print('atomic radius = %gm done')
#%%
pl.figure()
pl.plot(T4[0],P4[0],'x',label = 'r = 0.1cm',color = 'r')
pl.legend()
pl.ylabel('Pressure (Pa*m)')
pl.xlabel('Temperature (K)')
#%%
'''plotting equation of state for different ball radii'''
i = 0
r = multipliers*0.1*100
pl.figure()
pl.plot(Tnew,gradient*Tnew,label = 'Ideal Gas Eqn')
for x,y in zip(T4,P4):
    temp = 1.5*np.array(x)
    p = y
    line = pl.plot(temp,p,'x', label = 'r = %gcm'%r[i] )
    c = line[0].get_color()
    if i ==0:
        temp1 = temp[:-1].copy()
        p1 = p[:-1].copy()
        fit,cov = np.polyfit(temp1,p1,1,cov = True)
    else:
        fit,cov = np.polyfit(temp,p,1,cov = True)
    poft = np.poly1d(fit)
    pl.plot(temp,poft(temp),color = c)
    i+=1

#pl.plot(Tnew,P,'x',label='r=10cm')

pl.legend()
pl.ylabel('Pressure (Pa*m)')
pl.xlabel('Temperature (K)')
#%%
'''Task 14'''

n = 100/6.02e23 #N/N_A
N=100
k_B = 1.38e-23
R = 10 #radius of container in m
V3 = np.pi*R**2
r = 0.031 #radius of atom in T4[1]
gradient = N*k_B/V3
Temp = 1.5*np.array(T4[1]) #correction to temperature and KE formula
Pres = np.array(P4[1])
#from scipy.optimize import curve_fit
#
#def P_VdW(T,b):
#    return (N*k_B*T)/((V3/n) -b)
#popt,pcov = curve_fit(P_VdW, Temp,Pres,p0 =N*np.pi*r**2 )
#fitb = P_VdW(Temp,*popt)

fit, cov = np.polyfit(Temp,Pres,1,cov = True)
pp = np.poly1d(fit)
pl.figure()
pl.plot(Temp,Pres,'x')
pl.plot(Temp, pp(Temp),label = 'VdW fit')
#pl.plot(Temp,fitb,label='VdW fit')
pl.plot(Temp,gradient*Temp,label = 'Ideal Gas Eqn')
pl.legend()
pl.xlabel('Temperature (K)')
pl.ylabel('Pressure (Pa)')

grad = fit[0]
interc = fit[1]
#grad = Nk_B/(V3/n - b)
b = V3/n - N*k_B/grad
#b1=V3/N - k_B/grad
#berr = N*k_B/np.sqrt(cov[0,0])
print('b = %g'%b)

a = -interc*V3**2/N**2
print('a = %g'%a)
#%%
'''Task 13'''
from scipy.optimize import curve_fit
Temp = sim1.Temp()
#m = sim1.listb[2].mass()

def boltz(v,A,vmean):
    k_B = 1.38e-23
    m = 1
    return A*v*np.exp(-(0.5*m*(v-vmean)**2)/(k_B*Temp))

v13 = []
pos13 = []
for ball in sim1.listb[1:]:
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
#%%

'''Task 11'''
T1 = []
K1 = []
P1 = []
V1 = [] #actually area in 2d

multipliers = np.arange(1.0,5.0,0.5)
for x in multipliers:
    sim1 = Simulation(pos,v,num=100,R=10*x)
    sim1.run(1,animate = False,pause = 1e-12)
    for i in range(10):
        
        sim1.runagain(100,animate = False,pause = 1e-11)
        print('101 collisions performed')
        
    print('%r times the original speed done'%x)
    T1.append(sim1.Temp())
    K1.append(sim1.avgKE())
    P1.append(sim1.P_avg())
    vol = np.pi*((10*x)**2)
    V1.append(vol)
#%%
pl.figure()
pl.plot(K1)
pl.figure()
pl.plot(P1,V1,'x')
pl.xlabel('Pressure (Pa*m)')
pl.ylabel('Area (m^2)')
#%%
'''Task 11 graph 2'''


#%%
def web(points):
    
        radius = 1
        rings = (10-2*radius)//(2*radius)
        r = []
        for i in range(1,rings+1):
            r.append(np.full((points,2),2*i*radius))
            
        return r
r1,r2,r3,r4 = web(4)
#%%
  
def web():
        
        radius = 0.1
        rings = (10-2*radius)//(2*radius)
        points,dtheta = self.spacings(radius,rings)
#        points = int(points)
#        print(points)
#        print(dtheta)
        r = []
        for i in range(1,int(rings+1)):
            r.append(np.full((int(points[i-1]),2),2*i*radius,dtype =np.float32))
#        print(r)
#        r = np.asarray(r)
#        print(np.shape(r))
#        r.ravel()
        pcoords = []
        for i in range(np.shape(r)[0]):
            a = r[i]
            for j in range(np.shape(a)[0]):
                a[j][1] = 0
                a[j][1] += j*dtheta[i]
                pcoords.append(a)

#        print(pcoords)
#        print(np.shape(pcoords)  )          
#        r1,r2,r3,r4 = pcoords
#        pcoords  = np.vstack((r1,r2,r3,r4 ))

            
#        newarr = np.array_split(pcoords)
#        pcoords = np.vstack(newarr)
        cartcoords = self.tocart(pcoords)
#        xy = np.hstack(cartcoords)
#        xy = np.array_split(xy,) 
#        print(cartcoords)
#            print(a)
            
#        print(r)
        
        return pcoords, cartcoords
#        return pcoords
#        return r
#%%
import pylab as pl

def run(self, num_frames, animate=False):
    if animate:
        f = pl.figure()
        ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
        ax.add_artist(self._container.get_patch())
        ax.add_patch(self._ball.get_patch())
    for frame in range(num_frames):
        self.next_collision()
        if animate:
            pl.pause(0.001)
    if animate:
        pl.show()