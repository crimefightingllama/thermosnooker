# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:39:25 2020

@author: Shuyu
"""

import numpy as np
import pylab as pl

class Ball():
    '''
    Ball object
    '''
#    time = 0
    numball = 0
    sum_Impulse = 0
    
    def __init__(self,mass = 1, radius = 1, pos = [0.0,0.0],vel =[0,0],cont = False):
        Ball.numball+=1
        self._mass = mass
        self._radius = radius
        self._pos = np.asarray(pos,dtype = np.float32)
        self._vel = np.asarray(vel,dtype = np.float32)
        self._ke = self.ke()
        if cont:
            self._patch = pl.Circle(pos,radius, ec = 'b',fill = False)
        else: 
            self._patch = pl.Circle(pos,radius, fc = 'r')
    def __repr__(self):
        return "Ball(mass = %r, radius = %r)"%(self._mass, self._radius)
    def get_patch(self):
        return self._patch
    def pos(self):
        return self._pos
    def vel(self):
        return self._vel
    def mass(self):
        return self._mass
    def ke(self):
        return 0.5*self._mass*np.dot(self._vel,self._vel)
    def move(self,dt):
        self._pos += self._vel*dt
        self._patch.center = self._pos
#        Ball.time +=dt
        
#    def clock(self,dt):
#        Ball.time += dt
    def time_to_collision(self,other, wall = False):
        del_pos = self._pos - other._pos
        rr = np.dot(del_pos,del_pos)
        del_v = self._vel - other._vel
        vv = np.dot(del_v, del_v) ##cant = 0 ! division by 0!
        rv = np.dot(del_pos, del_v)
        Rbb = self._radius + other._radius
        Rbw = self._radius - other._radius #radius of container
        if not wall:
            R = np.sqrt(rr)-Rbb
        else:
            R = 1 #arbtrary
        
        if not wall :
            discrim = 4*rv**2 - 4*vv*(rr-Rbb**2)
        else:
            discrim = 4*rv**2 - 4*vv*(rr-Rbw**2)
            
        if discrim < 0:
        #"does not collide"
            return 1e10
        elif vv == 0: #division by 0!
            return 1e10
        elif R <=0:
            #already touching, CANT COINCIDE
            return 1e10
        else:
            time1 = (-2*rv + np.sqrt(discrim))/(2*vv)
            time2 = (-2*rv - np.sqrt(discrim))/(2*vv)
            ## 2 times for a bw collision, and bb collsion,both past and future

            if time1 <=0 and time2<=0:
#             "moving away from each other!"
                return 1e10
            elif time1<=0:
                return time2
            elif time2<=0:
                return time1
            else:
                a = np.array([time1,time2])
                if not wall:
                    return np.amin(a)
                else:
                    return np.amax(a)
                
    def collide(self,other,wall = False):
        if wall:
            vmag = np.sqrt(np.dot(other._vel, other._vel))
            impulse = 2*other._mass*vmag  #2m(vf-vi)
            Ball.sum_Impulse += impulse
            
        del_r = self._pos - other._pos
        len_r = np.sqrt(np.dot(del_r,del_r))
        unit_r= del_r/len_r #to unit vector, so can find projection of velocities unto line between centres
        u1_para = np.dot(self._vel,unit_r)*unit_r
        u1_perpen = self._vel - u1_para
        u2_para = np.dot(other._vel,unit_r)*unit_r
        u2_perpen = other._vel - u2_para
        m1 = self._mass
        m2 = other._mass
        v2_para =( 2*u1_para + u2_para*(m2-m1)/m1 )*m1/(m1+m2)
        v1_para =( 2*u2_para + u1_para*(m1-m2)/m2 )*m2/(m2+m1)
        self._vel = v1_para + u1_perpen # perpendicular component does NOT change
        other._vel= v2_para + u2_perpen

class Simulation(Ball):
    '''
    Container of radius 10
    '''
    def __init__(self,*args,num = 2,R = 10):    
        Ball.time = 0 #initialise for this simulation
        Ball.numball = -1 #not counting container
        Ball.sum_Impulse = 0
        self.container = Ball(mass = 10e20,radius = R, pos = [0,0],vel = [0,0],cont = True)
        #self.b1 = Ball(mass = 1, radius = 1, pos = bpos, vel = bvel)
        self._num = num
        self.timepassed = 0
        self.listb = [''] #first one is empty, to start counting from index 1

        if args ==():
            
            
            places = self.place(num)
            velocity = self.vdist(num)
            for ball in range(num):
                self.listb.append(Ball(mass = 1, radius = 0.1, pos = places[ball],vel = velocity[ball]))
        else:
#            print(args)
            pos = args[0]
            v = args[1]
            if np.size(args)==3:            
                r = args[2]
            else:
                r = 0.1 #default radius
                    
            for ball in range(num):
                self.listb.append(Ball(mass = 1, radius = r, pos = pos[ball],vel = v[ball]))            
#        
    
    def __repr__(self):
        
        return 'Simulation(container radius: %r,num = %r, ball radius =%r)'%(\
                          self.container._radius,self.numball\
                          ,self.listb[1]._radius)
    
    def clock(self,dt):
        self.timepassed+=dt
        
    def place(self,num):
#        X,Y = np.mgrid[-8:10:2,-8:10:2]
#        xy = np.vstack((X.flatten(), Y.flatten())).T
#        #print(xy)
        xy = self.web()
        step = int(np.shape(xy)[0]//num)
        
        places = xy[0::step]
        places = places[:num]
        return places
    
    def web(self):
        '''draws a web shaped(polar coords) for ball position initialization, 
        then converts to cartesian'''
        
        radius = 0.1
        rings = (10-2*radius)//(2*radius)
        points,dtheta = self.spacings(radius,rings)
        r = []
        for i in range(1,int(rings+1)):
            r.append(np.full((int(points[i-1]),2),2*i*radius,dtype =np.float32))
        pcoords = []
        for i in range(np.shape(r)[0]):
            a = np.copy(r[i])
            for j in range(np.shape(a)[0]):
                a[j][1] = 0
                a[j][1] += j*dtheta[i]
            pcoords.append(a)


        pcoords = np.concatenate(np.asarray(pcoords))  
        cartcoords = self.tocart(pcoords)

        return cartcoords
    
    def vdist(self,num):
        '''assign velocities in a random, uniform distribution, with mean 0'''
        vx = 2*np.random.randn(num)
        vxmean = np.mean(vx)
        vx -= vxmean #offset to make mean speed 0
        vy = 2*np.random.randn(num) 
        vymean = np.mean(vy)
        vy -= vymean
        vel = np.column_stack((vx,vy))
        return vel
        
    def tocart(self,pcoords):
        '''convert polar coordinates to cartesian coordinates'''
        newcoords = []
        for coord in pcoords:
            r = coord[0]
            theta = coord[1]
            x= r*np.cos(theta)
            y =r*np.sin(theta)
            newcoords.append([x,y])
        return newcoords
    
    def spacings(self,radius,rings):
        dtheta = []
        points = []
        for i in range(1,int(rings+1)):   
           theta = np.arcsin(1/(2*i))
           theta = 2*theta
           pts = (2*np.pi)//theta
           dtheta.append(theta)
           points.append(pts)
           
        return points,dtheta #number of points at a given distance 
    #from centre, and dtheta at each distance
        
    def totalKE(self):
        list_ = self.listb[1:]
        total = 0
        for ball in list_:
            total+= ball.ke()
        return total
    
    def total_momentum(self):
        list_ = self.listb[1:]
        px = 0
        py = 0
        for ball in list_:
            px += ball._mass*ball._vel[0]
            py += ball._mass*ball._vel[1]
            
        px += self.container._mass* self.container._vel[0]
        py += self.container._mass*self.container._vel[1]
        return px,py
    
    def avgKE(self):
        list_ = self.listb[1:]
        v2total = 0
        mass = list_[0]._mass
        for ball in list_:
            v2total+= np.dot(ball._vel, ball._vel)
        num = np.shape(list_)[0]
        vms = v2total/num
        avgKE = 0.5*mass*vms
        return avgKE
    
    def Temp(self):
        k_B = 1.38e-23
        T = 2/3*self.avgKE()/k_B
        return T
    
    def next_collision(self,dt,pause,animate = False):
        list_ = self.listb[1:]
        ddt = 0.3
        step = dt//ddt
        if animate:          
            for i in range(0,int(step)):
     
                for ball in list_:
                    ball.move(ddt)
        
                pl.pause(pause)
                
            for ball in list_:
                ball.move(dt-step*ddt)
            pl.pause(pause)
        else:
            
            for ball in list_:
                ball.move(dt)                
        self.clock(dt)
        if self._nextball[1][0] == 0: #with container
            self.container.collide(self.listb[self._nextball[0][0]],wall = True) 
            #prints container vel then ball vel
        else: #with another ball
            self.listb[self._nextball[0][0]].collide(self.listb[self._nextball[1][0]])
            
    def nextt(self):
        num = self._num
        times = np.full((num+1,num+1),1e10) #1e10 is null value, ignore
        #print('initial times: %r' %times)
        for x in range(1,num+1):
            for y in range(num+1):
                if y ==0:
#                    print(x,y)
                    times[x][y]=self.container.time_to_collision(self.listb[x],wall = True)
                else:
#                    print(x,y)
                    times[x][y] = self.listb[x].time_to_collision(self.listb[y])   
        times = np.asarray(times)
        dt = times.min()
        self._nextball = np.where(times == dt)    #(ball1, ball2/container)   
        return dt   
         
    def run(self, num_frames, animate=False,pause =1e-11):
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10),aspect = 1)
            ax.add_artist(self.container.get_patch())
            list_ = self.listb[1:]
            for ball in list_:
                ax.add_patch(ball.get_patch())
            pl.pause(0.1)
            times = []
        for frame in range(num_frames):
            #dt = self.b1.time_to_collision(self.container,wall = True)
            dt = self.nextt()
#            times.append(dt)
         
#            if animate:
#                pl.pause(0.1)
            self.next_collision(dt,pause)   #contains move() which updates the patch centre
                #pl.pause(0.5)
        if animate:
            pl.show()
            
    def runagain(self, num_frames, animate=False,pause =1e-11):
        for frame in range(num_frames):
            #dt = self.b1.time_to_collision(self.container,wall = True)
            dt = self.nextt()
#            times.append(dt)
         
#            if animate:
#                pl.pause(0.1)
            self.next_collision(dt,pause,animate)   #contains move() which updates the patch centre
                #pl.pause(0.5)
        if animate:
            pl.show()
        
        
    def F_avg(self):
        F = self.sum_Impulse/self.timepassed
        return F
    def P_avg(self):
        L= 2*np.pi*10
        P = self.F_avg()/L
        return P
    
    def dcentre(self):
        pl.figure()
        list_ = self.listb[1:]
        distance = []
        for ball in list_:
            dis = np.sqrt(np.dot(ball._pos, ball._pos))
            distance.append(dis)
        pl.hist(distance,bins = 20)
        pl.xlabel('distance from centre(m)')
        pl.ylabel('number of balls')
        
    def dinterball(self):
        pl.figure()
        list_ = self.listb[1:]
        balls = np.shape(self.listb[1:])[0]
        distance = []
        for i in range(1,balls):
            for j in range(i+1,balls+1):
                r1r2 = self.listb[i]._pos - self.listb[j]._pos
                sep = np.sqrt(np.dot(r1r2,r1r2))
                distance.append(sep)
        pl.hist(distance,bins = 20)
        pl.xlabel('separation(m)')
        pl.ylabel('pairs')
        
        