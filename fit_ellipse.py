#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:19:05 2021

@author: Constantine
"""

from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle

import math
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from kivy.core.window import Window


class MyPaintWidget(Widget):
    fx=0;fy=0
    chosen_ix = [];
    bestr = 0
    bestrmaxEL = 0
    best_noofpoints = 0
    best_noofpointsEL = 0

    def bootit(self):
        """Draws the grid every time"""
        color = (100, 0.5, 0.5)
        with self.canvas:
            Color(*color, mode='hsv')
            for i in range(0,FitEllipseApp().xv.shape[0]):
                for j in range(0,FitEllipseApp().xv.shape[1]):
                    Rectangle(pos=(FitEllipseApp().xv[i,j] - FitEllipseApp().rectsize / 2, FitEllipseApp().yv[i,j] - FitEllipseApp().rectsize / 2), size=(FitEllipseApp().rectsize, FitEllipseApp().rectsize))
    
    def on_touch_down(self, touch): 
        """Finds if the user clicked in a rect and if yes toggles it"""
        # bring the current loc in the form for cdist
        fixr = np.tile([touch.x,touch.y],(FitEllipseApp().xv1.shape[0],1))
        # calc all grid distances from the current pos, rank them and take the closest to see if in a rect
        alld = cdist(fixr,FitEllipseApp().all,'euclidean')
        dists = alld[0,:]
        ixs = np.argsort(dists)
        # if in rect change its color,store it in the list of chosen1s 
        if dists[ixs[0]] < FitEllipseApp().rectsize:
            if ixs[0] in self.chosen_ix:
                self.chosen_ix.remove(ixs[0])
                color = (100, 0.5, 0.5)#grey
                with self.canvas:
                    Color(*color, mode='hsv')
                    Rectangle(pos=(FitEllipseApp().xv1[ixs[0]] - FitEllipseApp().rectsize / 2, FitEllipseApp().yv1[ixs[0]] - FitEllipseApp().rectsize / 2), size=(FitEllipseApp().rectsize, FitEllipseApp().rectsize))                
            else:
                self.chosen_ix.append(ixs[0])
                color = (50, 1, 1) #red
                with self.canvas:
                    Color(*color, mode='hsv')
                    Rectangle(pos=(FitEllipseApp().xv1[ixs[0]] - FitEllipseApp().rectsize / 2, FitEllipseApp().yv1[ixs[0]] - FitEllipseApp().rectsize / 2), size=(FitEllipseApp().rectsize, FitEllipseApp().rectsize))

    def fitellipse(self):
        """fit an ellipse to the chosen points by a simple guided random walk that minimizes the mean square of the sum of the distances to the foci"""
#        some_ixs = np.array(self.chosen_ix)
        if len(self.chosen_ix)==0: return
         # if mean square dist less than this stop, it increases with the # of chosen points 
        terminationcritirio = len(self.chosen_ix)
        # here stop no mater what
        maxiterations = 1000
        notdone = 1 #for while loop that follows

#        print(self.chosen_ix)
        some_ixs = self.chosen_ix
        some_xvs = FitEllipseApp().xv1[some_ixs];mx = np.mean(some_xvs) 
        some_yvs = FitEllipseApp().yv1[some_ixs];my = np.mean(some_yvs)
#        some_dists = dists[some_ixs]
        #bring the chosen coords to the form pdist likes to calculate their distances and estimate initial axis of the ellipse 
        some_all = np.stack((some_xvs,some_yvs), axis =1)
        
        if self.bestrmaxEL==0 or len(self.chosen_ix) != self.best_noofpointsEL:
            alld = squareform(pdist(some_all,'euclidean'))
            # estimate initial big axis from the max of distances
            rad_max = 0.5*np.max(alld[0,:])
            # estimate initial small axis from the average of distances
            rad_min = 0.5*np.mean(alld[0,:])
            # covariance of the chosen points to use for finding the dominant (perron) eigenvector
            SAC = np.cov(np.transpose(some_all))
            # eiganvalue, eigenvector decomposition
            W, V = np.linalg.eig(SAC)
            #find index of largst eigenvalue
            ixs = np.argsort(-1*abs(W)) #negate to make it descending
            #this is tthe dominat eigenvector, the slope of the ellipse is estimeted on this direction (as of standard PCA)
            perron_v = V[:,ixs[0]]
    #        print(some_all.shape, SAC.shape, W.shape)
    #        print(perron_v)
            #estimate the slope from the eigenvectror
            theta = np.arctan(perron_v[1]/perron_v[0])
            #estimate initial center from the mean xs and ys
            centerx = mx; centery = my
    
            # calculate the coords of foci 
            c = math.sqrt(rad_max*rad_max-rad_min*rad_min)
            focax = centerx + (c * np.cos(theta)); focay = centery + (c * np.sin(theta))
            focbx = centerx - (c * np.cos(theta)); focby = centery - (c * np.sin(theta))
    
            # bring them in the way cdist likes
            fixa = np.tile([focax,focay],(some_xvs.shape[0],1))
            fixb = np.tile([focbx,focby],(some_xvs.shape[0],1))
            # calc distances of chosen points from two foci
            allda = cdist(fixa,some_all,'euclidean')
            alldb = cdist(fixb,some_all,'euclidean')
            # add them and take residual from what holds for elipsis points
            alldr = (allda[0,:] + alldb[0,:]) - (2 * rad_max)
            # square the residuals 
            alldr2 = alldr * alldr #np does it pointwise
            # and take their mean, this is the performance critirio
            ssq = np.mean(alldr2)
            # if is better from what is now, it becomes the best 
       
            self.bestEL_ssq = ssq
            self.bestxEL = centerx
            self.bestyEL = centery
            self.bestrminEL = rad_min
            self.bestrmaxEL = rad_max
            self.bestthetaEL = theta
            self.best_noofpointsEL = len(self.chosen_ix)
        else:
            ssq = self.bestEL_ssq
            centerx = self.bestxEL
            centery = self.bestyEL
            rad_min = self.bestrminEL
            rad_max = self.bestrmaxEL
            theta = self.bestthetaEL
            if self.bestEL_ssq < terminationcritirio:
                print('Already optimized!!')
                notdone = 0
     
            #angles = np.linspace(0,2*np.pi,360)
            #and the ellipse is drawn
        self.draw_ellipse_and_canvas((300, 1, 1), centerx, centery, rad_max, rad_min, theta)           

        counter = 0 
        foundbetter = 0 
        stepsizes = np.array([2,2,1,1,1])
        while notdone:
            counter += 1
            # stop anyways on max iterations
            if counter > maxiterations:
                print('Exiting from max it')
                notdone = 0
            # calculate new direction only if we didnt improve
            if foundbetter == 0:
                rands1 = np.random.randint(0,2,5)
                rands1[rands1==0] = -1
                #rands = np.random.rand(3)
                dss = stepsizes * rands1
            centerx += dss[0]; centery += dss[1];
            #make sure big axis is indeed bigger after the updates
            rad_max = np.max([rad_max + dss[2], abs(dss[2])]);
            rad_min = np.max([rad_min + dss[3], abs(dss[3])]);# yes it is np.max also to keep it >0
            theta += dss[4]
            t1 = np.max([rad_min, rad_max]); t2 = np.min([rad_min, rad_max]);
            rad_max = t1; rad_min = t2;

            #print(rad_max, rad_min)
            #calculate new foci
            c = math.sqrt(rad_max*rad_max-rad_min*rad_min)
            focax = centerx + (c * np.cos(theta)); focay = centery + (c * np.sin(theta))
            focbx = centerx - (c * np.cos(theta)); focby = centery - (c * np.sin(theta))

            # and distances of chosen points from foci as before
            fixa = np.tile([focax,focay],(some_xvs.shape[0],1))
            fixb = np.tile([focbx,focby],(some_xvs.shape[0],1))
            allda = cdist(fixa,some_all,'euclidean')
            alldb = cdist(fixb,some_all,'euclidean')
            alldr = (allda[0,:] + alldb[0,:]) - (2 * rad_max)
            #alldr = abs(alld[0,:]-rad)
            alldr2 = alldr * alldr #np does it pointwise
            ssq = np.mean(alldr2)
            
            # ternimate if small enough error- it does happen
            if ssq<terminationcritirio:
                notdone = 0
                print('Exiting from optimum')
                
            
            if ssq<self.bestEL_ssq:
                self.bestEL_ssq = ssq
                self.bestxEL = centerx
                self.bestyEL = centery
                self.bestrminEL = rad_min
                self.bestrmaxEL = rad_max
                self.bestthetaEL = theta
                foundbetter = 1
            else:
                foundbetter = 0
            # if better ellipse found draw it          
            if foundbetter == 1:
                print('Drawing New')
                self.draw_ellipse_and_canvas((50, 1, 1), centerx, centery, rad_max, rad_min, theta)   

    def draw_ellipse_and_canvas(self, color, centerx, centery, rad_max, rad_min, theta):
        self.canvas.clear()
        self.bootit()
        color = (50, 1, 1) #red
        with self.canvas:
            for i in self.chosen_ix:
                Color(*color, mode='hsv')
                Rectangle(pos=(FitEllipseApp().xv1[i] - FitEllipseApp().rectsize / 2, FitEllipseApp().yv1[i] - FitEllipseApp().rectsize / 2), size=(FitEllipseApp().rectsize, FitEllipseApp().rectsize))
        
        angles = np.linspace(0,2*np.pi,360)
        circ_x = centerx + rad_max*np.cos(angles)*np.cos(theta) - rad_min*np.sin(angles)*np.sin(theta)
        circ_y = centery + rad_max*np.cos(angles)*np.sin(theta) + rad_min*np.sin(angles)*np.cos(theta)
        color = (300, 1, 1) #blue
        with self.canvas:
            Color(*color, mode='hsv')
            a = Line(points=(circ_x[0], circ_y[0]))
        for i in range(1, len(circ_x)):
            a.points += [circ_x[i], circ_y[i]]        
        del circ_x, circ_y


class FitEllipseApp(App):
    """Fits in least square sense an ellipse to chosen points on the grid via a controled random walk"""
    wsize = Window.size
    nx, ny = (20, 20)
    rectsize = 10.
    x = np.linspace(rectsize/2, wsize[0], nx)
    y = np.linspace(rectsize/2, wsize[1], ny)
    xv, yv = np.meshgrid(x, y)
    xv1 = np.ndarray.flatten(xv,'F')
    yv1 = np.ndarray.flatten(yv,'F')
    all = np.stack((xv1,yv1), axis =1)
    buttonw = 80
    buttonh = 30
    noofbuttons = 3
    leftpad = rightpad = buttonw/2
    L = list(np.linspace(leftpad, wsize[0]-rightpad-buttonw, noofbuttons, dtype='int'))
    buttons_x = [int(x) for x in L]
    
    def build(self):
        parent = Widget()
        self.painter = MyPaintWidget()
       
        # fitbtn = Button(text='Fit Circle D', pos =(self.buttons_x[0], 0), size =(self.buttonw, self.buttonh))
        # fitbtn0 = Button(text='Fit Circle R', pos =(self.buttons_x[1], 0), size =(self.buttonw, self.buttonh))
        fitbtn1 = Button(text='Fit Ellipse', pos =(self.buttons_x[0], 0), size =(self.buttonw, self.buttonh))
        clearbtn = Button(text='Clear', pos =(self.buttons_x[1], 0), size =(self.buttonw, self.buttonh))
        fitbtn2 = Button(text='Exit', pos =(self.buttons_x[2], 0), size =(self.buttonw, self.buttonh))
        clearbtn.bind(on_press=self.clear_canvas)
        fitbtn1.bind(on_press=self.fit_ellipse)
        fitbtn2.bind(on_press=self.stop_app)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(fitbtn1)
        parent.add_widget(fitbtn2)
        self.painter.bootit()
        return parent


    def stop_app(self, obj):
        self.root_window.close()   
        FitEllipseApp().stop()
         
    def fit_ellipse(self, obj):
        self.painter.fitellipse()
    
    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.painter.bootit()
        self.painter.fx = 0
        self.painter.fy = 0
        self.painter.chosen_ix = []
#        self.painter.best_ssq = 1000000
#        self.painter.bestEL_ssq = 1000000
        self.painter.bestr = 0
        self.painter.bestrmaxEL = 0
        


if __name__ == '__main__':
    FitEllipseApp().run()
