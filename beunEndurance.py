#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:23:03 2021

@author: tblaha
"""
from numpy import random as rng
rng.seed(0)

class endurance_profile():
    
    def __init__(self):
        self.state  = 0
        self.Iavg = 15 # Amp
        self.DTacc   = 3  # sec
        self.DTdec   = 1
        self.DTcoast = 1
        self.Iacc = 65
        self.Idec = ( - self.Iacc * self.DTacc + self.Iavg * (self.DTacc+self.DTdec+self.DTcoast)) / self.DTdec
        self.t_in_DT = 0
        self.DT = 1
        self.I    = 0
        self.t_prev = 0
        
        self.Ttrack       = 1400
        self.Tdrivechange = 200
    
    def genNewSegment(self):
        if self.state == 1:
            self.state = -1
        elif self.state == -1:
            self.state = 0
        elif self.state == 0:
            self.state = 1
        
        self.t_in_DT = 0
        
        if self.state == 1:
            self.DT = max(self.DTacc + 2*rng.random(), 0)
            self.I  = max(self.Iacc + 25*rng.random(), 0)
        elif self.state == 0:
            self.DT = max(self.DTcoast + 0.5*rng.random(), 0)
            self.I  = 0
        elif self.state == -1:
            self.DT = max(self.DTdec + 0.5*rng.random(), 0)
            self.I  = min(self.Idec + 25*rng.random(), 0)
        
    def getI(self, t):
        
        if t > 0 and (t < 700 or t > 900) and t < 1600:
            self.t_in_DT += t - self.t_prev
            if self.t_in_DT > self.DT:
                self.genNewSegment()
            I = self.I # ampere
        else:
            I = 0
        
        self.t_prev = t
        
        return I