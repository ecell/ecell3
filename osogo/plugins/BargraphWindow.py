#!/usr/bin/env python

from gtk import *
from Numeric import *
import gtk.gdk
import re
import string
import operator
from PlotterPluginWindow import *
from Plot import *
from ecell.ecssupport import *

class BargraphWindow( PlotterPluginWindow ):

	def __init__( self, dirname, data, pluginmanager, root=None ):
		#initializa variables:
		#initiates Plotterpluginwindow
		PlotterPluginWindow.__init__(self, dirname, data, pluginmanager,\
					    'BarPlot',root)

	def update(self):
		values={}
		for fpn in self.displayedFullPNStringList:
		    values[fpn]=self.getlatestdata(fpn)[1]    
		self.thePlotInstance.addpoints(values)
		return True
		
		
	def addtrace_to_plot(self,aFullPNList):
		#calls superclass

		PlotterPluginWindow.addtrace_to_plot(self,aFullPNList)    
		
