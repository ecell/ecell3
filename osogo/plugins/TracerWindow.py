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

class TracerWindow( PlotterPluginWindow ):

	def __init__( self, dirname, data, pluginmanager, root=None ):
		#initializa variables:
		#initiates Plotterpluginwindow
		PlotterPluginWindow.__init__(self, dirname, data, pluginmanager,\
					    'TracerPlot',root)
		#sets stripinterval, disable history buttons
		self['entry1'].set_text(str(self.thePlotInstance.getstripinterval()))
		#sets additional button handlers(toggle strip/history, zoom), 
		self.addHandlers({'on_entry1_focus_out_event' :  self.stripinterval_changes , 
		'on_button13_clicked'    :self.createlogger_pressed,  
		'on_togglebutton3_toggled' :self.togglestrip })
		self.theLoggerMap={}
		self.lastTime=self.theSession.theSimulator.getCurrentTime()

	def update(self):
		#depending on whether it is in strip mode or history mode
		#strip mode: gets last values from simulator for each fullPN  
		#and calls theplotinstance.addpoint(string,x:y values)
		#history:  calls
		#theplotinstance.addpoint
		
		aTime=self.theSession.theSimulator.getCurrentTime()
		if aTime-self.lastTime>=self.thePlotInstance.pixelwidth:
		    values={}
		    for fpn in self.displayedFullPNStringList:
			values[fpn]=self.getlatestdata(fpn)
		    self.thePlotInstance.addpoints(values)
		    self.lastTime=aTime
		return True
		
	def togglestrip(self, obj):
		#if history, change to strip, try to get data for strip interval
		stripmode=self.thePlotInstance.getstripmode()
		if stripmode=='history':
		    #get latest data
		    values={}
		    for fpn in self.displayedFullPNStringList:
			values[fpn]=self.getlatestdata(fpn)
		    self.thePlotInstance.setmode_strip(values)
		else:
		    pass_flag=True
		    for fpn in self.displayedFullPNStringList:
			if not self.haslogger(fpn): pass_flag=False
		    if pass_flag:
			self.thePlotInstance.setmode_history()
		    else:
			self.theSession.message("can't change to history mode, because not every trace has logger.\n")
			obj.set_active(0)
			
		# can only issue plotinstance.setmode('history') if all fullPN is logger	
		

	def createlogger_pressed(self,obj):
		#creates logger in simulator for a FullPNString, if 
		selected_list=self.getselected()
		for fpn in selected_list:
		    if not self.haslogger(fpn[0]):
			self.theSession.theSimulator.createLogger(fpn[0])
		
	def recache(self, aFullPNString, value_from, value_to, interval):
		#it is called from the plotinstance
		#returns as much data as it can
		return_list=[]
		if value_from<value_to:
		    a=self.theSession.theSimulator.getLoggerData(aFullPNString,
					value_from,
					value_to,interval)
		    for t in range(size(a,0)):
			return_list.append([a[t,0],a[t,1]])
		
		return return_list
			
	def haslogger(self, aFullPNString):
		#called from the plotinstance
		#to be implemeted, needs to call LoggerWindow or new
		#function in the Simulator
		loggerlist=self.theSession.theSimulator.getLoggerList()
		return loggerlist.__contains__(aFullPNString)
		
	def getloggerstart(self,aFullPNString):
		#called from the plotinstance
		return self.theSession.theSimulator.getLoggerStartTime(aFullPNString)
		
	def getloggerend(self, aFullPNString):
		#called from the plotinstance
		return self.theSession.theSimulator.getLoggerEndTime(aFullPNString)
		
	def addtrace_to_plot(self,aFullPNList):
		#checks that newpn has logger if mode is history
		#calls superclass
		pass_flag=0

		if self.thePlotInstance.getstripmode()=='history':
		    for aFullPN in aFullPNList:
			aFullPNString= createFullPNString( aFullPN)
			if not self.haslogger(aFullPNString):
			    self.printMessage(aFullPNString, " doesn't have associated logger.")
			    pass_flag=1    		
		if pass_flag==0: 
		    PlotterPluginWindow.addtrace_to_plot(self,aFullPNList)    
		
	def stripinterval_changes(self, obj, event): #this is an event handler again
		#get new value
		#call plotterinstance
		try:
		    a=float(self['entry1'].get_text())
		except ValueError:
		    self.theSession.message("Enter a valid number, please.")
		    self['entry1'].set_text(str(self.thePlotInstance.getstripinterval()))
		else:
		    if a>1:
			self.thePlotInstance.setstripinterval(int(a))
		    else:
			self.theSession.message("Enter a valid number, please.")
			self['entry1'].set_text(str(self.thePlotInstance.getstripinterval()))
			
		
##
##
## Secondary functions
##
##

