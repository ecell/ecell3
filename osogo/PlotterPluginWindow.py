#!/usr/bin/env python

import gtk
from Numeric import *
 
import gtk.gdk
import re
import string
import operator
from OsogoPluginWindow import *
import gobject
from ecell.ecssupport import *
from Plot import *
COL_LOG=0
COL_PIX=1
COL_ON=2
COL_TXT=3
class PlotterPluginWindow( OsogoPluginWindow ):
	#
	#initiates plotter sends data for processing to superclass
	#
	def __init__( self, dirname, data, pluginmanager, plot_type, root=None ):
		#PluginWindow.__init__( self, dirname, data, pluginmanager, root )
		self.thePluginManager=pluginmanager
		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
		self.openWindow()
		aWindowWidget = self.getWidget( 'frame8' )
		self.displayedFullPNStringList=[]
		
		self.ListWindow=self.getWidget('clist1')
		self.ListStore=gtk.ListStore(gobject.TYPE_BOOLEAN,\
		    gobject.TYPE_OBJECT, gobject.TYPE_BOOLEAN,\
		    gobject.TYPE_STRING)
		self.ListWindow.set_model(self.ListStore)
		renderer=gtk.CellRendererToggle()
		renderer.connect('toggled',self.toggle_pressed,self.ListStore)
		renderer2=gtk.CellRendererPixbuf()
		renderer4=gtk.CellRendererToggle()
		renderer4.connect('toggled',self.trace_toggled,self.ListStore)
		column1=gtk.TreeViewColumn('color',renderer2,pixbuf=COL_PIX)
		column4=gtk.TreeViewColumn('on',renderer4,active=COL_ON)
		column3=gtk.TreeViewColumn('lg',renderer,active=COL_LOG)
		column2=gtk.TreeViewColumn('trace',gtk.CellRendererText(),text=COL_TXT)
		self.ListWindow.append_column(column3)
		self.ListWindow.append_column(column1)
		self.ListWindow.append_column(column2)
		self.ListWindow.append_column(column4)
		self.ListSelection=self.ListWindow.get_selection()
		self.ListSelection.set_mode(gtk.SELECTION_MULTIPLE)
		self.theWindow=self.getWidget(self.__class__.__name__)
		#init plotter instance
		if plot_type=='TracerPlot':
		    self.thePlotInstance=TracerPlot(self, 'linear',self.theWindow)
		#attach plotterwidget to window
		self.PlotWidget= self.thePlotInstance.getWidget() 
		aWindowWidget.add( self.PlotWidget )

		#add handlers to buttons
		self.addHandlers({\
		    'on_button9_clicked' : self.remove_trace,\
		    'on_button12_clicked'  : self.change_scale})
		aWindowWidget.show_all()

		self.thePluginManager.appendInstance( self )                    
		#init clist
	 			
		#get session
		self.theSession = pluginmanager.theSession
		#addtrace to plot
		self.addtrace_to_plot(self.theFullPNList())
		self.refresh_loggers()

	def refresh_loggers(self):
	    #refreshes loggerlist
	    iter=self.ListStore.get_iter_first()
	    while iter!=None:
		text=self.ListStore.get_value(iter,COL_TXT)
		if self.haslogger(text):
		    fixed=gtk.TRUE
		else:
		    fixed=gtk.FALSE
		self.ListStore.set(iter,COL_LOG,fixed)
		iter=self.ListStore.iter_next(iter)
		
	def trace_toggled(self,cell, path, model):
#	    iter=model.get_iter((int (path),))
#	    text=self.ListStore.get_value(iter,2)
	    iter=model.get_iter((int (path),))
#	    onoff=model.get_value(iter,COL_ON)
	    text=self.ListStore.get_value(iter,COL_TXT)
	    onoff=self.thePlotInstance.toggle_trace(text)
	    self.ListStore.set_value(iter,COL_ON,onoff)
	    
	def toggle_pressed(self,cell,path,model):
	    iter=model.get_iter((int (path),))
	    fixed=model.get_value(iter,COL_LOG)
	    text=self.ListStore.get_value(iter,COL_TXT)
	    
	    if fixed==gtk.FALSE:
		self.create_logger([text])
		self.refresh_loggers()	
	
	def addtrace_to_plot(self,aFullPNList): #make possible to add multiple
	    pass_list=[]
	    for aFullPN in aFullPNList: 
		aFullPNString = createFullPNString( aFullPN )
		#gets most recent value
		#check whether there's enough room left
		if len(self.displayedFullPNStringList)<self.thePlotInstance.getmaxtraces():
		#adds trace to plotinstance,clist, add to displaylist, colorlist
		    aValue = self.getlatestdata( aFullPNString )
		    if operator.isNumberType( aValue[1] ):
			self.displayedFullPNStringList.append(aFullPNString)
			pass_list.append([aFullPNString,aValue])
		    else:
			self.theSession.message('%s cannot be displayed, because it is not numeric\n' % aFullPNString)
	    added_list=self.thePlotInstance.addtrace(pass_list)
	    
	    self.add_trace_to_list(added_list)
	
	def getlatestdata(self,fpn):
	    value=self.theSession.theSimulator.getEntityProperty(fpn)
	    time=self.theSession.theSimulator.getCurrentTime()
	    return [time,value]
	    
	def remove_trace(self, obj):
	    #identify selected FullPNs
	    selected_list=self.getselected()
	    for aselected in selected_list:
    		#remove from fullpnlist
		FullPNList=self.theRawFullPNList[:]
		for afullpn in FullPNList:
		    if aselected[0]==createFullPNString( afullpn):
			self.theRawFullPNList.remove(afullpn)
			break
		    
		#remove from displaylist
		self.displayedFullPNStringList.remove(aselected[0])
	    #remove from plotinstance
	    fpnlist=[]
	    for aselected in selected_list:
		fpnlist.append(aselected[0])
	    self.thePlotInstance.remove_trace(fpnlist)
	    #delete selected from list
	    sel=self.getselected()
	    for selected in sel:
		    self.ListStore.remove(selected[1])	    
	
	def getselected(self):
	    self.selection_list=[]
	    self.ListSelection.selected_foreach(self.selection_function)
	    return self.selection_list
	    #se;ection list, [0]=text, [1], iter
	    
	def selection_function(self,model,path,iter):
		text=self.ListStore.get_value(iter,COL_TXT)
		self.selection_list.append([text,iter])
	    
	def addtrace(self, aFullPN): #called from outside
	    #add to FullPNlist
	    #addtrace_to_plot
	    print "addtrace" #dummy command
	    
	def change_scale(self,obj):  #this is a buttonhandler
	    #simply calls plotinstance.change_scale
	    self.thePlotInstance.change_scale()

##
##
##	Private functions
##
##



	def add_trace_to_list(self,added_list):
	    for added_item in added_list:
		iter=self.ListStore.append()
		self.ListStore.set_value(iter,COL_PIX,added_item[1]) #set pixbuf
		self.ListStore.set_value(iter,COL_TXT,added_item[0]) #set text
		self.ListStore.set_value(iter,COL_ON,gtk.TRUE) #trace is on by default
	    
	def remove_trace_from_list(self,aFullPNString):
		print "remove_trace_from_list"
	    
