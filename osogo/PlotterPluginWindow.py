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
		self.ListStore=gtk.ListStore(gobject.TYPE_OBJECT,gobject.TYPE_STRING)
		self.ListWindow.set_model(self.ListStore)
		column1=gtk.TreeViewColumn('color',gtk.CellRendererPixbuf(),pixbuf=0)
		column2=gtk.TreeViewColumn('trace',gtk.CellRendererText(),text=1)
		self.ListWindow.append_column(column1)
		self.ListWindow.append_column(column2)
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
		text=self.ListStore.get_value(iter,1)
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
		self.ListStore.set_value(iter,0,added_item[1]) #set pixbuf
		self.ListStore.set_value(iter,1,added_item[0]) #set pixbuf
	    
	def remove_trace_from_list(self,aFullPNString):
		print "remove_trace_from_list"
	    
