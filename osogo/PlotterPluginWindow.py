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
import ConfirmWindow

COL_LOG=2
COL_PIX=1
COL_ON=0
COL_TXT=3
class PlotterPluginWindow( OsogoPluginWindow ):
	#
	#initiates plotter sends data for processing to superclass
	#
	def __init__( self, dirname, data, pluginmanager, plot_type, root=None ):
		PlotterPluginWindow.theViewType = MULTIPLE
		#PluginWindow.__init__( self, dirname, data, pluginmanager, root )
		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
		self.thePluginManager=pluginmanager
		self.thePlotType = plot_type
		#get session
		self.theSession = pluginmanager.theSession

		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		if operator.isNumberType( aValue ) == FALSE:
			aMessage = "Error: (%s) is not numerical data" %aFullPNString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow.ConfirmWindow(0,aMessage,'Error!')
			raise TypeError( aMessage )
	
	def openWindow(self):
		OsogoPluginWindow.openWindow(self)

		#self.openWindow()
		aWindowWidget = self.getWidget( 'frame8' )
		self.displayedFullPNStringList=[]
		
		self.ListWindow=self.getWidget('clist1')
		self.ListStore=gtk.ListStore(gobject.TYPE_BOOLEAN,\
		    gobject.TYPE_OBJECT, gobject.TYPE_BOOLEAN,\
		    gobject.TYPE_STRING)
		self.ListWindow.set_model(self.ListStore)
		renderer4=gtk.CellRendererToggle()
		renderer4.connect('toggled',self.trace_toggled,self.ListStore)
		renderer=gtk.CellRendererToggle()
		renderer.connect('toggled',self.toggle_pressed,self.ListStore)
		renderer2=gtk.CellRendererPixbuf()
		column1=gtk.TreeViewColumn('color',renderer2,pixbuf=COL_PIX)
		column2=gtk.TreeViewColumn('trace',gtk.CellRendererText(),text=COL_TXT)
		column3=gtk.TreeViewColumn('lg',renderer,active=COL_LOG)
		column4=gtk.TreeViewColumn('on',renderer4,active=COL_ON)
		column1.set_resizable(gtk.TRUE)
		column2.set_resizable(gtk.TRUE)
		column3.set_resizable(gtk.TRUE)
		column4.set_resizable(gtk.TRUE)
		
		self.ListWindow.append_column(column4)
		self.ListWindow.append_column(column1)
		self.ListWindow.append_column(column3)
		self.ListWindow.append_column(column2)
		self.ListSelection=self.ListWindow.get_selection()
		self.ListSelection.set_mode(gtk.SELECTION_MULTIPLE)
		self.theWindow=self.getWidget(self.__class__.__name__)
		#init plotter instance
		#if plot_type=='TracerPlot':
		if self.thePlotType=='TracerPlot':
		    #self.thePlotInstance=TracerPlot(self, 'linear',self.theWindow)
		    self.thePlotInstance=TracerPlot(self, 'linear', self.getParent() )
		#attach plotterwidget to window
		self.PlotWidget= self.thePlotInstance.getWidget() 
		aWindowWidget.add( self.PlotWidget )

		aWindowWidget.show_all()

		self.thePluginManager.appendInstance( self )                    

		#add handlers to buttons
		self.addHandlers({\
		    'on_button9_clicked' : self.remove_trace,\
		    'on_button12_clicked'  : self.change_scale})

		self.ListWindow.connect("button-press-event",self.button_pressed_on_list)

		self['button12'].set_label('Log10 Scale')

		#init clist
	 			
		##get session
		#self.theSession = pluginmanager.theSession
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
	    self.check_history_button()
	    self.check_remove_button()	    
	    
	def getlatestdata(self,fpn):
	    value=self.theSession.theSimulator.getEntityProperty(fpn)
	    time=self.theSession.theSimulator.getCurrentTime()
	    return [time,value]
	    
	def remove_trace(self, obj):
	    #identify selected FullPNs
	    fpnlist=[]	    
	    selected_list=self.getselected()
	    for aselected in selected_list:
    		#remove from fullpnlist
		if len(self.displayedFullPNStringList)==1:
		    break
		FullPNList=self.theRawFullPNList[:]
		for afullpn in FullPNList:
		    if aselected[0]==createFullPNString( afullpn):
			self.theRawFullPNList.remove(afullpn)
			break	    
	    #remove from displaylist
		self.displayedFullPNStringList.remove(aselected[0])
		fpnlist.append(aselected[0])
		self.ListStore.remove(aselected[1])	    
	    #remove from plotinstance
		
	    self.thePlotInstance.remove_trace(fpnlist)
	    #delete selected from list
	    self.check_history_button()
	    self.check_remove_button()

	def check_remove_button(self):
	    remove_button=self['button9']
	    if len(self.displayedFullPNStringList)>1:
		remove_button.set_sensitive(gtk.TRUE)
	    else:
		remove_button.set_sensitive(gtk.FALSE)


	def getselected(self):
	    self.selection_list=[]
	    self.ListSelection.selected_foreach(self.selection_function)
	    return self.selection_list
	    #se;ection list, [0]=text, [1], iter
	    
	def selection_function(self,model,path,iter):
		text=self.ListStore.get_value(iter,COL_TXT)
		self.selection_list.append([text,iter])
	    
	#def addtrace(self, aFullPN): #called from outside
	    #add to FullPNlist
	    #addtrace_to_plot
		
	    
	def change_scale(self,obj):  #this is a buttonhandler
	    #simply calls plotinstance.change_scale
	    self.thePlotInstance.change_scale()
	
	def set_scale_button(self,text):
	    self['button12'].set_label(text)
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

	def change_trace_color(self):
	    selected_list=self.getselected()
	    if len(selected_list)>0:
		fpn=selected_list[0][0]
		iter=selected_list[0][1]
		pixbuf=self.thePlotInstance.change_trace_color(fpn)
		self.ListStore.set_value(iter,COL_PIX,pixbuf)
	
	def button_pressed_on_list(self, aWidget, anEvent):
	    if anEvent.button==3:
		self.change_trace_color()
	    
	    
	def remove_trace_from_list(self,aFullPNString):
		pass
	    
