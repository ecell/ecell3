#!/usr/bin/env python

import gtk
from Numeric import *
import gtk.gdk
import re
import string
import operator
from Plot import *
from ecell.ecssupport import *
#LoggerMinimumInterval=1
import ConfirmWindow
from OsogoPluginWindow import *

COL_LOG=2
COL_PIX=1
COL_ON=0
COL_TXT=3
_SMALL_PLOTWIDTH = 400
_SMALL_PLOTHEIGTH = 250
_LARGE_PLOTWIDTH = 600
_LARGE_PLOTHEIGTH = 350


class TracerWindow( OsogoPluginWindow ):



	def __init__( self, dirname, data, pluginmanager, root=None ):
		#initializa variables:
		#initiates OsogoPluginWindow
		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
		self.thePluginManager=pluginmanager
		self.displayedFullPNStringList=[]		
		self.theLoggerMap={}
		self.no_expose=False
		#get session
		self.theSession = pluginmanager.theSession
		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		if operator.isNumberType( aValue ) == FALSE:
			aMessage = "Error: (%s) is not numerical data" %aFullPNString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(0,aMessage,'Error!')
			raise TypeError( aMessage )
		self.theViewType = MULTIPLE
		self.min_button_clicked=False


	def openWindow(self):
		OsogoPluginWindow.openWindow(self)

		#self.openWindow()
		self.ListWindow=self['clist1']
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
		self.theScrolledWindow=self['scrolledwindow1']
		self.theFixed=self['fixed1']
		self.ListSelection=self.ListWindow.get_selection()
		self.ListSelection.set_mode(gtk.SELECTION_MULTIPLE)
		self.theWindow=self.getWidget(self.__class__.__name__)
		#determine plotsize
		if self.isStandAlone():
			self.desired_plot_size = [ _LARGE_PLOTWIDTH, _LARGE_PLOTHEIGTH] 
			
		else:
			self.desired_plot_size = [ _SMALL_PLOTWIDTH, _SMALL_PLOTHEIGTH] 

		self.thePlotHeigth = self.desired_plot_size[1]
		self.thePlotWidth = self.desired_plot_size[0]
		#init plotter instance
		self.thePlotInstance=Plot(self, 'linear', self.getParent(), self.thePlotWidth, self.thePlotHeigth )
		#attach plotterwidget to window
		self.PlotWidget= self.thePlotInstance.getWidget() 
		aWindowWidget = self.getWidget( 'frame8' )
		aWindowWidget.add( self.PlotWidget )
		aWindowWidget.show_all()
		self.thePluginManager.appendInstance( self )                    

		#add handlers to buttons
		self.addHandlers({\
		    'on_button9_clicked' : self.remove_trace,\
		    'on_button12_clicked'  : self.change_scale,\
		    'on___minimize_clicked'  : self.__minimize_clicked})
		self.ListWindow.connect("button-press-event",self.button_pressed_on_list)
		self['button12'].set_label('Log10 Scale')

		#addtrace to plot
		self.addtrace_to_plot(self.theFullPNList())
		self.refresh_loggers()

		#sets stripinterval, disable history buttons
		self['entry1'].set_text(str(self.thePlotInstance.getstripinterval()))
		self['entry1'].connect('activate',self.entry1_activated)
		self['entry1'].connect('focus_out_event',self.stripinterval_changes)
		self['button13'].connect('clicked',self.__createlogger_pressed)
		self['togglebutton3'].connect('toggled',self.__togglestrip)
		self.lastTime=self.theSession.theSimulator.getCurrentTime()
		if not self.isStandAlone():
			self.minimize()
		else:
			self['TracerWindow'].connect('expose-event',self.__resize)



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
		
	def create_logger(self,fpnlist):
		LoggerMinimumInterval=float(self.theSession.getParameter('logger_min_interval'))
		for fpn in fpnlist:
		    if not self.haslogger(fpn):
			try:
			    self.theSession.theSimulator.createLogger(fpn)
			except:
			    self.theSession.message('Error while creating logger\n logger for '+ fpn + ' not created\n')
			else:
			    self.theSession.theSimulator.setLoggerMinimumInterval(fpn,LoggerMinimumInterval)
			    self.theSession.message("Logger created for "+fpn)
		self.check_history_button()
		self.thePluginManager.updateFundamentalWindows()
			
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

	def check_history_button(self):
	    history_button=self['togglebutton3']
	    if len(self.displayedFullPNStringList)==0:
		history_button.set_sensitive(gtk.FALSE)
		return None	
	    for fpn in self.displayedFullPNStringList:
		if not self.haslogger(fpn):
			history_button.set_sensitive(gtk.FALSE)
			return None	
	    history_button.set_sensitive(gtk.TRUE)      
		
	def getloggerstart(self,aFullPNString):
		#called from the plotinstance
		lstart=self.theSession.theSimulator.getLoggerStartTime(aFullPNString)
		return lstart
		
	def getloggerend(self, aFullPNString):
		#called from the plotinstance
		lend=self.theSession.theSimulator.getLoggerEndTime(aFullPNString)
		return lend 
		

	def appendRawFullPNList( self, aRawFullPNList ):
		"""overwrites superclass method
		aRawFullPNList  -- a RawFullPNList to append (RawFullPNList) 
		Returns None
		"""

		# calls superclass's method
		OsogoPluginWindow.appendRawFullPNList( self, aRawFullPNList )
		# creates FullPNList to plot
		aFullPNList = map( self.supplementFullPN, aRawFullPNList )

		for aFullPN in aFullPNList:
			aFullPNString = createFullPNString( aFullPN )
			aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
			if operator.isNumberType( aValue ) == FALSE:
				aMessage = "Error: (%s) is not numerical data" %aFullPNString
				self.thePluginManager.printMessage( aMessage )
				aDialog = ConfirmWindow(0,aMessage,'Error!')
				raise TypeError( aMessage )

		# appends FullPNList as plot data
		self.addtrace_to_plot( aFullPNList )

			
		
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



	def addtrace_to_plot(self,aFullPNList):
		#checks that newpn has logger if mode is history
		#calls superclass
		pass_flag=0

		if self.thePlotInstance.getstripmode()=='history':
		    for aFullPN in aFullPNList:
			aFullPNString= createFullPNString( aFullPN)
			if not self.haslogger(aFullPNString):
			    self.theSession.message(aFullPNString+" doesn't have associated logger.")
			    pass_flag=1    		
		if pass_flag==1: 
			return -1

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
				pass_list.append([aFullPNString,aValue, aFullPN])
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
	    
	def set_scale_button(self,text):
	    self['button12'].set_label(text)


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
	    
	def remove_trace_from_list(self,aFullPNString):
		pass

	def shrink_to_fit(self):
		self['TracerWindow'].resize(self.desired_window_size[0],
				self.desired_window_size[1])

	
	
	def maximize(self):
		self['vbox1'].add(self['scrolledwindow1'])
		self['vbox2'].add(self['fixed1'])
		alloc_rect2=self['fixed1'].size_request()
		alloc_rect3=self['vbox1'].size_request()
		aHeigth= _LARGE_PLOTHEIGTH + alloc_rect2[1] + alloc_rect3[1] +4
		aWidth = max(_LARGE_PLOTWIDTH + 4, alloc_rect2[0], alloc_rect3[0]) 
		self.desired_plot_size= [_LARGE_PLOTWIDTH, _LARGE_PLOTHEIGTH]
		top_frame_size = [ aWidth, aHeigth ]
		if not self.isStandAlone():
			plot_size_alloc=[ _LARGE_PLOTWIDTH, _LARGE_PLOTHEIGTH]
		else:
			plot_size_alloc = self.__get_frame8_space()
			self.desired_window_size= [ max( plot_size_alloc[0], alloc_rect2[0], alloc_rect3[0]),\
				plot_size_alloc[1] + alloc_rect2[1] + alloc_rect3[1]]
		self['top_frame'].set_size_request(top_frame_size[0],top_frame_size[1])
		self.thePlotInstance.resize(plot_size_alloc)
		self.getParent().shrink_to_fit()


	def minimize(self):
		self.thePlotInstance.minimize()
		if not self.isStandAlone() or self.min_button_clicked:
			self.desired_plot_size = [_SMALL_PLOTWIDTH, _SMALL_PLOTHEIGTH]
			plot_size_alloc = self.desired_plot_size[:]
		else:
			self.desired_plot_size = [_LARGE_PLOTWIDTH, _LARGE_PLOTHEIGTH]
			plot_size_alloc = self.__get_frame8_space()
		self.desired_window_size = [plot_size_alloc[0] + 4, plot_size_alloc[1] + 4]
		self['vbox1'].remove(self['scrolledwindow1'])
		self['vbox2'].remove(self['fixed1'])
		self['top_frame'].set_size_request(self.desired_plot_size[0]+4,
			self.desired_plot_size[1]+4)
		self.thePlotInstance.resize( plot_size_alloc )
		self.getParent().shrink_to_fit()
		self.thePlotInstance.printTraceLabels()


	def __get_frame8_space (self):
		aSizeAlloc=self['frame8'].get_allocation()
		return_width = aSizeAlloc[2] - 4
		return_heigth = aSizeAlloc[3] - 4
		if return_width < self.desired_plot_size[0]:
			return_width = self.desired_plot_size[0]
		if return_heigth < self.desired_plot_size[1]:
			return_heigth = self.desired_plot_size[1]
		return [ return_width, return_heigth ]

			
	# ========================================================================
	def resize( self, width, heigth ):
		"""resizes this window according to width and heigth.
		Returns None
		"""
		self[self.__class__.__name__].resize( width, heigth)
		self.__resize(None, None)

		
#----------------------------------------------
#SIGNAL HANDLERS
#-------------------------------------------------


	
	#----------------------------------------------
	#this signal handler is called when ENTER is pushed on entry1
	#-------------------------------------------------
	
	def entry1_activated(self,obj):
	    self.stripinterval_changes(obj,None)
	
	#--------------------------------------------------------
	#this signal handler is called when TAB is presses on entry1
	#---------------------------------------------------------
		
	def stripinterval_changes(self, obj, event): #this is an event handler again
		#get new value
		#call plotterinstance
		try:
		    a=float(self['entry1'].get_text())
		except ValueError:
		    self.theSession.message("Enter a valid number, please.")
		    self['entry1'].set_text(str(self.thePlotInstance.getstripinterval()))
		else:
		    self.thePlotInstance.setstripinterval(a)


	#--------------------------------------------------------
	#this signal handler is called when mousebutton is pressed over the fullpnlist
	#---------------------------------------------------------

	def button_pressed_on_list(self, aWidget, anEvent):
	    if anEvent.button==3:
		self.change_trace_color()

	#--------------------------------------------------------
	#this signal handler is called when an on-off checkbox is pressed over of the fullpnlist
	#---------------------------------------------------------
		
	def trace_toggled(self,cell, path, model):
	    iter=model.get_iter((int (path),))
	    text=self.ListStore.get_value(iter,COL_TXT)
	    onoff=self.thePlotInstance.toggle_trace(text)
	    self.ListStore.set_value(iter,COL_ON,onoff)
	    
	#--------------------------------------------------------
	#this signal handler is called when create logger checkbox is pressed over the fullpnlist
	#---------------------------------------------------------

	def toggle_pressed(self,cell,path,model):
	    iter=model.get_iter((int (path),))
	    fixed=model.get_value(iter,COL_LOG)
	    text=self.ListStore.get_value(iter,COL_TXT)
	    
	    if fixed==gtk.FALSE:
		self.create_logger([text])
		self.refresh_loggers()	
	
	#--------------------------------------------------------
	#this signal handler is called when "linear scale" or "log10 scale" button is pressed
	#---------------------------------------------------------
   
	def change_scale(self,obj = None):  #this is a buttonhandler
	    #simply calls plotinstance.change_scale
	    self.thePlotInstance.change_scale()
	
	#--------------------------------------------------------
	#this signal handler is called when "Remove Trace" button is pressed
	#---------------------------------------------------------

	def remove_trace(self, obj = None):
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

	#--------------------------------------------------------
	#this signal handler is called when "Log All" button is pressed
	#---------------------------------------------------------

	def __createlogger_pressed(self,obj):
	    #creates logger in simulator for all FullPNs 
	    self.create_logger(self.displayedFullPNStringList)		
	    self.refresh_loggers()
	    
	#--------------------------------------------------------
	#this signal handler is called when "Show History" button is toggled
	#---------------------------------------------------------
		
	def __togglestrip(self, obj):
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

	#--------------------------------------------------------
	#this signal handler is called when "Minimize" button is pressed
	#--------------------------------------------------------

	def __minimize_clicked(self,button_obj):
		self.min_button_clicked=True
		self.minimize()
		self.min_button_clicked=False

	#--------------------------------------------------------
	#this signal handler is called when TracerWindow is __resized
	#--------------------------------------------------------

	def __resize(self, window, event):
		self.thePlotInstance.resize( self.__get_frame8_space() )
		return gtk.FALSE

