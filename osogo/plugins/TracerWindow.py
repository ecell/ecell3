#!/usr/bin/env python

from gtk import *
from gtkextra import *
#from Window import *
from Numeric import *
import GDK
import GTK
import re
import string
import operator
from OsogoPluginWindow import *

from ecell.ecssupport import *
#from testTracerglade import *
#from data1 import *

class TracerWindow( OsogoPluginWindow ):

	def __init__( self, dirname, data, pluginmanager, root=None ):


		#PluginWindow.__init__( self, dirname, data, pluginmanager, root )
		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
		IDflag = 1
		if len( self.theFullPNList() ) > 1:
			for aFullID in self.theFullIDList():
				if aFullID == self.theFullID():
					IDflag = 0
				else:
					IDflag = 1

        
		self.theFirstLog = []	

		self.theSession = pluginmanager.theSession
		aFullPNString = createFullPNString( self.theFullPN() )

		#print ' IDflag = %s' %IDflag


		if IDflag == 1:
			aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
			#if operator.isNumberType( aValue[0] ):
			if operator.isNumberType( aValue ):
				self.openWindow()
				self.thePluginManager.appendInstance( self )                    
				#self.initialize()
				# ------------------------------------------------------
				self.xaxis = None
				self.yaxis = None
				self.scale = "linear"
				self.arg = 0
				self.step_size = 0
				self.StartTime = 0
				self.theLoggerList = []
				self.theDataList = []
				self.xList = []
				self.yList = []
				self.FullDataList = []

				self['toolbar1'].set_style(GTK.TOOLBAR_ICONS)
				self.addHandlers( { 'on_button9_clicked' : self.popInputWindow,
				                    'checkbutton1_clicked' : self.changeScale } )
				                    #'window_exit' : self.exit,
        
				aWindowWidget = self.getWidget( 'frame8' )
				self.theColorMap = aWindowWidget.get_colormap()

				self.canvas = GtkPlotCanvas(450,300)
				self.canvas.set_background( self.theColorMap.alloc("light blue") )


				self.plot = GtkPlot(width=0.8, height=0.8)
				self.plot.set_background( self.theColorMap.alloc("white") )
				self.plot.set_yscale(PLOT_SCALE_LINEAR)
				self.plot.autoscale()
				self.plot.axis_set_ticks(PLOT_AXIS_X, 1, 1)
				self.plot.axis_set_visible(PLOT_AXIS_RIGHT, FALSE)
				self.plot.axis_set_visible(PLOT_AXIS_TOP, FALSE)
				self.plot.grids_set_visible(TRUE, FALSE, TRUE, FALSE)
				self.plot.major_hgrid_set_attributes( PLOT_LINE_SOLID, 1.0,
				                                     self.theColorMap.alloc("light blue") )
				self.plot.major_vgrid_set_attributes( PLOT_LINE_SOLID, 1.0,
				                                     self.theColorMap.alloc("light blue") )


				self.plot.axis_hide_title(PLOT_AXIS_TOP)
				self.plot.axis_hide_title(PLOT_AXIS_RIGHT)
				self.plot.axis_hide_title(PLOT_AXIS_LEFT)
				self.plot.axis_hide_title(PLOT_AXIS_BOTTOM)                
				self.plot.hide_legends()
				self.canvas.add_plot(self.plot, 0.16, 0.05)
				self.theDataList = []
				for num in range( 2,9 ):
					data = GtkPlotData()
					data.set_symbol(PLOT_SYMBOL_NONE, PLOT_SYMBOL_OPAQUE, 0, 0, self.getColor(num-2))
					data.set_line_attributes(PLOT_LINE_SOLID, 1, self.getColor(num-2))
        
					self.plot.add_data(data)
					self.theDataList.append(data)

				self['frame8'].add(self.canvas)

				self.createLogger()
				self.plot.clip_data(TRUE)

				aWindowWidget.show_all()
				self.update()
				# ------------------------------------------------------

			else:

				aMessage = "Error: (%s) is not numerical data" %aFullPNString 
				self.thePluginManager.printMessage( aMessage ) 
				aDialog = ConfirmWindow(0,aMessage,'Error!')
                
		else:
			aClassName = self.__class__.__name__
			for aFullPN in self.theFullPNList():
				if self.theSession.theSimulator.isNumber( aFullPN ):
					a = self.thePluginManager.createInstance( aClassName, (aFullPN,), root)
				else:
					aFullPNString = createFullPNString( self.theFullPN() )     
					self.theSession.printMessage( "%s: not numerical data\n" % aFullPNString )                    
		#self.addPopupMenu(0,1,0)

	# end of __init__

	def popInputWindow(self, obj):

		aInputWindow = Window('plugins/InputWindow.glade')
		aInputWindow.openWindow()

		self.addHandlers({'entry1_activate' : self.changexaxis,
		                  'entry2_activate' : self.changeyaxis,
		                  'entry3_activate' : self.changeyaxis,
		                  'entry4_activate' : self.changeyaxis,
		                  'entry5_activate' : self.changeyaxis,
		                  'entry6_activate' : self.changeyaxis,
		                  'entry7_activate' : self.changeyaxis,
		                  'entry8_activate' : self.changeyaxis,
		                 })

		for num in range( 1,8 ):
		#for num in range( 1,7 ):
			#self['toolbar%i'%num].set_style(GTK.TOOLBAR_ICONS)
			aInputWindow['toolbar%i'%num].set_style(GTK.TOOLBAR_ICONS)

		#self['entry1'].set_text( 'time' )
		aInputWindow['entry1'].set_text( 'time' )
		num = 1
		for fpn in self.theFullPNList():
			entrynum = num + 1
			#self['entry%i'%entrynum].set_text( 'X%i'%num )
			aInputWindow['entry%i'%entrynum].set_text( 'X%i'%num )
			num += 1
            
	def changeScale( self, obj ):

		if obj.get_active():
			self.plot.set_yscale(PLOT_SCALE_LOG10)
			self.scale = "log"
			self.plot.autoscale()
            
		else:
			self.plot.set_yscale(PLOT_SCALE_LINEAR)
			self.scale = 'linear'
			self.plot.autoscale()

	def createLogger( self ):

		n = 1
		for fpn in self.theFullPNList():

			ID = fpn[2]
			aFullPNString = createFullPNString( fpn )
			self.theSession.createLogger( aFullPNString )
			self.theLoggerList.append( aFullPNString )
			label = "label%d"%(n)
			self[label].set_text(ID)
			n += 1

		while n <= 7:

			self['button%i'%n].set_mode(TRUE)
			n += 1

		self.thePluginManager.updateFundamentalWindows()
        

	def getColor( self, num ):

		aColorList = ["red", "blue", "orange", "green",
		              "purple", "navy", "brown", "black",
		              "white", "light_blue", "light_yellow", "black"]

		return self.theColorMap.alloc( aColorList[num] )
    

	def updateLoggerDataList(self):
		self.LoggerDataList =[]
		#for aLogger in self.theLoggerList:
		#	aLoggerStub = LoggerStub( self.theSession.theSimulator, aLogger )
		#	aEndTime = aLoggerStub.getEndTime()
		#	self.LoggerDataList.append( aLoggerStub.getDataWithStartEndInterval(self.StartTime, aEndTime, 1) )

		#self.StartTime = aEndTime

		for aLogger in self.theLoggerList:
			aLoggerStub = LoggerStub( self.theSession.theSimulator, aLogger )
			aEndTime = aLoggerStub.getEndTime()
			if self.StartTime < aEndTime:
				self.LoggerDataList.append( aLoggerStub.getDataWithStartEndInterval(self.StartTime, aEndTime, 1) )
			else:
				aValue = self.theSession.theSimulator.getEntityProperty( createFullPNString(self.theFullPN()) )
				self.theFirstLog = [[self.StartTime,aValue,aValue,aValue,aValue]]
				self.LoggerDataList.append( self.theFirstLog )

		self.StartTime = aEndTime


	def setStateList( self ):
		aStateList = []
		if self['togglebutton2'] == None:
			aStateList = [0,0,0,0,0,0,0]
		else:
			for togglebuttonnum in range( 2,9 ):
				togglebutton = 'togglebutton%i'%togglebuttonnum
				aStateList.append(self[togglebutton].get_active())
		return aStateList

	def updateFullDataList( self ):
		num = 0
		FullDataList_prev = self.FullDataList
		self.FullDataList = []
		for LoggerData in self.LoggerDataList:
			if len(FullDataList_prev) > num :
				self.FullDataList.append( concatenate( (FullDataList_prev[num], array( LoggerData ))))
			if num >= len(FullDataList_prev):
				if LoggerData != self.theFirstLog:
					self.FullDataList.append( array(LoggerData) )
                
			num += 1

		num = 0
		self.entry = {}
		while num < len(self.theLoggerList):
			if self.LoggerDataList[num] == ():
				pass
			else:
				self.entry['X%s' % str(num+1)] = self.FullDataList[num]
#just temporarily                
				self.entry['time'] = self.FullDataList[0]
			num += 1

	def update( self ):
		if self.arg % 10 == 5:
			theStateList = self.setStateList()
			self.updateLoggerDataList()
            
			try:
				if self.LoggerDataList[0] == ():
					pass
				else:

					self.updateFullDataList()
					self.xList = []
					self.yList = []
					if self.xaxis == None or self.xaxis == 'time':
						if self.yaxis == None:
							for num in range(len(self.LoggerDataList)):
								self.xList.append(self.FullDataList[num][:,0])
								self.yList.append(self.FullDataList[num][:,1])

						else:
							num = 2
							for yaxis in self.yaxis:
								if yaxis == '':
									self.xList.append (['None'])
									self.yList.append (['None'])
								else:
## Must execute interpolation when the caluculated objects don't have the same step intervals
## Now I assume they have the same step intervals                        
                                
									try:
										self.xList.append ( self.FullDataList[0][:,0] )    
										self.yList.append ( eval(yaxis, self.entry)[:,1] )
									except NameError:
										self.theSession.printMessage( "name '%s' is not defined \n" % yaxis )
										self["entry%i"%num].set_text ('')
										self.changeyaxis( self["entry%i"%num])
										self.yList.append(['None'])
									except (SyntaxError,TypeError):
										self.theSession.printMessage( "'%s' is SyntaxError or TypeError\n" % yaxis )
										self["entry%i"%num].set_text ('')
										self.changeyaxis( self["entry%i"%num])
										self.yList.append(['None'])
								num += 1
                                    
					else:
						if self.yaxis == None:
							for num in range(len(self.LoggerDataList)):
								try :
									self.xList.append ( eval(self.xaxis, self.entry)[:,1] )
								except NameError:
									self.theSession.printMessage( "name '%s' is not defined \n" % self.xaxis )
									self["entry1"].set_text ('time')
									self.changexaxis( self["entry1"] )
									self.xList.append(self.FullDataList[0][:,0])
								except (SyntaxError,TypeError):
									self.theSession.printMessage( "'%s' is SyntaxError or TypeError\n" % self.xaxis )
									self["entry1"].set_text ('time')
									self.changexaxis( self["entry1"] )                                
									self.xList.append(self.FullDataList[0][:,0]) 
								self.yList.append ( self.FullDataList[num][:,1] )
						else:
							num = 2
							for yaxis in self.yaxis:
								if yaxis == '':
									self.xList.append (['None'])
									self.yList.append (['None'])
								else:
									try:
										self.xList.append( eval(self.xaxis, self.entry)[:,1])
									except NameError:
										self.theSession.printMessage( "name '%s' is not defined \n" % self.xaxis )
										self["entry1"].set_text ('time')
										self.changexaxis( self["entry1"] )                                    
										self.xList.append(self.FullDataList[0][:,0])
									except (SyntaxError,TypeError):
										self.theSession.printMessage( "'%s' is SyntaxError or TypeError\n" % self.xaxis )
										self["entry1"].set_text ('time')
										self.changexaxis( self["entry1"] )                                    
										self.xList.append(self.FullDataList[0][:,0])                                    

									try:
										self.yList.append( eval(yaxis, self.entry)[:,1])
									except NameError:
										self.theSession.printMessage( "name '%s' is not defined \n" % yaxis )
										self["entry%i"%num].set_text ('')
										self.changeyaxis( self["entry%i"%num] )
										self.yList.append(['None'])
									except (SyntaxError,TypeError):
										self.theSession.printMessage( "'%s' is SyntaxError or TypeError \n" % yaxis )
										self["entry%i"%num].set_text ('')
										self.changeyaxis( self["entry%i"%num] )
										self.yList.append(['None'])
								num += 1
           	 
				#
				if self.yaxis == None:
					num = 0
					for LoggerData in self.LoggerDataList:
						if LoggerData == ():
							pass
						else:
							if theStateList[num] == 1:
								self.theDataList[num].set_points( None, None )
							else:
								if self.scale == "log":
									if min(array(LoggerData)[:,1]) <= 0:
										self.theSession.printMessage( "value is under 0, set yaxis to linear scale\n" )
										self.plot.set_yscale(PLOT_SCALE_LINEAR)
										self.scale = "linear"
								self.theDataList[num].set_points( self.xList[num], self.yList[num] )
						num += 1

				else:
					num = 0
					for yaxis in self.yaxis:
						if theStateList[num] == 1:
							self.theDataList[num].set_points(None,None)
						elif yaxis == '':
							self.theDataList[num].set_points(None,None)
						else :
							if self.scale == 'log':
								if min(array(LoggerData)[:,1] ) <= 0:
									self.theSession.printMessage( "value is under 0, set yaxis to linear scale\n" )
									self.plot.set_yscale(PLOT_SCALE_LINEAR)
									self.scale = 'linear'
							self.theDataList[num].set_points(self.xList[num],self.yList[num])
						num += 1
            
			except:
				#print "error"
				pass

			self.plot.autoscale()
			self.canvas.paint()
			self.canvas.refresh()
            

		self.arg += 1
		return TRUE

    
	def changexaxis(self, obj):
		self.xaxis = obj.get_text()
        
        
	def changeyaxis(self, obj):
		self.yaxis = []
		for num in range( 2,9 ):
			self.yaxis.append(self['entry%i'%num].get_text())



#######test code###########

if __name__ == '__main__':
    class simulator :

        def __init__( self ):
            self.dic = {('Variable', '/CELL/CYTOPLASM', 'ATP','Value') : (1950,),}

        def getEntityProperty( self, fpn ):
            return self.dic[fpn]

        def setEntityProperty( self, fpn, value ):
            self.dic[fpn] = value

        def getLogger( self, fpn ):
            logger= Logger( fpn )
            return logger

        def getLoggerList( self ):
            fpnlist = ((VARIABLE, '/CELL/CYTOPLASM', 'ATP', 'Value'),
                       (VARIABLE, '/CELL/CYTOPLASM', 'ADP', 'Value'))
                       
            return fpnlist

    fpnlist = (('Variable','/CELL/CYTOPLASM','ATP',''),
               ('Variable','/CELL/CYTOPLASM','ADP',''))


    class Logger:

        def __init__( self, fpn ):
            if(fpn ==('Variable', '/CELL/CYTOPLASM', 'ATP', '')) :
                a = 'CYTOPLASM-Cln2_C.ecd'
                alist = []
                fp = open(a,'r')

                for line in fp.readlines():
                    if re.search('^\d+.\d+\s',line):
                        alist+=[[string.atof(re.split('\s+',line)[0]),string.atof(re.split('\s+',line)[1])]]
            elif(fpn ==('Variable', '/CELL/CYTOPLASM', 'ADP', '')) :
                a = 'CYTOPLASM-mass_C.ecd'
                alist = []
                fp = open(a,'r')

                for line in fp.readlines():
                    if re.search('^\d+.\d+\s',line):
                        alist+=[[string.atof(re.split('\s+',line)[0]),string.atof(re.split('\s+',line)[1])]]
            else:
                print "error"
            self.Data = array(alist)
            
        def getStartTime( self ):
            return 0

        def getEndTime( self ):
            return 100

        def getLoggerData( self, start, end):
            return self.Data[start:end]
        

               
    def mainQuit( obj, data ):
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        atestTracer = TracerWindow(simulator,fpnlist)
        mainloop()
#        mainLoop()

    main() 





 
