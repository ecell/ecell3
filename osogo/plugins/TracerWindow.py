#!/usr/bin/env python

import gtk
import Numeric as nu
import gtk.gdk
import re
import string
import operator
from Plot import *
from ecell.ecssupport import *
#LoggerMinimumInterval=1

from OsogoPluginWindow import *
from ConfirmWindow import *
import os
import os.path

COL_LOG = 2
COL_PIX = 1
COL_ON = 0
COL_TXT = 4
COL_X = 3



class TracerWindow( OsogoPluginWindow ):



    def __init__( self, dirname, data, pluginmanager, root=None ):
        #initializa variables:
        #initiates OsogoPluginWindow
        OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
        self.thePluginManager = pluginmanager
        self.theDataGenerator = self.thePluginManager.getDataGenerator()
        self.displayedFullPNStringList = []
        
        #get session
        self.theSession = self.thePluginManager.theSession
        self.theViewType = MULTIPLE
        self.isGUIShown = False
        self.theSaveDirectorySelection = gtk.FileSelection( 'Select File' )
        self.theSaveDirectorySelection.ok_button.connect('clicked', self.changeSaveDirectory)
        self.theSaveDirectorySelection.cancel_button.connect('clicked', self.closeParentWindow)


    def openWindow(self):
        OsogoPluginWindow.openWindow(self)

        #self.openWindow()
        self.theListWindow = self['clist1']
        self.theListStore = gtk.ListStore(gobject.TYPE_BOOLEAN,\
            gobject.TYPE_OBJECT, gobject.TYPE_BOOLEAN,\
            gobject.TYPE_BOOLEAN,
            gobject.TYPE_STRING )
        self.theListWindow.set_model( self.theListStore )

        onoffRenderer = gtk.CellRendererToggle()
        onoffRenderer.connect( 'toggled', self.onoffTogglePressed, self.theListStore )
        loggerRenderer = gtk.CellRendererToggle()
        loggerRenderer.connect( 'toggled', self.loggerTickBoxChecked, self.theListStore )
        pixbufRenderer = gtk.CellRendererPixbuf()
        xaxisRenderer = gtk.CellRendererToggle()
        xaxisRenderer.connect( 'toggled', self.xaxisToggled, self.theListStore )

        pixBufColumn = gtk.TreeViewColumn( 'color', pixbufRenderer, pixbuf = COL_PIX )
        fullpnColumn = gtk.TreeViewColumn( 'trace', gtk.CellRendererText(), text = COL_TXT )
        loggerColumn = gtk.TreeViewColumn( 'lg', loggerRenderer, active=COL_LOG )
        onoffColumn = gtk.TreeViewColumn( 'on', onoffRenderer, active = COL_ON )
        xaxisColumn = gtk.TreeViewColumn ( 'x', xaxisRenderer, active = COL_X )

        pixBufColumn.set_resizable( gtk.TRUE )
        fullpnColumn.set_resizable( gtk.TRUE )
        loggerColumn.set_resizable( gtk.TRUE )
        onoffColumn.set_resizable( gtk.TRUE )
        xaxisColumn.set_resizable( gtk.TRUE )

        self.theListWindow.append_column( onoffColumn )
        self.theListWindow.append_column( pixBufColumn )
        self.theListWindow.append_column( loggerColumn )
        self.theListWindow.append_column( xaxisColumn )
        self.theListWindow.append_column( fullpnColumn )

        self.theScrolledWindow = self['scrolledwindow1']
        self.theFixed = self['fixed1']

        self.theListSelection = self.theListWindow.get_selection()
        self.theListSelection.set_mode( gtk.SELECTION_MULTIPLE )
        self.theWindow = self.getWidget( self.__class__.__name__ )

        #determine plotsize
        self.thePlotWidget = self['drawingarea1']

        #init plotter instance
        self.thePlotInstance = Plot( self, self.getParent(), self.thePlotWidget )
        #attach plotterwidget to window

        aWindowWidget = self.getWidget( 'frame8' )
        self.noHandle = False

        aWindowWidget.show_all()

        #add handlers to buttons

        self.theListWindow.connect( "button-press-event", self.buttonPressedOnList)

        self.setIconList(
        os.environ['OSOGOPATH'] + os.sep + "ecell.png",
        os.environ['OSOGOPATH'] + os.sep + "ecell32.png")

        #addtrace to plot
        self.addTraceToPlot( self.theFullPNList() )

        #sets stripinterval, disable history buttons
        self['entry1'].set_text( str(self.thePlotInstance.getStripInterval()) )
        self['entry1'].connect( 'activate', self.stripIntervalChangedEnter )
        self['entry1'].connect( 'focus_out_event', self.stripIntervalChanged )

        if not self.isStandAlone():
            self.minimize()


        self.thePluginManager.appendInstance( self )

        self.showHistory()               


    def update(self):
        self.thePlotInstance.update()
        # later update value shown


    def createLogger( self, fpnlist ):
        if self.theSession.isRunning():
            return
        logPolicy = self.theSession.getLogPolicyParameters()
        for fpn in fpnlist:
            if not self.hasLogger(fpn):
                try:
                    self.theSession.theSimulator.createLogger(fpn, logPolicy )
                except:
                    self.theSession.message( 'Error while creating logger\n logger for ' + fpn + ' not created\n' )
                else:
                    self.theSession.message( "Logger created for " + fpn )
        #self.checkHistoryButton()
        self.thePluginManager.updateFundamentalWindows()


    def requestData( self, numberOfElements ):
        for aSeries in self.thePlotInstance.getDataSeriesList():
            self.theDataGenerator.requestData( aSeries, numberOfElements )

    def requestDataSlice( self, aStart, anEnd, aRequiredResolution ):
        for aSeries in self.thePlotInstance.getDataSeriesList(): 
            self.theDataGenerator.requestDataSlice( aSeries, aStart, anEnd, aRequiredResolution )

    def requestNewData( self, aRequiredResolution ):
        for aSeries in self.thePlotInstance.getDataSeriesList():
            self.theDataGenerator.requestNewData( aSeries, aRequiredResolution )


    # ========================================================================
    def hasLogger(self, aFullPNString):
        loggerlist=self.theSession.theSimulator.getLoggerList()
        return aFullPNString in loggerlist

    # ========================================================================
#    def checkHistoryButton(self):
#        history_button = self['togglebutton3']
#        if len( self.displayedFullPNStringList ) == 0:
#            history_button.set_sensitive( gtk.FALSE )
#            return None 
#        for fpn in self.displayedFullPNStringList:
#            if not self.hasLogger(fpn):
#                history_button.set_sensitive(gtk.FALSE)
#                return None 
#        history_button.set_sensitive(gtk.TRUE)


    # ========================================================================
    def appendRawFullPNList( self, aRawFullPNList ):
        """overwrites superclass method
        aRawFullPNList  -- a RawFullPNList to append (RawFullPNList) 
        Returns None
        """

        # calls superclass's method
        OsogoPluginWindow.appendRawFullPNList( self, aRawFullPNList )
        # creates FullPNList to plot
        aFullPNList = map( self.supplementFullPN, aRawFullPNList )

        # appends FullPNList as plot data
        self.addTraceToPlot( aFullPNList )


    # ========================================================================
    def refreshLoggers(self):
        #refreshes loggerlist
        iter = self.theListStore.get_iter_first()
        while iter != None:
            text = self.theListStore.get_value( iter, COL_TXT )
            if self.hasLogger( text ):
                fixed=gtk.TRUE
            else:
                fixed=gtk.FALSE
            self.theListStore.set(iter, COL_LOG, fixed)
            iter = self.theListStore.iter_next(iter)


    # ========================================================================
    def addTraceToPlot(self,aFullPNList):
        #checks that newpn has logger if mode is history
        #calls superclass
        pass_flag = 0

        if self.theSession.getParameter('log_all_traces'):
            for aFullPN in aFullPNList:
                aFullPNString = createFullPNString( aFullPN )
                self.createLogger( [aFullPNString] )

        if self.thePlotInstance.getStripMode() == 'history':
            for aFullPN in aFullPNList:
                aFullPNString= createFullPNString( aFullPN)
                if not self.hasLogger(aFullPNString):
                    self.theSession.message(aFullPNString+" doesn't have associated logger.")
                    pass_flag = 1
            if pass_flag==1:
                return -1

        pass_list = []
        for aFullPN in aFullPNList: 
            aFullPNString = createFullPNString( aFullPN )
            if aFullPNString in self.displayedFullPNStringList:
                continue
            #gets most recent value
            #check whether there's enough room left
            if len(self.displayedFullPNStringList) < self.thePlotInstance.getMaxTraces():
                #adds trace to plotinstance,clist, add to displaylist, colorlist
                aValue = self.getLatestData( aFullPNString )
                if operator.isNumberType( aValue[1] ):
                    self.displayedFullPNStringList.append( aFullPNString )
                    pass_list.append( aFullPNString )
                else:
                    self.theSession.message('%s cannot be displayed, because it is not numeric\n' % aFullPNString)
        added_list = self.thePlotInstance.addTrace( pass_list )
        self.addTraceToList( added_list )
        #self.checkHistoryButton()
        #self.checkRemoveButton()
        
    # ========================================================================
    def getLatestData( self, fpn ):
        value = self.theSession.theSimulator.getEntityProperty( fpn )
        time = self.theSession.theSimulator.getCurrentTime()
        return nu.array( [time,value, value, value, value] )
        
    # ========================================================================
#    def checkRemoveButton( self ):
#        remove_button = self['button9']
#        if len( self.displayedFullPNStringList ) > 1:
#            remove_button.set_sensitive( gtk.TRUE )
#        else:
#            remove_button.set_sensitive( gtk.FALSE )


    # ========================================================================
    def getSelected(self):
        self.selectionList=[]
        self.theListSelection.selected_foreach(self.selection_function)
        return self.selectionList
        

    # ========================================================================
    def selection_function( self, model, path, iter ):
        text = self.theListStore.get_value( iter, COL_TXT )
        self.selectionList.append([text,iter])
        

    # ========================================================================
    def addTraceToList( self, added_list ):

        xAxis = self.thePlotInstance.getXAxisFullPNString()
        for fpn in added_list:
            iter=self.theListStore.append()
            self.noHandle = True
            aSeries = self.thePlotInstance.getDataSeries( fpn )
            self.theListStore.set_value( iter, COL_PIX, aSeries.getPixBuf() ) #set pixbuf
            self.theListStore.set_value( iter, COL_TXT, fpn ) #set text
            self.theListStore.set_value( iter, COL_ON, aSeries.isOn() ) #trace is on by default
            self.theListStore.set_value( iter, COL_X, fpn == xAxis ) #set text
            self.noHandle = False
        self.refreshLoggers()



    # ========================================================================
    def changeTraceColor(self):
        selected_list = self.getSelected()
        if len( selected_list ) > 0:
            fpn = selected_list[0][0]
            iter = selected_list[0][1]
            aSeries = self.thePlotInstance.getDataSeries( fpn )
            aSeries.changeColor()
            pixbuf = aSeries.getPixBuf()
            self.theListStore.set_value( iter, COL_PIX, pixbuf )
        
    def removeTraceFromList(self,aFullPNString):
        pass

    def shrink_to_fit(self):
        pass

    
    
    # ========================================================================
    def maximize(self):
        if self['top_frame'] != self['vbox2'].get_parent():
            if self.isStandAlone():
                self.__adjustWindowHeight(  - self.shiftWindow )
            self['top_frame'].remove( self['drawingarea1'] )
            self['vpaned1'].add( self['drawingarea1'] )
            self['top_frame'].add( self['vbox2'] )
        
        self.thePlotInstance.showGUI( gtk.TRUE )


    # ========================================================================
    def minimize(self):

        self.thePlotInstance.showGUI( gtk.FALSE )
        if self['top_frame'] == self['vbox2'].get_parent():
            if self.isStandAlone():
                dividerPos = self['vpaned1'].get_position()
                panedHeight = self['vpaned1'].get_allocation()[3]
                self.shiftWindow = panedHeight - dividerPos

            self['top_frame'].remove( self['vbox2'] )
            self['vpaned1'].remove( self['drawingarea1'] )
            self['top_frame'].add( self['drawingarea1'] )
            if self.isStandAlone():
                self.__adjustWindowHeight(  self.shiftWindow )
                
    def __adjustWindowHeight ( self, deltaHeight ):
        aWindow = self.getParent()['TracerWindow']
        windowAlloc = aWindow.get_allocation()
        windowHeight = windowAlloc[3]
        windowWidth = windowAlloc[2]
        aWindow.resize( windowWidth , windowHeight - deltaHeight )
        
    
    
    
   # ========================================================================
    def setScale( self, theOrientation, theScaleType ):
        """
        sets scale type of the axis defined by orientation
        theOrientation is either "Horizontal" or "Vertical"
        theScaleType is either "Linear" or "Log10"
        """ 
        self.thePlotInstance.changeScale( theOrientation, theScaleType )
 
   # ========================================================================
    def setXAxis( self, theFullPN ):
        """ sets XAxis
        either a FullPN String or "time" literal
        """
        if theFullPN != "Time":
            if theFullPN not in self.thePlotInstance.getDataSeriesNames():
                return

        self.thePlotInstance.setXAxis( theFullPN )
        #switch off trace
        anIter=self.theListStore.get_iter_first( )
        while True:
            if anIter == None:
                return None
            aTitle = self.theListStore.get_value(anIter, COL_TXT )


#            if aTitle == theFullPN:
            self.noHandle = True
            aSeries = self.thePlotInstance.getDataSeries( aTitle )
            self.theListStore.set_value( anIter, COL_ON, aSeries.isOn() )
            self.theListStore.set_value( anIter, COL_X, theFullPN == aTitle )
            self.noHandle = False

            anIter=self.theListStore.iter_next( anIter )
        
            
    
    # ========================================================================
    def setStripInterval( self, anInterval ):
        """ sets striptinterval of graph to anInterval """
        self['entry1'].set_text( str( anInterval ) )
        self.stripIntervalChanged(None, None )
    

    # ========================================================================
    def showHistory (self):
        """ changes Plot to History mode
            e.g. draws plot from logger information
            will fall back to strip mode if not each and every
            FullPN has a logger
            returns None
        """
        if self.thePlotInstance.getStripMode() != 'history':
            self.toggleStripAction( None )
            

    # ========================================================================
    def showStrip (self):
        """ changes Plot to Strip mode
            e.g. shows the most recent datapaoints
            spanning an interval set by StripInterval
        """
        if self.thePlotInstance.getStripMode() == 'history':
            self.toggleStripAction( None )

    # ========================================================================
    def logAll(self):
        """ creates logger for all traces on TracerWindow """
        self.logAllAction( None )

    # ========================================================================
    #def setTraceColor(aFullPN, red, green, blue):
    #TBD

    # ========================================================================
    def setTraceVisible (self, aFullPNString, aBoolean):
        """ sets visible trace of identified by FullPNString 
            aBoolean:
            gtk.TRUE - Display
            gtk,FALSE - Don't display trace
        """
        if aFullPNString not in self.thePlotInstance.getDataSeriesNames():
            return
        aSeries = self.thePlotInstance.getDataSeries( aFullPNString )
        currentState = aSeries.isOn()
        if currentState:
            return None
        if currentState == aBoolean:
            return None
        anIter=self.theListStore.get_iter_first()
        while True:
            if anIter == None:
                return None
            aTitle = self.theListStore.get_value(anIter, COL_TXT )

            if aTitle == aFullPNString:
                aSeries = self.thePlotInstance.getDataSeries( aFullPNString )
                if aBoolean:
                    aSeries.switchOn()
                else:
                    aSeries.switchOff()
                self.noHandle = True
                self.theListStore.set_value( anIter, COL_ON, aSeries.isOn() )
                self.noHandle = False

                break
            anIter=self.theListStore.iter_next( anIter )


    # ========================================================================
    def zoomIn (self, x0,x1, y0, y1 ):
        """ magnifies a rectangular area of  Plotarea
            bordered by x0,x1,y0,y1
        """
        if x1<0 or x1<=x0 or y1<=y0:
            self.thePluginManager.theSession.message("bad arguments")
            return
        self.thePlotInstance.zoomIn( [x0, x1], [y1, y0])

    # ========================================================================
    def zoomOut(self, aNum = 1):
        """ zooms out aNum level of zoom ins 
        """
        for i in range(0, aNum):
            self.thePlotInstance.zoomOut()

    # ========================================================================
    def showGUI ( self ):
        """ shows GUI and sets plot to its normal size """
        self.maximize()

    # ========================================================================
    def hideGUI (self ):
        """doesn't change Plot size, but hides GUI components """
        self.minimize()

    # ========================================================================
    def checkRun( self ):
        if self.theSession.isRunning():
            # displays a Confirm Window.
            aMessage = "Cannot create new logger, because simulation is running.\n"
            aMessage += "Please stop simulation if you want to create a logger" 
            aDialog = ConfirmWindow(OK_MODE,aMessage,'Warning!')
            return True
        return False


#----------------------------------------------
#SIGNAL HANDLERS
#-------------------------------------------------



    #----------------------------------------------
    #this signal handler is called when ENTER is pushed on entry1
    #-------------------------------------------------

    def stripIntervalChangedEnter( self, obj ):
        self.stripIntervalChanged( obj, None )

    #--------------------------------------------------------
    #this signal handler is called when TAB is presses on entry1
    #---------------------------------------------------------

    def stripIntervalChanged(self, obj, event): #this is an event handler again
        #get new value
        #call plotterinstance
        try:
            a = float( self['entry1'].get_text() )
        except ValueError:
            self.theSession.message("Enter a valid number, please.")
            self['entry1'].set_text( str( self.thePlotInstance.getstripinterval() ) )
        else:
            self.thePlotInstance.setStripInterval( a )


    #--------------------------------------------------------
    #this signal handler is called when mousebutton is pressed over the fullpnlist
    #---------------------------------------------------------

    def buttonPressedOnList(self, aWidget, anEvent):
        if anEvent.button == 3:
        # user menu: remove trace, log all, save data,edit policy, hide ui, change color
            selectedList = self.getSelected()
            allHasLogger = True
            xAxisSelected = False
            xAxis = self.thePlotInstance.getXAxisFullPNString()
            for aSelection in selectedList:
                if not self.hasLogger( aSelection[0] ):
                    allHasLogger = False
                if aSelection[0] == xAxis:
                    xAxisSelected = True
                

            theMenu = gtk.Menu()
            listCount = len( self.displayedFullPNStringList )
            if len( selectedList ) > 0 and listCount - len(selectedList )  > 0 and not xAxisSelected:
            
                removeItem = gtk.MenuItem( "Remove" )
                removeItem.connect( "activate", self.removeTraceAction )
                theMenu.append( removeItem )
                theMenu.append( gtk.SeparatorMenuItem() )

            if allHasLogger and listCount> 0:
                logItem = gtk.MenuItem( "Save data" )
                logItem.connect( "activate", self.__saveData, selectedList )
                theMenu.append( logItem )
                editPolicy = gtk.MenuItem( "Edit policy" )
                editPolicy.connect( "activate", self.__editPolicy, selectedList )
                theMenu.append( editPolicy )
                theMenu.append( gtk.SeparatorMenuItem() )
                
            if len( selectedList ) == 1:
                toggleColorItem = gtk.MenuItem( "Toggle color" )
                toggleColorItem.connect("activate", self.__toggleColor, selectedList )
                theMenu.append( toggleColorItem )
            theMenu.show_all()
            theMenu.popup( None, None, None, 1, 0 )


    def __editPolicy( self, *args ):
        fpnList = args[1]
        if len( fpnList ) == 1:
            # get loggerpolicy
            aLoggerStub = self.theSession.createLoggerStub( fpnList[0][0] )
            aLogPolicy = aLoggerStub.getLoggerPolicy()
        else:
            aLogPolicy = [ 1, 0, 0, 0 ]
        newLogPolicy = self.theSession.openLogPolicyWindow( aLogPolicy, "Set log policy for selected loggers" )
        if newLogPolicy == None:
            return
        for anItem in fpnList:
            aFullPN = anItem[0]
            aLoggerStub = self.theSession.createLoggerStub( aFullPN )
            aLoggerStub.setLoggerPolicy( newLogPolicy )


    def __toggleColor( self, *args ):
        self.changeTraceColor()


    def __saveData( self, *args ):
        self.theSaveDirectorySelection.show_all()


    def changeSaveDirectory( self, obj ):
        aSaveDirectory = self.theSaveDirectorySelection.get_filename()
        self.theSaveDirectorySelection.hide()
        selectedFullPNList = []
        for anItem in self.getSelected():
            selectedFullPNList.append( anItem[0] )
        # If same directory exists.
        if os.path.isdir(aSaveDirectory):
            aConfirmMessage = "%s directory already exist.\n Would you like to override it?"%aSaveDirectory
            confirmWindow = ConfirmWindow(1,aConfirmMessage)

            if confirmWindow.return_result() == 0:
                pass
            else:
                return None

        # If same directory dose not exists.
        else:
            try:
                os.mkdir(aSaveDirectory)
            except:
                aErrorMessage='couldn\'t create %s!\n'%aSaveDirectory
                aWarningWindow = ConfirmWindow(0,aErrorMessage)
                return None


        try:
            self.theSession.saveLoggerData( selectedFullPNList, aSaveDirectory, -1, -1, -1 )
        except:
            anErrorMessage = "Error : could not save "
            aWarningWindow = ConfirmWindow(0,anErrorMessage)
            return None
        
        


    def closeParentWindow( self, obj ):
        aParentWindow = self.theSaveDirectorySelection.cancel_button.get_parent_window()
        aParentWindow.hide()


    #--------------------------------------------------------
    #this signal handler is called when an on-off checkbox is pressed over of the fullpnlist
    #---------------------------------------------------------

    def onoffTogglePressed(self,cell, path, model):
        if self.noHandle:
            return
        iter = model.get_iter( (int( path ), ) )
        text = self.theListStore.get_value( iter, COL_TXT )
        aSeries = self.thePlotInstance.getDataSeries( text )
        if aSeries.isOn( ):
            aSeries.switchOff()
        else:
            aSeries.switchOn()
        self.thePlotInstance.totalRedraw()
        self.noHandle = True
        self.theListStore.set_value( iter, COL_ON, aSeries.isOn() )
        self.noHandle = False

    #--------------------------------------------------------
    #this signal handler is called when create logger checkbox is pressed over the fullpnlist
    #---------------------------------------------------------

    def loggerTickBoxChecked(self, cell, path, model):
        if self.noHandle:
            return
        iter = model.get_iter( ( int ( path ), ) )
        fixed = model.get_value( iter, COL_LOG )
        text = self.theListStore.get_value( iter, COL_TXT )

        if fixed == gtk.FALSE:
            if self.checkRun():
                return
            self.createLogger( [text] )
            self.refreshLoggers()

    #--------------------------------------------------------
    #this signal handler is called when xaxis is toggled
    #--------------------------------------------------------

    def xaxisToggled(self, cell, path, model):
        if self.noHandle:
            return
        iter = model.get_iter( ( int ( path ), ) )
        fixed = model.get_value( iter, COL_X )
        text = self.theListStore.get_value( iter, COL_TXT )
        
        if fixed == gtk.FALSE:
            self.setXAxis( text )
        else:
            self.setXAxis( "Time" )



    #--------------------------------------------------------
    #this signal handler is called when "Remove Trace" button is pressed
    #---------------------------------------------------------

    def removeTraceAction(self, *args ):
        #identify selected FullPNs
        fpnlist=[]      
        selected_list=self.getSelected()
        for aselected in selected_list:
            #remove from fullpnlist
            if len(self.displayedFullPNStringList)==1:
                break
                
            FullPNList = self.theRawFullPNList[:]
            for afullpn in FullPNList:
                if aselected[0] == createFullPNString( afullpn):
                    self.theRawFullPNList.remove(afullpn)
                    break       
            #remove from displaylist
            self.displayedFullPNStringList.remove( aselected[0] )
            fpnlist.append( aselected[0] )
            self.theListStore.remove( aselected[1] )
            #remove from plotinstance
        
        self.thePlotInstance.removeTrace( fpnlist )
        #delete selected from list
        #self.checkHistoryButton()
        #self.checkRemoveButton()

    #--------------------------------------------------------
    #this signal handler is called when "Log All" button is pressed
    #---------------------------------------------------------

    def logAllAction(self,obj):
        if not self.checkRun():
            return
        #creates logger in simulator for all FullPNs 
        self.createLogger( self.displayedFullPNStringList )      
        self.refreshLoggers()
        
    #--------------------------------------------------------
    #this signal handler is called when "Show History" button is toggled
    #---------------------------------------------------------
        
    def toggleStripAction(self, obj):
        #if history, change to strip, try to get data for strip interval
        stripmode = self.thePlotInstance.getStripMode()
        if stripmode == 'history':
            self.thePlotInstance.setStripMode( 'strip' )
        else:
            pass_flag = True
            for fpn in self.displayedFullPNStringList:
                if not self.hasLogger(fpn): 
                    pass_flag = False
                    break
            if pass_flag:
                self.thePlotInstance.setStripMode( 'history' )
            else:
                self.theSession.message("can't change to history mode, because not every trace has logger.\n")


    #--------------------------------------------------------
    #this signal handler is called when "Minimize" button is pressed
    #--------------------------------------------------------

    def hideGUIAction(self,button_obj):
        self.minimize()


# tracer window button removal - Hide GUI, Log All, change scale, show history, only stripinterval stays

# tracer resize
# boardwindow
