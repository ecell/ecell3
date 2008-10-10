#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

import os
import os.path
import gtk
import gobject
import numpy as nu
import gtk.gdk
import re
import string
import operator

import ecell.util as util
import ecell.ui.osogo.config as config
from ecell.ui.osogo.constants import *
from ecell.ui.osogo.OsogoPluginWindow import OsogoPluginWindow
from ecell.ui.osogo.utils import *
from ecell.ui.osogo.Plot import *

COL_LOG = 2
COL_PIX = 1
COL_ON = 0
COL_TXT = 4
COL_X = 3
COL_FULLPN = 5

class TracerWindow( OsogoPluginWindow ):
    theViewType = MULTIPLE

    def __init__( self, dirname, data, aPluginManager ):
        #initializa variables:
        #initiates OsogoPluginWindow
        OsogoPluginWindow.__init__(
                self, dirname, data, aPluginManager.theSession )
        self.theDataGenerator = self.theSession.getDataGenerator()
        self.displayedFullPNList = []
        self.thePixmapDict = {} #key is color, value pixmap
        
        #get session
        self.isControlShown = True
        self.theSaveDirectorySelection = gtk.FileSelection( 'Select File' )
        self.theSaveDirectorySelection.ok_button.connect('clicked', self.changeSaveDirectory)
        self.theSaveDirectorySelection.cancel_button.connect('clicked', self.closeParentWindow)

    def initUI( self ):
        OsogoPluginWindow.initUI( self )

        self.theListWindow = self['clist1']
        self.theTopFrame = self['top_frame'] 
        self.theVbox2= self['vbox2']
        self.thePaned = self['vpaned1']
        self.thePlotInstance = self['drawingarea1'] 
        self.thePlotInstance.setOwner( self )
        self.thePlotInstance.connect( 'button-press-event',
            lambda w, e: e.button == 3 and self.showMenu() )
        self.thePlotInstance.show()
        self.theEntry = self['entry1']
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

        pixBufColumn.set_resizable( True )
        fullpnColumn.set_resizable( True )
        loggerColumn.set_resizable( True )
        onoffColumn.set_resizable( True )
        xaxisColumn.set_resizable( True )

        self.theListWindow.append_column( onoffColumn )
        self.theListWindow.append_column( pixBufColumn )
        self.theListWindow.append_column( loggerColumn )
        self.theListWindow.append_column( xaxisColumn )
        self.theListWindow.append_column( fullpnColumn )

        self.theScrolledWindow = self['scrolledwindow1']
        self.theFixed = self['fixed1']

        self.theListSelection = self.theListWindow.get_selection()
        self.theListSelection.set_mode( gtk.SELECTION_MULTIPLE )

        #attach plotterwidget to window
        aWindowWidget = self.getWidget( 'frame8' )
        self.noHandle = False

        aWindowWidget.show_all()

        #add handlers to buttons
        self.theListWindow.connect( "button-press-event", self.buttonPressedOnList)

        self.setIconList(
            os.path.join( config.glade_dir, "ecell.png" ),
            os.path.join( config.glade_dir, "ecell32.png" ) )
        #addtrace to plot
        self.addTraceToPlot( self.getFullPNList() )
        #sets stripinterval, disable history buttons
        self.theEntry.set_text( str(self.thePlotInstance.getStripInterval()) )
        self.theEntry.connect( 'activate', self.stripIntervalChangedEnter )
        self.theEntry.connect( 'focus_out_event', self.stripIntervalChanged )

        if not self.isStandAlone():
            self.minimize()

        self.showHistory()               

    def update(self):
        self.thePlotInstance.update()
        # later update value shown

    def createLogger( self, fpnlist ):
        if self.theSession.isRunning():
            return
        for aFullPN in aFullPNlist:
            if not self.hasLogger( aFullPN ):
                try:
                    self.theSession.createLogger( aFullPN )
                except:
                    self.theSession.message( 'Error while creating logger\n logger for ' + aFullPN + ' not created\n' )
                else:
                    self.theSession.message( "Logger created for " + aFullPN )

    def requestData( self, numberOfElements ):
        for aSeries in self.thePlotInstance.getDataSeriesList():
            self.theDataGenerator.requestData( aSeries, numberOfElements )

    def requestDataSlice( self, aStart, anEnd, aRequiredResolution ):
        for aSeries in self.thePlotInstance.getDataSeriesList(): 
            self.theDataGenerator.requestDataSlice(
                aSeries, aStart, anEnd, aRequiredResolution )

    def requestNewData( self, aRequiredResolution ):
        for aSeries in self.thePlotInstance.getDataSeriesList():
            self.theDataGenerator.requestNewData( aSeries, aRequiredResolution )

    def allHasLogger( self ):
        loggerList = self.theSession.getLoggedPNList()
        for aSeries in self.thePlotInstance.getDataSeriesList():
            if aSeries.getFullPN() not in loggerList:
                return False
        return True
    
    def hasLogger(self, aFullPN):
        return aFullPN in self.theSession.getLoggedPNList()

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

    def refreshLoggers(self):
        #refreshes loggerlist
        iter = self.theListStore.get_iter_first()
        aLoggerList = self.theSession.getLoggedPNList()
        while iter != None:
            aFullPN = identifiers.FullPN(
                self.theListStore.get_value( iter, COL_TXT ) )
            self.theListStore.set( iter, COL_LOG, aFullPN in aLoggerList )
            iter = self.theListStore.iter_next( iter )

    def addTraceToPlot( self, aFullPNList ):
        #checks that newpn has logger if mode is history
        #calls superclass
        pass_flag = 0
        if self.theSession.getParameter( 'log_all_traces' ):
            for aFullPN in aFullPNList:
                self.theSession.createLogger( aFullPN )
        aLoggerList = self.theSession.getLoggedPNList()
        if self.thePlotInstance.getStripMode() == 'history':
            for aFullPN in aFullPNList:
                if not aFullPN in aLoggerList:
                    self.theSession.message(
                        str( aFullPN ) + " doesn't have associated logger.")
                    pass_flag = 1
            if pass_flag == 1:
                return -1

        pass_list = []
        for aFullPN in aFullPNList: 
            if aFullPN in self.displayedFullPNList:
                continue
            #gets most recent value
            #check whether there's enough room left
            if len( self.displayedFullPNList ) < self.thePlotInstance.getMaxTraces():
                #adds trace to plotinstance,clist, add to displaylist, colorlist
                aValue = self.getLatestData( aFullPN )
                if aValue != None:
                    self.displayedFullPNList.append( aFullPN )
                    pass_list.append( aFullPN )
                else:
                    self.theSession.message(
                        '%s cannot be displayed, because it is not numeric' % aFullPNS )

        added_list = []
        try:
            for aFullPN in pass_list:
                added_list.append(
                    self.thePlotInstance.addTrace( aFullPN ) )
        except RuntimeError:
            pass
        self.addTraceToList( added_list )
        self.thePlotInstance.sync()
        
    def getLatestData( self, fpn ):
        value = self.theSession.getEntityProperty( fpn )
        if not operator.isNumberType( value ):
            return None
        time = self.theSession.getCurrentTime()
        return nu.array( [time,value, value, value, value] )
        
    def getSelected( self ):
        selectionList = []
        self.theListSelection.selected_foreach(
            lambda model, path, iter: selectionList.append(
                (
                    identifiers.FullPN( model.get_value( iter, COL_TXT ) ),
                    iter
                    ) ) )
        return selectionList
        
    def addTraceToList( self, added_list ):

        xAxis = self.thePlotInstance.getXAxisFullPN()
        for aSeries in added_list:
            iter = self.theListStore.append()
            aFullPN = aSeries.getFullPN()
            self.noHandle = True
            self.theListStore.set_value(
                iter, COL_PIX, self.getPixmap( aSeries.getColor() ) )
            self.theListStore.set_value( iter, COL_TXT, str( aFullPN ) )
            self.theListStore.set_value( iter, COL_ON, aSeries.isOn() )
            self.theListStore.set_value( iter, COL_X, aFullPN == xAxis )
            self.noHandle = False
        self.refreshLoggers()

    def changeTraceColor(self):
        selected_list = self.getSelected()
        if len( selected_list ) > 0:
            fpn = selected_list[0][0]
            iter = selected_list[0][1]
            aSeries = self.thePlotInstance.getDataSeries( fpn )
            aSeries.changeColor()
            self.theListStore.set_value( iter, COL_PIX,
                    self.getPixmap( aSeries.getColor() ) )
        
    def removeTraceFromList( self,aFullPN ):
        pass

    def shrink_to_fit(self):
        pass

    def maximize( self ):
        if self.theTopFrame!= self.theVbox2.get_parent():
            if self.isStandAlone():
                self.__adjustWindowHeight(  - self.shiftWindow )
            self.theTopFrame.remove( self.thePlotInstance )
            self.thePaned.add( self.thePlotInstance )
            self.theTopFrame.add( self.theVbox2 )
        self.isControlShown = True
        self.thePlotInstance.showControl( True )

    def minimize( self ):
        self.thePlotInstance.showControl( False )
        if self.theTopFrame== self.theVbox2.get_parent():
            if self.isStandAlone():
                dividerPos = self.thePaned.get_position()
                panedHeight = self.thePaned.allocation.height
                self.shiftWindow = panedHeight - dividerPos
            self.theTopFrame.remove( self.theVbox2 )
            self.thePaned.remove( self.thePlotInstance )
            self.theTopFrame.add( self.thePlotInstance )
            if self.isStandAlone():
                self.__adjustWindowHeight( self.shiftWindow )
        self.isControlShown = False

    def showMenu( self ):
        theMenu = gtk.Menu()
        if self.thePlotInstance.theZoomLevel > 0:
            zoomUt = gtk.MenuItem( "Zoom out" )
            zoomUt.connect ("activate", lambda w: self.zoomOut() )
            theMenu.append( zoomUt )
            theMenu.append( gtk.SeparatorMenuItem() )

        if self.isControlShown:
            guiMenuItem = gtk.MenuItem( "Hide Control" )
            guiMenuItem.connect( "activate", lambda w: self.minimize() )
        else:
            guiMenuItem = gtk.MenuItem( "Show Control" )
            guiMenuItem.connect( "activate", lambda w: self.maximize() )

        def generate( anOrientation ):
            return lambda w: self.thePlotInstance.changeScale(
                anOrientation,
                self.thePlotInstance.getScaleType( anOrientation ) == \
                    SCALE_LINEAR and SCALE_LOG10 or SCALE_LINEAR
                )

        xToggle = gtk.MenuItem ( "Toggle X axis" )
        xToggle.connect( "activate", generate( PLOT_HORIZONTAL_AXIS ) )
        yToggle = gtk.MenuItem ( "Toggle Y axis" )
        yToggle.connect( "activate", generate( PLOT_VERTICAL_AXIS ) )

        #take this condition out if phase plotting works for history
        if self.allHasLogger():
            if self.thePlotInstance.getStripMode() == MODE_STRIP:
                toggleStrip = gtk.MenuItem("History mode")
                toggleStrip.connect( "activate", 
                    lambda w: self.thePlotInstance.setStripMode( MODE_HISTORY )  )
            else:
                toggleStrip = gtk.MenuItem( "Strip mode" )
                toggleStrip.connect( "activate", 
                    lambda w: self.thePlotInstance.setStripMode( MODE_STRIP )  )
            theMenu.append( toggleStrip )
            theMenu.append( gtk.SeparatorMenuItem() )   
        theMenu.append( xToggle )
        theMenu.append( yToggle )
        theMenu.append( gtk.SeparatorMenuItem() )
        theMenu.append( guiMenuItem )
        theMenu.show_all()
        theMenu.popup( None, None, None, 1, 0 )
                
    def __adjustWindowHeight ( self, deltaHeight ):
        if self.theOuterFrame != None:
            self.theOuterFrame.resize(
                self.theOuterFrame.allocation.width,
                self.theOuterFrame.allocation.height - deltaHeight )
        
    def setScale( self, theOrientation, theScaleType ):
        """
        sets scale type of the axis defined by orientation
        theOrientation is either "Horizontal" or "Vertical"
        theScaleType is either "Linear" or "Log10"
        """ 
        self.thePlotInstance.changeScale( theOrientation, theScaleType )
 
    def setXAxis( self, theFullPN ):
        """ sets XAxis
        either a FullPN String or "time" literal
        """
        if theFullPN != "Time":
            if theFullPN not in self.thePlotInstance.getDataSeriesNames():
                return
        self.thePlotInstance.setXAxis( theFullPN )
        #switch off trace
        anIter=self.theListStore.get_iter_first()
        while True:
            if anIter == None:
                return None
            aFullPN = identifiers.FullPN(
                self.theListStore.get_value(anIter, COL_TXT ) )
            self.noHandle = True
            aSeries = self.thePlotInstance.getDataSeries( aFullPN )
            self.theListStore.set_value( anIter, COL_ON, aSeries.isOn() )
            self.theListStore.set_value( anIter, COL_X, theFullPN == aFullPN )
            self.noHandle = False
            anIter=self.theListStore.iter_next( anIter )

    def setStripInterval( self, anInterval ):
        """ sets striptinterval of graph to anInterval """
        self.theEntry.set_text( str( anInterval ) )
        self.stripIntervalChanged(None, None )
    
    def showHistory (self):
        """ changes Plot to History mode
            e.g. draws plot from logger information
            will fall back to strip mode if not each and every
            FullPN has a logger
            returns None
        """
        if self.thePlotInstance.getStripMode() != 'history':
            self.toggleStripAction( None )
            
    def showStrip (self):
        """ changes Plot to Strip mode
            e.g. shows the most recent datapaoints
            spanning an interval set by StripInterval
        """
        if self.thePlotInstance.getStripMode() == 'history':
            self.toggleStripAction( None )

    def logAll(self):
        """ creates logger for all traces on TracerWindow """
        self.logAllAction( None )

    def setTraceVisible (self, aFullPN, toDisplay):
        """ sets visible trace of identified by FullPNString 
            toDisplay:
            True - Display
            False - Don't display trace
        """
        if aFullPN not in self.thePlotInstance.getDataSeriesNames():
            return
        aSeries = self.thePlotInstance.getDataSeries( aFullPN )
        currentState = aSeries.isOn()

        if currentState == toDisplay:
            return None
        anIter=self.theListStore.get_iter_first()
        while True:
            if anIter == None:
                return None
            aTitle = identifiers.FullPN(
                self.theListStore.get_value(anIter, COL_TXT ) )
            if aTitle == aFullPN:
                aSeries = self.thePlotInstance.getDataSeries( aFullPN )
                if toDisplay:
                    aSeries.switchOn()
                else:
                    aSeries.switchOff()
                self.noHandle = True
                self.theListStore.set_value( anIter, COL_ON, aSeries.isOn() )
                self.noHandle = False
                self.thePlotInstance.totalRedraw()

                break
            anIter=self.theListStore.iter_next( anIter )

    def zoomIn (self, x0,x1, y0, y1 ):
        """ magnifies a rectangular area of  Plotarea
            bordered by x0,x1,y0,y1
        """
        if x1<0 or x1<=x0 or y1<=y0:
            self.theSession.message("bad arguments")
            return
        self.thePlotInstance.zoomIn( [float(x0), float(x1)], [float(y1), float(y0)])

    def zoomOut(self, aNum = 1):
        """ zooms out aNum level of zoom ins 
        """
        for i in range(0, aNum):
            self.thePlotInstance.zoomOut()

    def checkRun( self ):
        if self.theSession.isRunning():
            # displays a Confirm Window.
            aMessage = "Cannot create new logger, because simulation is running.\n"
            aMessage += "Please stop simulation if you want to create a logger" 
            showPopupMessage( OK_MODE, aMessage, 'Warning' )
            return True
        return False

    def stripIntervalChangedEnter( self, obj ):
        self.stripIntervalChanged( obj, None )

    def stripIntervalChanged(self, obj, event): #this is an event handler again
        """
        this signal handler is called when TAB is presses on entry1
        """
        #get new value
        #call plotterinstance
        try:
            a = float( self.theEntry.get_text() )
        except ValueError:
            self.theSession.message("Enter a valid number, please.")
            self.theEntry.set_text( str( self.thePlotInstance.getstripinterval() ) )
        else:
            self.thePlotInstance.setStripInterval( a )

    def buttonPressedOnList(self, aWidget, anEvent):
        """
        this signal handler is called when mousebutton is pressed over the
        fullpnlist
        """

        if anEvent.button == 3:
        # user menu: remove trace, log all, save data,edit policy, hide ui, change color
            selectedList = self.getSelected()
            allHasLogger = True
            xAxisSelected = False
            xAxis = self.thePlotInstance.getXAxisFullPN()
            aLoggerList = self.theSession.getLoggedPNList()
            for aSelection in selectedList:
                if not aSelection[0] in aLoggerList:
                    allHasLogger = False
                if aSelection[0] == xAxis:
                    xAxisSelected = True
                

            theMenu = gtk.Menu()
            listCount = len( self.displayedFullPNList )
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
            aLoggerStub = self.theSession.createLogger(
                util.createFullPN( fpnList[0][0] ) )
            aLogPolicy = aLoggerStub.getLoggerPolicy()
        else:
            aLogPolicy = [ 1, 0, 0, 0 ]
        newLogPolicy = self.theSession.openLogPolicyWindow( aLogPolicy, "Set log policy for selected loggers" )
        if newLogPolicy == None:
            return
        for anItem in fpnList:
            aFullPNString = anItem[0]
            aLoggerStub = self.theSession.createLogger(
                identifiers.FullPN( aFullPNString ) )
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
            aConfirmMessage = "%s directory already exists.\nWould you like to override it?" % aSaveDirectory
            if showPopupMessage( OKCANCEL_MODE,
                aConfirmMessage,
                'Question' ) == OK_PRESSED:
                pass
            else:
                return None

        # If same directory dose not exists.
        else:
            try:
                os.mkdir(aSaveDirectory)
            except:
                aErrorMessage = 'Cooudl not create %s\n'%aSaveDirectory
                showPopupMessage( OK_MODE, aErrorMessage, 'Error' )
                return None


        try:
            self.theSession.saveLoggerData( selectedFullPNList, aSaveDirectory, -1, -1, -1 )
        except:
            anErrorMessage = "Failed to save the log data"
            showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
            return None
        
    def closeParentWindow( self, obj ):
        aParentWindow = self.theSaveDirectorySelection.cancel_button.get_parent_window()
        aParentWindow.hide()

    def onoffTogglePressed(self,cell, path, model):
        """
        this signal handler is called when an on-off checkbox is pressed over of the fullpnlist
        """
        if self.noHandle:
            return
        iter = model.get_iter( (int( path ), ) )
        aFullPN = identifiers.FullPN( self.theListStore.get_value( iter, COL_TXT ) )
        aSeries = self.thePlotInstance.getDataSeries( aFullPN )
        if aSeries.isOn( ):
            aSeries.switchOff()
        else:
            aSeries.switchOn()
        self.thePlotInstance.totalRedraw()
        self.noHandle = True
        self.theListStore.set_value( iter, COL_ON, aSeries.isOn() )
        self.noHandle = False

    def loggerTickBoxChecked(self, cell, path, model):
        """
        this signal handler is called when create logger checkbox is pressed
        over the fullpnlist
        """
        if self.noHandle:
            return
        iter = model.get_iter( ( int ( path ), ) )
        fixed = model.get_value( iter, COL_LOG )

        if fixed == False:
            if self.checkRun():
                return
            aFullPN = identifiers.FullPN(
                self.theListStore.get_value( iter, COL_TXT ) )
            self.theSession.createLogger( util.createFullPN( text ) )
            self.refreshLoggers()

    def xaxisToggled(self, cell, path, model):
        """
        this signal handler is called when xaxis is toggled
        """
        if self.noHandle:
            return
        iter = model.get_iter( ( int ( path ), ) )
        fixed = model.get_value( iter, COL_X )
        if fixed == False:
            aFullPN = identifiers.FullPN( 
                self.theListStore.get_value( iter, COL_TXT ) )
            self.setXAxis( aFullPN )
        else:
            self.setXAxis( "Time" )

    def removeTraceAction(self, *args ):
        """
        this signal handler is called when "Remove Trace" button is pressed
        """
        #identify selected FullPNs
        fpnlist = []
        for aSelectedFullPN, anIter in self.getSelected():
            # XXX: is the following check really necessary?
            if len( self.displayedFullPNList ) == 1:
                break
            matchedList = []
            if aSelectedFullPN in self.theRawFullPNList:
                self.theRawFullPNList.remove( aSelectedFullPN )
            #remove from displaylist
            self.displayedFullPNList.remove( aSelectedFullPN )
            fpnlist.append( aSelectedFullPN )
            self.theListStore.remove( anIter )
        self.thePlotInstance.removeTrace( fpnlist )

    def logAllAction(self,obj):
        """ this signal handler is called when "Log All" button is pressed """
        if not self.checkRun():
            return
        #creates logger in simulator for all FullPNs 
        self.createLogger( self.displayedFullPNSList )      
        self.refreshLoggers()
        
    def toggleStripAction(self, obj):
        """
        this signal handler is called when "Show History" button is toggled
        """
        #if history, change to strip, try to get data for strip interval
        stripmode = self.thePlotInstance.getStripMode()
        if stripmode == 'history':
            self.thePlotInstance.setStripMode( 'strip' )
        else:
            pass_flag = True
            aLoggerList = self.theSession.getLoggedPNList()
            for aFullPN in self.displayedFullPNList:
                if not aFullPN in aLoggerList:
                    pass_flag = False
                    break
            if pass_flag:
                self.thePlotInstance.setStripMode( 'history' )
            else:
                self.theSession.message("can't change to history mode, because not every trace has logger.\n")

    def getPixmap( self, aColor ):
        if self.thePixmapDict.has_key( aColor ):
            return self.thePixmapDict[ aColor ]

        aColorMap = self.theRootWidget.get_colormap()
        newgc = self.createGC()
        newgc.set_foreground( aColorMap.alloc_color( aColor ) )
        newpm = self.createPixmap( 10, 10 )
        newpm.draw_rectangle( newgc, True, 0, 0, 10, 10 )
        pb = gtk.gdk.Pixbuf( gtk.gdk.COLORSPACE_RGB, True, 8, 10, 10 )
        aPixmap = pb.get_from_drawable(
            newpm, aColorMap,
            0, 0, 0, 0, 10, 10)
        self.thePixmapDict[ aColor ] = aPixmap
        return aPixmap

    def handleSessionEvent( self, event ):
        if event.type == 'simulation_updated':
            self.thePlotInstance.update()


