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
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
#           'Masahiro Sugimoto <sugi@bioinformatics.org>' 
#           'Gabor Bereczki <gabor.bereczki@talk21.com>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import os
import sys
import gtk
import gobject 
import thread
from sets import Set
import re

import config
from constants import *

from ecell.Session import *
from ecell.ModelWalker import *
import ecell.util as util

from DataGenerator import DataGenerator
from MainWindow import MainWindow
from LoggerWindow import LoggerWindow
from InterfaceWindow import InterfaceWindow
from StepperWindow import StepperWindow
from BoardWindow import BoardWindow
from EntityListWindow import EntityListWindow
from OsogoWindow import OsogoWindow
from ConfigParser import ConfigParser
from OsogoPluginManager import OsogoPluginManager

import traceback

class GtkSessionMonitor:
    def __init__( self ):
        self.theAliveSessions = Set()

    def beginSession( self ):
        aSession = Session()
        aSessionFacade = GtkSessionFacade( self, aSession )
        self.theAliveSessions.add( aSessionFacade )
        return aSessionFacade

    def closeSession( self, aSession ):
        if aSession in self.theAliveSessions:
            self.theAliveSessions.remove( aSession )
            if len( self.theAliveSessions ) == 0:
                self.quitEventLoop()

    def enterEventLoop( self ):
        gobject.threads_init()
        gtk.main()

    def quitEventLoop( self ):
        self.theAliveSessions.clear()
        gtk.main_quit()

class GtkSessionEvent:
    def __init__( self, type, options ):
        self.type = type
        self.options = options

    def __getattr__( self, key ):
        if not self.options.has_key( key ):
            raise AttributeError( key )
        return self.options[ key ]

    def __setattr__( self, key, val ):
        if key in ('type', 'options'):
            self.__dict__[ key ] = val
        else:
            self.options[ key ] = val

class GtkSessionFacade:
    def findUserPrefsDir( self ):
        if not self.theUserPreferencesDir:
            path_list = (
                config.user_prefs_dir,
                config.home_dir
                )
            for parent_dir in path_list:
                path = os.path.join( parent_dir, CONFIG_FILE_NAME )
                if os.path.isfile( path ):
                    self.theUserPreferencesDir = parent_dir
            self.theUserPreferencesDir = path_list[0]
        return self.theUserPreferencesDir

    def findIniFile( self ):
        path_list = ( self.findUserPrefsDir(), ) + (
            config.conf_dir,
            config.lib_dir
            )
        for parent_dir in path_list:
            path = os.path.join( parent_dir, CONFIG_FILE_NAME )
            if os.path.isfile( path ):
                return path
        return None

    def __init__( self, aSessionMonitor, aSession ):
        """sets up the osogo session, creates Mainwindow and other fundamental
        windows but doesn't show them"""

        self.theSessionMonitor = aSessionMonitor
        self.theSession = aSession
        self.theModelWalker = None
        self.theMessageListenerSet = Set()
        self.theManagedWindowSet = Set()

        self.theSession.setMessageMethod( self.fireMessageEvent )

        self.theUserPreferencesDir = None

        # reads defaults from osogo.ini 
        aIniFileName = self.findIniFile()
        aConfigDB = ConfigParser()
        if aIniFileName != None:
            aConfigDB.read( aIniFileName )
        self.theConfigDB = aConfigDB
        self.theIniFileName = aIniFileName

        self.theUpdateInterval = 2000
        self.stuckRequests = 0
        self.theStepSizeOrSec = 1.0
        self.theRunningFlag = False
        self.theTimer = None

        # creates PluginManager
        self.thePluginPath = config.plugin_path
        self.thePluginManager = OsogoPluginManager( self.thePluginPath, self )
        self.theDataGenerator = DataGenerator( self.theSession )
        self.thePluginInstanceList = []
        self.thePluginWindowTitleMap = {}
        self.thePluginWindowPerTypeMap = {}
        self.theTitleFormat = "%s (%d)"
        # key is instance, value is None
        self.thePropertyWindowOnEntityListWindows = {}

        # key:window name(str) value:window instance
        self.theFundamentalWindows = {}

        # key:EntityListWindow instance value:instance
        # In deleteEntityListWindow method, an instance of EntityListWindow is
        # accessed directory. The sequence information of EntityListWindow does
        # not need. So the references to EntityListWindow instances should be 
        # held dict's key. Values of dict are not also imported.

        self.theManagedWindowSet = Set()

        # initializes for run method 
        #self.theSession.setEventChecker( gtk.events_pending )
        #self.theSession.setEventHandler( gtk.main_iteration )

        self.thePluginManager.loadAllPlugins()

        self.theMainWindow = self.createManagedWindow( MainWindow )
        self.theFundamentalWindows['MainWindow'] = self.theMainWindow

        for aWindow in self.theFundamentalWindows.values():
            aWindow.initUI()

        self.theMainWindow.show()

    def __del__( self ):
        for anInstance in self.theManagedWindowSet:
            anInstance.destroy()
        for anInstance in self.thePluginInstanceList:
            anInstance.destroy()

    def terminate( self ):
        self.theSession = None
        self.theSessionMonitor.closeSession( self )

    def openFundamentalWindow( self, aWindowName ):
        """opens up window and returns aWindowname instance
        aWindowName   ---  Window name (str)
        Returns FundamentalWindow or EntityListWindow list
        """
        if aWindowName != 'MainWindow' and not self.theSession.isModelLoaded():
            self.message( "Model has not yet been loaded. Can't open windows." )
            return None
        # When the WindowName does not match, create nothing.
        if self.theFundamentalWindows.has_key( aWindowName ):
            aWindow = self.theFundamentalWindows[ aWindowName ]
        else:
            self.message( "No such WindowType (%s) " %aWindowName )
            return None

        aWindow.show()

        if type( aWindow ) == OsogoWindow:
            aWindow.present()
        return aWindow

    def getFundamentalWindow( self, aWindowName ):
        """
        aWindowName   ---  Window name (str)
        Returns FundamentalWindow
        """

        # check fundamentalwindows
        if self.theFundamentalWindows.has_key( aWindowName ):
            return self.theFundamentalWindows[ aWindowName ]
        return None

    def displayWindow( self, aWindowName ):
        """When the Window is not created, calls its openWidow() method.
        When already created, move it to the top of desktop.
        aWindowName   ---  window name(str)
        Return None
        [None]:When the WindowName does not matched, creates nothing.
        """
        self.setFundamentalWindowVisibility( aWindowName, True )

    def isFundamentalWindowShown( self, aWindowName ):
        return self.theFundamentalWindows.has_key( aWindowName )

    def setFundamentalWindowVisibility( self, aWindowName, visible ):
        if visible:
            if not self.theFundamentalWindows.has_key( aWindowName ):
                aWindow = self.openManagedWindow( globals()[ aWindowName ] )
                def onFundamentalWindowDestroyed( aWindow ):
                    del self.theFundamentalWindows[ aWindowName ]
                    self.fireEvent(
                        'fundamental_window_destroyed',
                        window_name = aWindowName )
                aWindow.registerDestroyHandler( onFundamentalWindowDestroyed )
                self.theFundamentalWindows[ aWindowName ] = aWindow
            else:
                aWindow = self.theFundamentalWindows[ aWindowName ]
            aWindow.show()
            aWindow.present()
            self.fireEvent(
                'fundamental_window_created',
                window_name = aWindowName )
        else:
            if self.theFundamentalWindows.has_key( aWindowName ):
                self.theFundamentalWindows[ aWindowName ].destroy()

    def onPluginWindowTitleChanging( self, aPluginWindow, aNewTitle ):
        return not self.thePluginWindowTitleMap.has_key( aNewTitle )

    def onPluginWindowTitleChanged( self, aPluginWindow, aOldTitle ):
        if self.thePluginWindowTitleMap.has_key( aOldTitle ):
            del self.thePluginWindowTitleMap[ aOldTitle ]
        self.thePluginWindowTitleMap[ aPluginWindow.getTitle() ] = aPluginWindow
        self.fireEvent( 'plugin_window_title_changed', 
                        instance = aPluginWindow, old_title = aOldTitle )

    def createPluginWindow( self, aType, aFullPNList ):
        """
        opens and returns _PluginWindow instance of aType showing aFullPNList 
        returns None if pluginwindow could not have been created
        """
        anInstance = self.instantiatePlugin( aType, aFullPNList )
        anInstance.registerDestroyHandler( self.removePluginInstance )
        self.manageWindow( anInstance )
        return anInstance

    def openPluginWindow( self, aType, aFullPNList ):
        """
        opens and returns _PluginWindow instance of aType showing aFullPNList 
        returns None if pluginwindow could not have been created
        """
        plugin = self.loadModule( aType )
        anInstanceClass = plugin.getClass()

        if not self.thePluginWindowPerTypeMap.has_key( anInstanceClass ):
            aPluginWindowSet = Set()
            self.thePluginWindowPerTypeMap[ anInstanceClass ] = aPluginWindowSet
        else:
            aPluginWindowSet = self.thePluginWindowPerTypeMap[ anInstanceClass ]
        if anInstanceClass.theViewType == MULTIPLE:
            anInstance = plugin.createInstance( aFullPNList )
            aSerialNum = 1
            while True:
                aTitle = self.theTitleFormat % (
                    anInstance.getName(), aSerialNum )
                if anInstance.setTitle( aTitle ):
                    break
                aSerialNum += 1
        else:
            if len( aPluginWindowSet ) > 0:
                return None
            anInstance = plugin.createInstance( aFullPNList )
            assert anInstance.setTitle( anInstance.getName() )

        aPluginWindowSet.add( anInstance )
        anInstance.registerDestroyHandler( self.removePluginInstance )
        self.thePluginInstanceList.append( anInstance )
        self.fireEvent( 'plugin_instance_created', instance = anInstance )
        self.manageWindow( anInstance )
        anInstance.initUI()
        anInstance.setParent( None )
        anInstance.show()
        return anInstance

    def manageWindow( self, anInstance ):
        self.fireEvent( 'window_managed', window = anInstance )
        anInstance.registerDestroyHandler( self.unmanageWindow )
        self.theManagedWindowSet.add( anInstance )

    def unmanageWindow( self, anInstance ):
        self.theManagedWindowSet.discard( anInstance )
        self.fireEvent( 'window_unmanaged', window = anInstance )

    def createPluginOnBoard(self, aType, aFullPNList):    
        """ creates and adds plugin to pluginwindow and returns plugininstance """
        aBoardWindow = self.getWindow( 'BoardWindow' )
        if aBoardWindow == None:
            self.message('Board Window does not exist. Plugin cannot be added.')
            return None
        return aBoardWindow.addPluginWindows( aType, aFullPNList )

    def createManagedWindow( self, aManagedWindowType ):
        aManagedWindow = aManagedWindowType()
        aManagedWindow.setSession( self )
        self.manageWindow( aManagedWindow )
        return aManagedWindow

    def openManagedWindow( self, aManagedWindowType ):
        aManagedWindow = self.createManagedWindow( aManagedWindowType )
        aManagedWindow.initUI()
        aManagedWindow.setParent( None )
        aManagedWindow.show()
        return aManagedWindow

    def getMainWindow( self ):
        return self.theMainWindow

    def getStatusBar( self ):
        assert self.theMainWindow != None
        return self.theMainWindow.getStatusBar()

    def fireMessageEvent( self, aMessage ):
        print aMessage
        self.fireEvent( 'message', content = aMessage )

    def fireEvent( self, type, **options ):
        def broadcaster():
            # print type
            event = GtkSessionEvent( type, options )
            if event.type in ( 'simulation_updated', 'simulation_started', \
                               'simulation_stopped' ):
                self.theDataGenerator.update()
            for aManagedWindow in self.theManagedWindowSet:
                aManagedWindow.handleSessionEvent( event )
        gobject.idle_add( broadcaster )

    def updateUI( self ):
        # updates all entity list windows
        for aManagedWindow in self.theManagedWindowSet:
            aManagedWindow.update()
        # updates all plugin windows
        for aPluginWindow in self.thePluginInstanceList:
            aPluginWindow.update()

    def setUpdateInterval(self, secs):
        "plugins are refreshed every secs seconds"
        self.theUpdateInterval = secs
        self.fireEvent( 'update_interval_changed', interval = secs )
    
    def getUpdateInterval( self ):
        "returns the rate by plugins are refreshed "
        return self.theUpdateInterval

    def getParameter(self, aParameter):
        """
        tries to get a parameter from ConfigDB
        if the param is not present in either osogo or default section
        raises exception and quits
        """

        # first try to get it from osogo section
        if self.theConfigDB.has_section('osogo'):
            if self.theConfigDB.has_option('osogo',aParameter):
                return self.theConfigDB.get('osogo',aParameter)

        # gets it from default
        return self.theConfigDB.get('DEFAULT',aParameter)

    def setParameter(self, aParameter, aValue):
        """
        tries to set a parameter in ConfigDB
        if the param is not present in either osogo or default section
        raises exception and quits
        """
        # first try to set it in osogo section
        if self.theConfigDB.has_section('osogo'):
            if self.theConfigDB.has_option('osogo',aParameter):
                self.theConfigDB.set('osogo',aParameter, str(aValue))
        else:

            # sets it in default
            self.theConfigDB.set('DEFAULT',aParameter, str(aValue))

    def saveParameters( self ):
        """
        tries to save all parameters into a config file in home directory
        """
        try:
            fp = open(
                os.path.join(
                    self.theUserPreferencesDir,
                    CONFIG_FILE_NAME
                    ), 'w' )
            self.theConfigDB.write( fp )
        except:
            self.message("Failed to save preferences into file %s. Please check permissions for home directory." % self.theIniFileName)

    def getLogPolicyParameters( self ):
        """
        gets logging policy from config database
        """
        logPolicy = []
        logPolicy.append ( int( self.getParameter( 'logger_min_step' ) ) )
        logPolicy.append ( float ( self.getParameter( 'logger_min_interval' ) ) )
        logPolicy.append ( int( self.getParameter( 'end_policy' ) ) )
        logPolicy.append ( int (self.getParameter( 'available_space' ) ) )
        if logPolicy[0]<=0 and logPolicy[1]<=0:
            logPolicy[0]=1
        return logPolicy

    def setLogPolicyParameters( self, logPolicy ):
        """
        saves logging policy into config database
        """
        self.setParameter( 'logger_min_step', logPolicy[0] )
        self.setParameter( 'logger_min_interval', logPolicy[1] ) 
        self.setParameter( 'end_policy' , logPolicy[2] )
        self.setParameter( 'available_space' ,logPolicy[3] )
        self.saveParameters()

    def isModelLoaded( self ):
        return self.theSession.isModelLoaded()

    def loadScript( self, ecs, parameters={} ):
        self.theSession.loadScript( ecs, parameters )
        self.fireEvent( 'script_loaded' )

    def loadModel( self, aFileName ):
        assert not self.theSession.isModelLoaded()
        self.theSession.loadModel( aFileName )
        self.theModelWalker = ModelWalker( self.theSession.theSimulator )
        self.fireEvent( 'model_loaded' )

    def saveModel( self, aFileName ):
        self.theSession.saveModel( aFileName )
        self.fireEvent( 'model_saved' )

    def message( self, message ):
        self.theSession.message( message )

    def run( self , time = None ):
        """ 
        if already running: do nothing
        if time is given, run for the given time
        if time is not given:
            if Mainwindow is not opened create a stop button
            set up a timeout rutin and Running Flag 
        """
        if self.theRunningFlag == True:
            return
        try:
            self.theRunningFlag = True
            aCurrentTime = self.getCurrentTime()
            self.message( "%15s" % aCurrentTime + ":Start" )

            def thread_proc(): 
                self.fireEvent( 'simulation_started' )
                if time:
                    self.theSession.run( time )
                else:
                    aCounter = 0
                    while self.theRunningFlag:
                        self.theSession.step()
                        aCounter += 1
                        if ( aCounter > self.theUpdateInterval ):
                            self.fireEvent( 'simulation_updated' )
                            aCounter = 0
                self.fireEvent( 'simulation_stopped' )

            thread.start_new_thread( thread_proc, () )
        except:
            self.theRunningFlag = False
            anErrorMessage = traceback.format_exception(
                sys.exc_type, sys.exc_value, sys.exc_traceback )
            self.message(anErrorMessage)

    def stop( self ):
        """
        stop Simulation, remove timeout, set Running flag to false
        """
        if self.theRunningFlag == True:
            self.theRunningFlag = False
            self.theSession.stop()
            aCurrentTime = self.getCurrentTime()
            self.message( "%15s" % aCurrentTime + ":Stop" )

    def step( self, num = None ):
        """
        step according to num, if num is not given,
        according to set step parameters
        """
        if self.theRunningFlag:
            return

        if num == None:
            #set it to 1
                num = 1
                self.message( "Zero step value overridden to 1" )

        try:
            self.theRunningFlag = True

            self.message( "Step" )
            self.fireEvent( 'simulation_started' )
            self.theSession.step( int( num ) )
            self.fireEvent( 'simulation_updated' )
            self.fireEvent( 'simulation_stopped' )

            self.theRunningFlag = False
        except:
            anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
            self.message( anErrorMessage )

    def isRunning(self):
        return self.theRunningFlag

    def getModelWalker( self ):
        return self.theModelWalker

    def getNextEvent( self ):
        return self.theSession.getNextEvent()

    def setEventChecker( self, event ):
        self.theSession.setEventChecker( event )

    def setEventHandler( self, event ):
        self.theSession.setEventHandler( event )

    def getStepperList( self ):
        return self.theSession.getStepperList()

    def createStepperStub( self, id ):
        return self.theSession.createStepperStub( id )

    def getEntityList( self, entityType, systemPath ):
        return self.theSession.getEntityList( entityType, systemPath )

    def createEntityStub( self, fullid ):
        return self.theSession.createEntityStub( fullid )

    def getLoggerList( self ):
        return self.theSession.getLoggerList()

    def createLogger( self, aFullPN ):
        # XXX: remember refresh Tracer and Loggerwindows!!!
        aFullPNString = util.createFullPNString( aFullPN )
        aStub = self.theSession.createLoggerStub( aFullPNString )
        if not aStub.exists():
            aStub.create()
            aStub.setLoggerPolicy( self.getLogPolicyParameters() )
            self.fireEvent( 'logger_created', logger = aStub )
        return aStub

    def saveLoggerData( self, fullpn=0, aSaveDirectory='./Data', aStartTime=-1, anEndTime=-1, anInterval=-1 ):
        self.theSession.saveLoggerData( fullpn, aSaveDirectory, aStartTime, anEndTime, anInterval )

    def getDataGenerator( self ):
        return self.theDataGenerator 

    def loadModule( self, aClassName ):
        pluginLoadedAhead = self.thePluginManager.isModuleLoaded( aClassName )
        aPlugin = self.thePluginManager.loadModule( aClassName )
        if not pluginLoadedAhead:
            self.fireEvent(
                'module_loaded',
                class_name = aClassName )
        return aPlugin

    def getLoadedModules( self ):
        return self.thePluginManager.getLoadedModules()

    def getPluginInstanceList( self ):
        return list( self.thePluginInstanceList )

    def instantiatePlugin( self, classname, data ):
        """
        classname  --- a class name of PluginWindow (str)
        data       --- a RawFullPN (RawFullPN)
        """
        # XXX: should be checked earlier in the call chain
        # if len(data) == 0:
        #    self.theSession.message("Nothing is selected.")

        plugin = self.loadModule( classname )
        try:
            instance = plugin.createInstance( data )
        except TypeError:
            # XXX: Should be caught in an upper frame
            self.message( string.join(
                traceback.format_exception(
                    sys.exc_type,sys.exc_value,
                    sys.exc_traceback ), '\n' ) )
            return None
        self.thePluginInstanceList.append( instance )
        self.fireEvent(
            'plugin_instance_created', instance = instance )
        return instance

    def findPluginInstanceByTitle( self, aTitle ):
        if self.thePluginWindowTitleMap.has_key( aTitle ):
            return self.thePluginWindowTitleMap[ aTitle ]
        return None

    def removePluginInstance( self, anInstance ):
        try:
            anInstanceClass = anInstance.__class__
            if self.thePluginWindowPerTypeMap.has_key( anInstanceClass ):
                self.thePluginWindowPerTypeMap[ anInstanceClass ].discard(
                    anInstance )
            if self.thePluginWindowTitleMap.has_key( anInstance.getTitle() ):
                del self.thePluginWindowTitleMap[ anInstance.getTitle() ]
            del self.thePluginInstanceList[
                self.thePluginInstanceList.index( anInstance ) ]
            self.fireEvent(
                'plugin_instance_removed', instance = anInstance )
        except ValueError:
            self.message( str( anInstance ) + " is already removed" )

    def getEntityPropertyAttributes( self, aFullPN ):
        return self.theSession.theSimulator.getEntityPropertyAttributes(
            util.createFullPNString( aFullPN ) )

    def setEntityProperty( self, aFullPN, aValue ):
        aFullPNString = util.createFullPNString( aFullPN )
        anAttribute = self.theSession.theSimulator.getEntityPropertyAttributes(
            aFullPNString )

        if anAttribute[ SETTABLE ]:
            self.theSession.theSimulator.setEntityProperty(
                aFullPNString, aValue )
            self.theSession.fireEvent( 'entity_property_changed',
                fullPN = fullPN, value = aValue )
        else:
            self.message('%s is not settable' % aFullPNString )

    def getEntityProperty( self, aFullPN ):
        return self.theSession.theSimulator.getEntityProperty(
            util.createFullPNString( aFullPN ) )

    def createEntityStub( self, aFullID ):
        return self.theSession.createEntityStub(
            util.createFullIDString( aFullID ))

    def createStepperStub( self, aStepperID ):
        return StepperStub( self.theSession.theSimulator, aStepperID )

    def getEntityClassName( self, aFullID ):
        return self.theSession.theSimulator.getEntityClassName(
            util.createFullIDString( aFullID ) )

    def getCurrentTime( self ):
        return self.theSession.getCurrentTime()


