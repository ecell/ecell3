#!/usr/bin/env python2

import sys
import os
import imp
import glob
from ecell.ECS import *
from config import *

class PluginModule:

    def __init__( self, name, path=PLUGIN_PATH ):
	
        self.theName = name

        aFp, aPath, self.theDescription\
             = imp.find_module( self.theName, PLUGIN_PATH )

        self.theDirectoryName = os.path.dirname( aPath )

        try:
            self.theModule = imp.load_module( self.theName, aFp, aPath, self.theDescription )
        finally:
            # close fp even in exception
            if aFp:
                aFp.close()


    def createInstance( self, data, pluginmanager, root=None, parent=None ):

        aConstructor = self.theModule.__dict__[self.theName]
        anArgumentTuple = ( self.theDirectoryName,  data, pluginmanager, root )
        return apply( aConstructor, anArgumentTuple )
	


class PluginManager:

    def __init__( self, session, loggerwindow, interfacewindow ):

        self.thePluginMap = {}
        self.theInstanceList = []
        self.theSession = session
        self.theLoggerWindow = loggerwindow
	self.theInterfaceWindow = interfacewindow
        
    def createInstance( self, classname, data, root=None, parent=None ):
	
        try:
            aPlugin = self.thePluginMap[ classname ]
	    
        except KeyError:
            self.loadModule( classname )

        if root !='top_vbox':
            self.theSession.theSimulator.record( 'aPluginManager.createInstance( \'%s\', %s )' % (classname, data) )
	    self.theInterfaceWindow.addNewRecord( classname, data )
	
        anInstance = aPlugin.createInstance( data, self, root, parent )
        self.theSession.theSimulator.initialize()
        return anInstance


    def loadModule( self, classname ):
	
        aPlugin = PluginModule( classname )
        self.thePluginMap[ classname ] = aPlugin

    def loadAll( self ):
	
        for aPath in PLUGIN_PATH:
            aFileList = glob.glob( os.path.join( aPath, '*.glade' ) )
            for aFile in aFileList:
                aModulePath = os.path.splitext( aFile )[0]
                if( os.path.isfile( aModulePath + '.py' ) ):
                    aModuleName = os.path.basename( aModulePath )
                    self.loadModule( aModuleName )
                    self.theInterfaceWindow.thePluginWindowsNoDict[ aModuleName[ : -6 ] ] = 0

    def updateAllPluginWindow( self ):
	
        for anInstance in self.theInstanceList:
            anInstance.update()

    def appendInstance( self, instance ):
        
        self.theInstanceList.append( instance )
	
    def removeInstance( self, instance ):
        
        self.theInstanceList.remove( instance )

    def updateLoggerWindow( self ):
        
        self.theLoggerWindow.update()
	
    def showPlugin( self, num, obj ):

        anInstance = self.theInstanceList[ num + 1 ]
	anInstance.getWidget( anInstance.__class__.__name__ ).hide()	
	anInstance.getWidget( anInstance.__class__.__name__ ).show_all()

    def getModule( self, aNum, aTitle ):
        
	self.theInstanceList[ aNum + 1 ].editTitle( aTitle )
	
    def deleteModule( self, num, obj ):
        
	anInstance = self.theInstanceList[ num + 1 ]
	anInstance.getWidget( anInstance.__class__.__name__ ).destroy()
        
if __name__ == "__main__":
    pass
