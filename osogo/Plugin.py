#!/usr/bin/env python2

import sys
import os
import imp
import glob

from config import *

class PluginModule:

    def __init__( self, name, path=PLUGIN_PATH ):

        self.theName = name

        aFp, aPath, self.theDescription\
             = imp.find_module( self.theName, PLUGIN_PATH )

        self.theDirectoryName = os.path.dirname( aPath )
        
        try:
            self.theModule = imp.load_module( self.theName,
                                              aFp,
                                              aPath,
                                              self.theDescription )
        finally:
            # close fp even in exception
            if aFp:
                aFp.close()


    def createInstance( self, data, pluginmanager, root=None, parent=None ):
        aConstructor = self.theModule.__dict__[self.theName]
        anArgumentTuple = ( self.theDirectoryName,  data, pluginmanager, root )
        return apply( aConstructor, anArgumentTuple )
        


class PluginManager:

    def __init__( self, aMainWindow ):
        self.thePluginMap = {}
        self.theInstanceList = []
        self.theMainWindow = aMainWindow

    def createInstance( self, classname, data, root=None, parent=None ):
        try:
            aPlugin = self.thePluginMap[ classname ]
        except KeyError:
            self.loadModule( classname )


        anInstance = aPlugin.createInstance( data, self, root, parent )
        self.appendInstance( anInstance )

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

    def printMessage( self, aMessageString ):
        self.theMainWindow.printMessage( aMessageString )

    def printProperty( self, fullpn ):
        self.theMainWindow.printProperty( fullpn )
    
    def printAllProperties( self, fullid ):
        self.theMainWindow.printAllProperties( fullid )

    def printList( self, primitivetype, systempath, list ):
        self.theMainWindow.printList( primitive, systempath, list )

    def updateAllPluginWindow( self ):
        for anInstance in self.theInstanceList:
            anInstance.update()

    def appendInstance( self, instance ):
        self.theInstanceList.append( instance )

    def removeInstance( self, instance ):
        self.theInstanceList.remove( instance )

        
if __name__ == "__main__":
    pass
