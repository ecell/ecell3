

import sys
import os
import imp

from config import *

class PluginWindowManager:

    def __init__( self ):
        self.theModuleMap = {}
        self.theWindowList = []

    def createWindow( self, name, parent=None ):
        try:
            aModule = self.theModuleMap[ name ]
        except KeyError:
            aModule = PluginWindowModule( name )
            self.theModuleMap[ name ] = aModule

        aWindow = aModule.createWindow()
        self.appendWindow( aWindow )

        return aWindow

    def appendWindow( self, aWindow ):
        self.theWindowList.append( aWindow )

    def removeWindow( self, aWindow ):
        self.theWindowList.remove( aWindow )

    def getWindowList( self ):
        return self.theWindowList

        

class PluginWindowModule:

    def __init__( self, name, path=PLUGIN_PATH ):

        self.theName = name

        # check if it's already loaded
        try:
            return sys.modules[name]
        except KeyError:
            pass

        aFp, aPath, self.theDescription\
             = imp.find_module( self.theName, PLUGIN_PATH )
        
        try:
            self.theModule = imp.load_module( self.theName,
                                              aFp,
                                              aPath,
                                              self.theDescription )
        finally:
            # close fp even in exception
            if aFp:
                aFp.close()

        self.theGladefileName, dotpy = os.path.splitext( aPath )
        self.theGladefileName += '.glade'

        #FIXME: check if there is createWindow() in this module


    def createWindow( self, parent = None ):
        self.theModule.__dict__[self.theName]( self.theGladefileName )

