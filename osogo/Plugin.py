#!/usr/bin/env python2

import sys
import os
import imp

from config import *
from ViewWindow import *

class PluginWindow(ViewWindow):

    def __init__( self, dirname, data ):
        aGladeFileName = os.path.join( dirname ,
                                       self.__class__.__name__ + ".glade" )
        ViewWindow.__init__( self, aGladeFileName, data )



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


    def createInstance( self, data ):
        aConstructor = self.theModule.__dict__[self.theName]
        anArgumentTuple = tuple( self.theDirectoryName ) + data
        apply( aConstructor, anArgumentTuple )



class PluginManager:

    def __init__( self ):
        self.thePluginMap = {}
        self.theInstanceList = []

    def createInstance( self, name, parent=None ):
        try:
            aPlugin = self.thePluginMap[ name ]
        except KeyError:
            aPlugin = PluginModule( name )
            self.thePluginMap[ name ] = aPlugin

        anInstance = aPlugin.createInstance()
        self.appendInstance( anInstance )

        return anInstance

    def appendInstance( self, instance ):
        self.theInstanceList.append( instance )

    def removeInstance( self, instance ):
        self.theInstanceList.remove( instance )

        
if __name__ == "__main__":
    pass
