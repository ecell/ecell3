

import imp

from config import *

class PluginWindowMaker:

    thePluginMap = {}

    def createWindow( name, parent=None ):
        try:
            aModule = thePluginMap[ name ]
        except KeyError:
            aModule = PluginWindowModule( name )
            thePluginMap[ name ] = aModule

        return aModule.createWindow()

class PluginWindowModule:

    def __init__( name, path=PLUGIN_PATH ):

        self.theName = name

        # check if it's already loaded
        try:
            return sys.modules[name]
        except KeyError:
            pass

        self.theFp, self.thePath, self.theDescription\
                    = imp.find_module( name, PLUGIN_PATH )
        
        try:
            self.theModule = imp.load_module( name, fp, path, description )
        finally:
            # error, close fp
            if fp:
                fp.close()

        #FIXME: check if there is createWindow() in this module


        def createWindow( parent = None ):
            theModule.createWindow( self.name, self.thePath, parent )

