#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
# 'Yuusuke Saito'
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import sys
import os
import imp
import glob
from ecell.ecs_constants import *

# ---------------------------------------------------------------
# PluginModule 
#   - creates an instance of plugin module
# ---------------------------------------------------------------
class PluginModule:


    # ---------------------------------------------------------------
    # Constructor
    #   - loads module file
    #
    # aModuleName : module name
    # aModulePath : module path
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __init__( self, aModuleName, aModulePath ):
        self.theName = aModuleName
        aFp, aPath, self.theDescription\
            = imp.find_module( self.theName, aModulePath )

        self.theDirectoryName = os.path.dirname( aPath )

        try:
            self.theModule = imp.load_module( self.theName, aFp, aPath, self.theDescription )
        finally:
            # close fp even in exception
            if aFp:
                aFp.close()

    # end of __init__


    # ---------------------------------------------------------------
    # createInstance
    # Creates instance of module
    #
    # data          : module name
    # pluginManager : plugin namager
    # root          : None or 'menu'
    # return -> the result of apply function
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def createInstance( self, data, pluginManager, rootWidget=None, parent = None ):

        aConstructor = self.theModule.__dict__[self.theName]
        anArgumentTuple = ( self.theDirectoryName,  data, pluginManager, rootWidget )
        instance = apply( aConstructor, anArgumentTuple )
        instance.theParent = parent
        return instance
    
    # end of createInstance


# end of PluginModule

# ---------------------------------------------------------------
# PluginManager 
#   - manages all plugin modules
# ---------------------------------------------------------------
class PluginManager:

    # ---------------------------------------------------------------
    # Constructor
    #  - initializes pluginmap and instancelist
    #
    # ---------------------------------------------------------------
    def __init__( self, pluginPath ):
        self.thePluginPath = pluginPath
        self.thePluginMap = {}
        self.theInstanceList = []

    # end of __init__
        

    # ---------------------------------------------------------------
    # Creates instance of module
    #
    # classname     : class name
    # data          : data 
    # rootWidget    : None or the name of the root Widget
    # parent        : parent window
    # return -> one instance
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def createInstance( self, classname, data, rootWidget=None, parent=None ):
    
        try:
            aPlugin = self.thePluginMap[ classname ]
        
        except KeyError:
            self.loadModule( classname )

        anInstance = aPlugin.createInstance( data, self, rootWidget, parent )
        return anInstance

    # end of createInstance


    # ---------------------------------------------------------------
    # loadModule
    #   - loads a module
    #
    # aClassName     : class name
    # return -> None 
    # ---------------------------------------------------------------
    def loadModule( self, aClassname ):

        aPlugin = PluginModule( aClassname, self.thePluginPath )
        self.thePluginMap[ aClassname ] = aPlugin

    # end of loadModule


    # ---------------------------------------------------------------
    # loadAll
    #   - loads all modules
    #
    # return -> None 
    # ---------------------------------------------------------------
    def loadAll( self ):
    
        for aPath in self.thePluginPath:
            aFileList = glob.glob( os.path.join( aPath, '*.glade' ) )
            for aFile in aFileList:
                aModulePath = os.path.splitext( aFile )[0]
                if( os.path.isfile( aModulePath + '.py' ) ):
                    aModuleName = os.path.basename( aModulePath )
                    self.loadModule( aModuleName )


    # ---------------------------------------------------------------
    # updateAllPluginWindow
    #   - updates all plugin window
    #
    # return -> None 
    # ---------------------------------------------------------------
    def updateAllPluginWindow( self ):
    
        for anInstance in self.theInstanceList:

            anInstance.update()

    # end of updateAllPluginWindow


    # ---------------------------------------------------------------
    # appendInstance
    #   - appends an instance to instance list
    #
    # aInstance     : an instance
    # return -> None 
    # ---------------------------------------------------------------
    def appendInstance( self, instance ):

        self.theInstanceList.append( instance )


    # ---------------------------------------------------------------
    # removeInstance
    #   - removes an instance from instance list
    #
    # anInstance     : an instance
    # return -> None 
    # This method is throwable exception. (ValueError)
    # ---------------------------------------------------------------
    def removeInstance( self, anInstance ):
        
        try:
            self.theInstanceList.remove( anInstance )
        except:
            pass

    # end of removeInstance

    
    # ---------------------------------------------------------------
    # showPlugin
    #   - shows plugin window
    #
    # anIndex       : an index of module
    # *Objects     : dammy elements of argument
    # aInstance     : an instance
    # return -> None 
    # This method is throwable exception. (IndexError)
    # ---------------------------------------------------------------
    def showPlugin( self, aPluginInstance ):

        #aPluginInstance[ aPluginInstance.__class__.__name__ ].hide()	
        aPluginInstance[ aPluginInstance.__class__.__name__ ].show_all()
        aPluginInstance[ aPluginInstance.__class__.__name__ ].present()

    # end of showPlugin


    # ---------------------------------------------------------------
    # editModuleTitle
    #   - edits module title
    #
    # aPluginInstance    : instance that will be removed
    # aTitle              : title of instance
    #
    # return -> None 
    # This method is throwable exception. (IndexError)
    # ---------------------------------------------------------------
    def editModuleTitle( self, aPluginInstance, aTitle ):
        
        aPluginInstance.editTitle( aTitle )
    
    # end of getModule


    # ---------------------------------------------------------------
    # deleteModule
    #   - deletes a module
    #
    # anIndex     : index of instance
    # aPluginInstance    : instance that will be removed
    # *Object     : dammy elements of argument
    # return -> None 
    # This method is throwable exception. (IndexError)
    # ---------------------------------------------------------------
    def deleteModule( self, anIndex, *Objects ):

        anInstance = self.theInstanceList[ anIndex + 1 ]
        anInstance.getWidget( anInstance.__class__.__name__ ).destroy()

    # end of deleteModule


# end of PluginManager
        
if __name__ == "__main__":
    pass










