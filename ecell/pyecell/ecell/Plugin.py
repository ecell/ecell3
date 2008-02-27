#!/usr/bin/env python
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

__all__ = (
    'PluginModule',
    'PluginManager'
    )

class PluginModule:
    """
    PluginModule 
      - creates an instance of plugin module
    """
    def __init__( self, aModuleName, aModulePath, aPluginManager ):
        """
        Constructor
          - loads module file
        
        aModuleName : module name
        aModulePath : module path
        return -> None
        This method can throw an exception.
        """
        self.theName = aModuleName
        aFp, aPath, self.theDescription = \
            imp.find_module( self.theName, aModulePath )
        self.theDirectoryName = os.path.dirname( aPath )
        self.thePluginManager = aPluginManager
        try:
            self.theModule = imp.load_module(
                self.theName, aFp, aPath, self.theDescription )
        finally:
            # close fp even in exception
            if aFp:
                aFp.close()

    def getClass( self ):
        return getattr( self.theModule, self.theName )

    def createInstance( self, aModuleName, **options ):
        """
        createInstance
        Creates instance of module
        
        aModuleName: module name
        return -> the result of apply function
        This method can throw an exception.
        """
        aClass = self.getClass()
        instance = aClass(
            self.theDirectoryName, aModuleName, self.thePluginManager,
            **options )
        return instance

class PluginManager:
    """
    PluginManager 
    - manages all plugin modules
    """

    def __init__( self, aPluginPath ):
        """
        Constructor
         - initializes pluginmap and instancelist
        """ 
        self.thePluginMap = {}
        self.thePluginPath = aPluginPath

    def loadModule( self, aClassName ):
        """
        loadModule
          - loads a module
        
        aClassName     : class name
        return -> None 
        """
        if self.thePluginMap.has_key( aClassName ):
            return self.thePluginMap[ aClassName ]
        aPlugin = PluginModule( aClassName, self.thePluginPath, self )
        self.thePluginMap[ aClassName ] = aPlugin
        return aPlugin

    def getLoadedModules( self ):
        return dict( self.thePluginMap )

    def isModuleLoaded( self, aClassName ):
        return self.thePluginMap.has_key( aClassName )
