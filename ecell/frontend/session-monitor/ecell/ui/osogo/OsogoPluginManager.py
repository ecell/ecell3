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
#             'Yuusuke Saito'
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import os
import sys
import imp
import glob
import traceback

from ecell.ecs_constants import *

from ecell.Plugin import *
from ecell.ui.osogo.config import *

class OsogoPluginManager(PluginManager):
    """PluginManager specified for Osogo
    """

    # ========================================================================
    def __init__( self, aSession ):
        """Constructor
        aSession   ---  the instance of Session (Session)
        """

        PluginManager.__init__( self, PLUGIN_PATH )

        self.theSession = aSession

        self.thePluginTitleDict = {}     # key is instance , value is title
        self.thePluginWindowNumber = {}
        self.thePropertyWindowOnEntityListWindows = {}  # key is instance, value is None
        self.theAssociatedSession = self.theSession.theSession

    # end of __init__
        
    # ========================================================================
    def createInstance( self, classname, data, rootWidget=None, parent=None ):
        '''
        classname  --- a class name of PluginWindow (str)
        data       --- a RawFullPN (RawFullPN)
        rootWidget --- a root widget (str or None)
        parent     --- a parentWindow (Window)    # NOT gtk.Window
        '''
        if self.thePluginMap.has_key( classname ):
            pass
        else:
            self.loadModule( classname )

        plugin = self.thePluginMap[ classname ]

        # -------------------------------------------------------
        # If plugin window does not exist on EntryList,
        # then creates new title and sets it to plugin window.
        # -------------------------------------------------------
        title = ""

        if parent.__class__.__name__ == 'EntityListWindow':
            pass
        else:
            title = classname[:-6]

            if self.thePluginWindowNumber.has_key( classname ):
                self.thePluginWindowNumber[ classname ] += 1
            else:
                self.thePluginWindowNumber[ classname ] = 1
            title = "%s%d" %(title,self.thePluginWindowNumber[ classname ])

        try:
            instance = plugin.createInstance( data, self, rootWidget, parent )
        except TypeError:
            errorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value, sys.exc_traceback ) )
            self.theSession.message( errorMessage )
            return None

        instance.openWindow()

        #try:
        if parent.__class__.__name__ == 'EntityListWindow':
            self.thePropertyWindowOnEntityListWindows[ instance ] = None
        else:
            instance.editTitle( title )
            self.thePluginTitleDict[ instance ] = title
            self.theInstanceList.append( instance )
        # initializes session
        self.updateFundamentalWindows()
        #except:
        #    pass

        return instance

    # end of createInstance

    # ========================================================================
    def loadModule( self, aClassname ):
        """loads plugin window
        aClassname   ---   a class name of PluginWindow
        """

        PluginManager.loadModule( self, aClassname )

    # ========================================================================
    def loadAll( self ):
        """loads all plugin windows' files
        Returns None
        """

        for aPath in PLUGIN_PATH:
            aFileList = glob.glob( os.path.join( aPath, '*.glade' ) )
            for aFile in aFileList:
                aModulePath = os.path.splitext( aFile )[0]
                if( os.path.isfile( aModulePath + '.py' ) ):
                    aModuleName = os.path.basename( aModulePath )
                    self.loadModule( aModuleName )


    # ========================================================================
    def updateAllPluginWindow( self ):
        """updates all plugin windows
        Returns None
        """
        if self.theSession.theSession is not self.theAssociatedSession:
            self.theAssociatedSession = self.theSession.theSession
            for i in self.theInstanceList:
                i.close()
            return

        if self.theSession.theSession is None:
            return

        # updates all plugin windows
        PluginManager.updateAllPluginWindow(self)

        # updates PropertyWindow on EntityListWindow
        if self.thePropertyWindowOnEntityListWindows != None:
            for aPropertyWindowOnEntityListWindow in self.thePropertyWindowOnEntityListWindows.keys():
                aPropertyWindowOnEntityListWindow.update()


    # ---------------------------------------------------------------
    # appendInstance  (overrides PluginManager)
    #   - appends an instance to instance list
    #   - If catchs exception from the method of super class,
    #     then print message to MessageWindow.
    #
    # anInstance     : an instance
    # return -> None
    # ---------------------------------------------------------------
    # ========================================================================
    def appendInstance( self, anInstance ):

        pass

    # end of appendInstance


    # ========================================================================
    def removeInstance( self, anInstance ):
        """override superclass's method
        anInstance   --- a PluginWindow instance 
        Returns None
        """

        # calls superclass's method
        PluginManager.removeInstance(self, anInstance)

        # deletes it from the instance map
        
        if self.thePluginTitleDict.has_key( anInstance ):
            del self.thePluginTitleDict[ anInstance ] 
        else:
            pass

        # The following process is verbose
        # when the instance is not deleted, destroy it.
        if anInstance != None:
            if anInstance[anInstance.__class__.__name__] != None:
                anInstance[anInstance.__class__.__name__].destroy()

        # updaets fundamental windows
        self.theSession.updateFundamentalWindows()


    # ========================================================================
    def removeInstanceByTitle( self, aTitle ):
        """removes PluginWindow instance by title
        aTitle   --- a PluginWindow's title (str)
        Returns None
        """
        
        # converts the title to str type
        aTitle = str(aTitle)

        # removes the instance
        for anInstance in self.theInstanceList:
            if aTitle == self.thePluginTitleDict[ anInstance ]:
                self.removeInstance( anInstance )
                break

    # ========================================================================
    def editModuleTitle( self, aPluginInstance, aTitle ):
        """overwrites superclass's method
        edits PluginWindow's title
        aPluginInstance   --- the PluginWindow to change title (PluginWindow)
        aTitle            --- a new PluginWindow's title (str)
        Returns None
        """

        self.thePluginTitleDict[aPluginInstance] = aTitle
        PluginManager.editModuleTitle( self, aPluginInstance, aTitle)

    # ========================================================================
    def editInstanceTitle( self, anOldTitle, aNewTitle ):
        """edits PluginWindow's title
        anOldTitle   --- current PluginWindow's title (str)
        anNewTitle   --- a new PluginWindow's title (str)
        Returns None
        """

        # converts the title to str type
        anOldTitle = str(anOldTitle)
        aNewTitle = str(aNewTitle)

        # edits the instance's title
        for anInstance in self.theInstanceList:
            #print self.thePluginTitleDict[ anInstance ]
            if anOldTitle == self.thePluginTitleDict[ anInstance ]:
                self.editModuleTitle( anInstance, aNewTitle )
                break


    # ========================================================================
    def showPlugin( self, aPluginInstance ):
        """overwrites superclass's method
        aPluginInstance   ---  a PluginWindow instance 
        Returns None
        """

        try:
            PluginManager.showPlugin(self, aPluginInstance)

        # When the specified instance exists on BoardWindow.
        except AttributeError:
            self.theSession.getWindow('BoardWindow').present()


    # ========================================================================
    def deleteModule( self, *arg ):
        """overwrites superclass's method
        aPluginInstance   ---  a PluginWindow instance 
        Returns None
        """

        self.theSession.updateFundamentalWindows()


    # ========================================================================
    def updateFundamentalWindows( self ):
        """updates fundamental windows
        Returns None
        """

        try:
            self.theSession.updateFundamentalWindows()

        except:
            pass


    # ========================================================================
    def printMessage( self, aMessage ):
        """prints message on MessageWindow
        Returns None
        """

        self.theSession.message(aMessage)


    # ========================================================================
    def updateAllWindows( self ):
        """updates all windows
        Returns None
        """

        self.updateAllPluginWindow()

        self.theSession.updateFundamentalWindows()



    # ========================================================================
    def deletePropertyWindowOnEntityListWinsow( self, aPropertyWindowOnEntityListWindow ):
        """deletes PropertyWindow on EntityListWindow
        Returns None
        """

        if self.thePropertyWindowOnEntityListWindows.has_key(aPropertyWindowOnEntityListWindow):
            del self.thePropertyWindowOnEntityListWindows[aPropertyWindowOnEntityListWindow]

if __name__ == "__main__":
    pass

