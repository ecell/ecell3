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


import gtk

from Window import *



class ListWindow(Window):
    """EditorWindow
    - manages existance status.
    - destroys window when 'delede_event' is catched.
    """
    def __init__( self, aModelEditor, aRoot=None, aGladeFile=None ):
        """constructor
        aModelEditor  -- a reference to ModelEditor (ModelEditor)
        aGladeFile   -- a glade file name (str)
        """
        # calls superclass's constructor
        Window.__init__( self, aGladeFile, aRoot )

        if aRoot == None:
            self.isStandalone = True
        else:
            self.isStandalone = False

        # saves a reference to Session
        self.theModelEditor = aModelEditor
        self.theExist = False

    # ========================================================================
    def exists( self ):
        """Returns TRUE:When glade file is loaded and does not deleted.
                False:When glade file is not loaded yet or already deleted.
        """

        return self.theExist



    # ========================================================================
    def present( self ):
        """moves this window to the top of desktop.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """

    
        # When glade file is not loaded yet or already deleted, does nothing
        # calla present() method of Window widget of this window.
        if self.exists():


          #self[self.__class__.__name__].present()

            if self.isStandalone:
                self[self.__class__.__name__].present()
            else:
                self.theModelEditor.theMainWindow.presentTab( self )


    # ========================================================================
    def iconify( self ):
        """moves this window to the taskbar.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """

    
        # When glade file is not loaded yet or already deleted, does nothing
        # calls iconify() method of Window widget of this window.
        if self.exists():
            if self.isStandalone:

                self[self.__class__.__name__].iconify()

    # ========================================================================
    def move( self, xpos, ypos ):
        """moves this window on the desktop to (xpos,ypos).
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """

    
        # When glade file is not loaded yet or already deleted, does nothing
        # calls move(x,y) method of Window widget of this window.
        if self.exists():
            if self.isStandalone:

                self[self.__class__.__name__].move( xpos, ypos)

    # ========================================================================
    def resize( self, width, heigth ):
        """resizes this window according to width and heigth.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """

    
        # When glade file is not loaded yet or already deleted, does nothing
        # calls resize(width,heigth) method of Window widget of this window.
        if self.exists():
            if self.isStandalone:

                self[self.__class__.__name__].resize( width, heigth)

    # ========================================================================
    def deleted( self, *arg ):
        """ 
        Destroys window instances and all contained widgets
        destroy all reference to widgets before calling this (e.g. listcomponents)
        Returns TRUE
        """

        # destroys this window
        self.close()

        # does not widgets
        return True

    def initUI( self ):
        """overwrite super class's method
        When glade file is not loaded yet or already deleted, calls superclass's
        openWindow() method and connects 'delete_event' and self.delete() method.
        Returns None
        """
        # when glade file is not loaded yet or already deleted.
        if self.theExist == False:
            # sets theExist flag is TRUE
            self.theExist = True
            # calls superclass's method 
            Window.initUI( self )
            #self.theRoot='top_frame'
            # connects 'delete_event' and self.delete() method.

    # ========================================================================
    def update( self ):
        """
        Returns None
        """

        pass

    # ========================================================================
    def close ( self ):
        """ destroys Widgets and sets theExist FALSE """
        if self.exists():
            if self.theModelEditor.getLastUsedComponent() != None:
                if self.theModelEditor.getLastUsedComponent().getParentWindow() == self:
                    self.theModelEditor.setLastUsedComponent( None )

            if self.isStandalone:

                self[self.__class__.__name__].destroy()
            else:
                self.theModelEditor.theMainWindow.detachTab ( self )
                
            self.theExist = False
            self.widgets = None
            
# end of OsogoWindow


