#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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

import os

import gtk
import gtk.gdk

from ecell.ui.model_editor.Config import *
from ecell.ui.model_editor.Constants import *
import ecell.ui.model_editor.glade_compat as glade

class ViewComponent:
    """The super class of Component class.
    [Note]:This class is not  widget itself, but has widget instance.
    """

    # ==============================================================
    def __init__( self, pointOfAttach, anAttachment, aGladeFile=None ):
        """Constructor
        pointOfAttach : container widget the Compononent to attach to
        anAttachment: str name of widget to attach
        aGladeFile  --  a glade file name (str:absolute path/relative path)
        """
        self.theGladeFile = aGladeFile   # glade file name
        self.widgets = None              # widgets instance

        # ------------------------------------------------
        # loads GLADEFILE_PATH/CLASSNAME.glade by default
        # ------------------------------------------------
        if self.theGladeFile == None:
            self.theGladeFile = GLADEFILE_PATH
            self.theGladeFile += os.sep + self.__class__.__name__ + ".glade"
        else:

            # ------------------------------------------------
            # When absolute path
            # ------------------------------------------------
            if os.path.isabs( self.theGladeFile ) :
                pass
            # ------------------------------------------------
            # When relative path
            # ------------------------------------------------
            else:
                self.theGladeFile = GLADEFILE_PATH + os.sep + self.theGladeFile

        # ------------------------------------------------
        # checks and loads glade file
        # ------------------------------------------------
        if os.access( os.path.join( GLADEFILE_PATH, self.theGladeFile ), os.R_OK ):
                self.widgets = glade.XML( self.theGladeFile, root=anAttachment )
        else:
            raise IOError( "can't read %s." %self.theGladeFile )
        
        self.theRoot = self[anAttachment]  

        pointOfAttach.add( self.theRoot )
        

    # ==============================================================
    def addHandlers( self, aHandlers ):
        """sets handlers
        aHandlers  --  a signal handler map (dict)
        Returns None
        """

        if type(aHandlers) != dict:
            raise TypeError("%s must be dict." %str(aHandlers) )

        self.widgets.signal_autoconnect( aHandlers )


    # ==============================================================
    def __getitem__( self, aKey ):
        """returns wiget specified by the key
        aKey  --  a widget name (str)
        Returns a widget (gtk.Widget)
        [Note]:When this window has not the widget specified by the key,
               throws an exception.
        """

        return self.widgets.get_widget( aKey )


    # ==============================================================
    def getWidget( self, aKey ):
        """returns wiget specified by the key
        aKey  --  a widget name (str)
        Returns a widget (gtk.Widget)
        [Note]:This method is same as __getitem__ method.
        """

        return self[ aKey ]


    # ==============================================================
    def getParent( self ):
        """Returns a Parent Window (Window)   # Not gtk.Window
        """

        if self.theRoot == None:
            return self
        else:
            return self.__getParent( self.theRoot )


    # ==============================================================
    def __getParent( self, *arg ):
        """Returns a Parent Window (Window)   # Not gtk.Window
        """

        if arg[0].theRoot == None:
            return arg[0]
        else:
            return arg[0].__getParent( self.theRoot )



    # ================================================================
    def show_all( self ):
        """shows all widgets of this window
        Returns None
        """

        self.theRoot.show_all()


    # ================================================================
    def close( self ):
        """
        destroys widgets and reduces refcount to zero
        """
        self.theRoot.destroy()
        self.widgets = None

    # ================================================================
    def getADCPFlags( self, aType ):
        return [False, False, False, False]


    # ================================================================
    def getMenuItems( self ):

        aMenu = []
        
        print aMenu

        aFlags = self.theModelEditor.getADCPFlags()

        aMenu.append( ["add new", aFlags[ME_ADD_FLAG] ] )

        aMenu.append( [ "delete", aFlags[ME_DELETE_FLAG] ] ) 

        aMenu.append( [ "copy", aFlags[ME_COPY_FLAG] ] ) 

        aMenu.append( [ "cut", aFlags[ME_DELETE_FLAG] and aFlags[ME_COPY_FLAG] ] )

        aMenu.append( [ "paste", aFlags[ME_PASTE_FLAG] ] )
    
        return aMenu

    # ================================================================
    def applyMenuItem( self, aMenuPoint ):
        aMenuPoint = aMenuPoint.replace(' ', '_' )
        if aMenuPoint not in self.__class__.__dict__.keys():
            return False
        apply( self.__class__.__dict__[ aMenuPoint ], [self] )
        return True

# end of EditorComponent





