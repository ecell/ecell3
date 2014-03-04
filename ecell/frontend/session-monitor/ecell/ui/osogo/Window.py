#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
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
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import os

import gtk
import gtk.gdk

from ecell.ui.osogo.config import *
import ecell.ui.osogo.glade_compat as glade

class Window:
    """The super class of Window class.
    [Note]:This class is not Window widget itself, but has widget instance.
    """
    def __init__( self, gladeFile=None, rootWidget=None ):
        """Constructor
        gladeFile  --  a glade file name (str:absolute path/relative path)
        rootWidget --  a root widget (str or None)
        """

        self.gladeFile = gladeFile   # glade file name
        self.rootWidget = rootWidget        # a root property
        self.widgets = None              # widgets instance

        # Default title is classname of this class.
        self.theTitle = self.__class__.__name__

    def openWindow( self ):
        """
        loads Glade file
        Returns None
        [Note]:If IOError happens during reading Glade file,
               throws an exception.
        """
        # ------------------------------------------------
        # checks and loads glade file
        # ------------------------------------------------
        if os.access( self.gladeFile, os.R_OK ):
            if self.rootWidget != None:
                self.widgets = glade.XML( self.gladeFile, root= str( self.rootWidget ) )
            else:
                self.widgets = glade.XML( self.gladeFile, root= None )
        else:
            raise IOError( "can't read %s." %self.gladeFile )
        
    def addHandlers( self, aHandlers ):
        """sets handlers
        aHandlers  --  a signal handler map (dict)
        Returns None
        """

        if type(aHandlers) != dict:
            raise TypeError("%s must be dict." %str(aHandlers) )

        self.widgets.signal_autoconnect( aHandlers )

    def __getitem__( self, aKey ):
        """returns wiget specified by the key
        aKey  --  a widget name (str)
        Returns a widget (gtk.Widget)
        [Note]:When this window has not the widget specified by the key,
               throws an exception.
        """

        return self.widgets.get_widget( aKey )

    def getWidget( self, aKey ):
        """returns wiget specified by the key
        aKey  --  a widget name (str)
        Returns a widget (gtk.Widget)
        [Note]:This method is same as __getitem__ method.
        """

        return self[ aKey ]

    def setIconList( self, anIconFile16, anIconFile32 ):
        """sets the window icon according to icon size
        anIconFile16 --- icon 16x16 filename 
        anIconFile32 --- icon 32x32 filename 
        """
        aPixbuf16 = gtk.gdk.pixbuf_new_from_file(anIconFile16)
        aPixbuf32 = gtk.gdk.pixbuf_new_from_file(anIconFile32)

        theWidget=self[ self.__class__.__name__ ]
        if theWidget!=None:
                        theWidget.set_icon_list( aPixbuf16, aPixbuf32 )

    def editTitle( self, aTitle ):
        """edits and saves title
        aTitle  --  a title to save (str)
        Returns None
        """

        # save title
        # Although self.theTitle looks verbose, self.getTitle() method
        # returns self.theTitle. See the comment of getTitle() method
        self.theTitle = aTitle

        # get window widget ( The name of window widget is class name )
        theWidget=self[ self.__class__.__name__ ]

        # There are some cases theWidget is None.
        #  - When this method is called after 'destroy' signal.
        #  - When this window is attached other Window.
        # In those cases, do not change title.
        if theWidget!=None:
            theWidget.set_title( self.theTitle )

    def getTitle( self ):
        """gets title of this Window
        Returns a title (str)
        [Note]: This method returs not the title of widget but self.theTitle.
                Because when this method is called after 'destroy' signal,
                all widgets are None.
        """
        return self.theTitle

    def getParent( self ):
        """Returns a Parent Window (Window)   # Not gtk.Window
        """

        if self.rootWidget == None:
            return self
        else:
            return self.__getParent( self.rootWidget )

    def __getParent( self, *arg ):
        """Returns a Parent Window (Window)   # Not gtk.Window
        """

        if arg[0].rootWidget == None:
            return arg[0]
        else:
            return arg[0].__getParent( self.rootWidget )

    def getAllChildren( self ):
        """Returns all widget on this Window (list of widget) 
        Other windows in same glade file are not included.
        """

        aChildren = self[self.__class__.__name__].get_children()
        return self.__getChildren( aChildren )

    def __getChildren( self, aChildren ):
        """Returns all widget on this Window (list of widget) 
        Other windows in same glade file are not included.
        """

        aChildrenList = []    # list of widget

        for aChild in aChildren:

            # when aChild has no children, append it to list.
            try:
                aChild.get_children()
            except AttributeError:
                aChildrenList.append( aChild )
            else:

                # when aChild has no children, append it to list.
                if len(aChild.get_children()) == 0:
                    aChildrenList.append( aChild )

                else:

                    # when aChild has children, call this method.
                    aChildrenList += self.__getChildren( aChild.get_children() )

        return aChildren + aChildrenList 

    def show_all( self ):
        """shows all widgets of this window
        Returns None
        """

        self[self.__class__.__name__].show_all()

# end of Window
