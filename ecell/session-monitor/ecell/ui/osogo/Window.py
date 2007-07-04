#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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

import config

import os
import gtk
import gtk.gdk
import gtk.glade
import string

import utils as outil

# Register Custom component
import Plot

class Window:
    """
    The super class of Window class.
    [Note] This class is a thin wrapper around a GTK Widget provided by Glade.
    """

    def __init__( self, gladeFile = None, aRootWidgetName = None ):
        """Constructor
        gladeFile  --  a glade file name (str:absolute path/relative path)
        theRootWidgetName --  a root widget (str or None)
        """

        if gladeFile == None:
            gladeFile = os.path.join(
                config.glade_dir,
                self.__class__.__name__ + ".glade"
                )
        elif not os.path.isabs( gladeFile ) :
            gladeFile = os.path.join( config.glade_dir, gladeFile )

        # glade file name
        self.gladeFile = gladeFile
        # a root property
        self.theRootWidgetName = aRootWidgetName or self.__class__.__name__
        # root widget (a GTK widget object)
        self.theRootWidget = None
        # widgets instance
        self.glade = None
        # Default title is classname of this class.
        self.theTitle = self.__class__.__name__
        # Parent window (not a GTK Window object)
        self.theParent = None
        # The widget id where this window is attached
        self.theParentWidgetName = None
        # Child window list
        self.theChildWindowList = []
        # Window is shown or not
        self.shown = False
        # destroy handlers
        self.theDestroyHandlers = []

    def exists( self ):
        """
        Returns true if glade file is loaded and does not deleted,
                false otherwise.
        """
        return self.theRootWidget != None

    def loadWindowIcons( self ):
        return (
            gtk.gdk.pixbuf_new_from_file(
                os.path.join( config.glade_dir, 'ecell.png') ),
            gtk.gdk.pixbuf_new_from_file(
                os.path.join( config.glade_dir, 'ecell32.png') )
            )

    def initUI( self ):
        """
        loads Glade file
        Returns None
        [Note]:If IOError happens during reading Glade file,
               throws an exception.
        """
        if self.theRootWidget != None:
            return

        self.glade = gtk.glade.XML( self.gladeFile, self.theRootWidgetName )
        self.theRootWidget = self.glade.get_widget( self.theRootWidgetName )
        if isinstance( self.theRootWidget, gtk.Window ):
            self.theRootWidget.set_icon_list( *self.loadWindowIcons() )

        for aChild, aParentWidgetName in self.theChildWindowList:
            aChild.initUI()
            if aParentWidgetName != None:
                self.getWidget( aParentWidgetName ).add( aChild.theRootWidget )

        if self.theParent != None and self.theParent.exists() and \
           self.theParentWidgetName != None:
            self.theParent.getWidget( self.theParentWidgetName ).add(
                self.theRootWidget )

    def addHandlersAuto( self ):
        aHandlerDict = {}
        for aMemberName in self.__class__.__dict__:
            if aMemberName.startswith( "do" ):
                aMember = getattr( self, aMemberName, None )
                if aMember != None and callable( aMember ):
                    handlerName = '_'.join(
                        map( string.lower,
                            outil.splitCamelcasedName( aMemberName[2:] ) ) )
                    def generateHandler( handlerName, aMember ):
                        if handlerName.startswith( "toggle_" ):
                            def toggleHandler(w):
                                if isinstance( w, gtk.ToggleToolButton ) or \
                                   isinstance( w, gtk.ToggleButton ) or \
                                   isinstance( w, gtk.CheckMenuItem ):
                                    return aMember( w.get_active() )
                                else:
                                    return aMember(
                                        self.checkWidgetState( w.name ) )
                            return toggleHandler
                        elif handlerName.startswith( "set_" ):
                            return lambda w: aMember( w.get_text() )
                        else:
                            return lambda w: aMember()

                    aHandlerDict[ handlerName ] = generateHandler(
                        handlerName, aMember )
            elif aMemberName.startswith( "on" ):
                aMember = getattr( self, aMemberName, None )
                if aMember != None and callable( aMember ):
                    handlerName = '_'.join(
                        map( string.lower,
                            outil.splitCamelcasedName( aMemberName[2:] ) ) )
                    aHandlerDict[ handlerName ] = aMember

        self.glade.signal_autoconnect( aHandlerDict )

    def addHandlers( self, aHandlerDict ):
        """sets handlers
        aHandlerDict  --  a signal handler map (dict)
        Returns None
        """
        if type( aHandlerDict ) != dict:
            raise TypeError("%s must be dict." %str(aHandlerDict) )
        self.glade.signal_autoconnect( aHandlerDict )

    def __getitem__( self, aKey ):
        """
        returns wiget specified by the key
        aKey  --  a widget name (str)
        Returns a widget (gtk.Widget)
        [Note]:When this window has not the widget specified by the key,
               throws an exception.
        """
        return self.getWidget( aKey )

    def getRootWidget( self ):
        return self.theRootWidget

    def getWidget( self, aKey ):
        """
        returns wiget specified by the key
        aKey  --  a widget name (str):
        Returns a widget (gtk.Widget)
        [Note]:This method is same as __getitem__ method.
        """
        return self.glade.get_widget( aKey )

    def setIconList( self, anIconFile16, anIconFile32 ):
        """
        sets the window icon according to icon size
        anIconFile16 --- icon 16x16 filename 
        anIconFile32 --- icon 32x32 filename 
        """
        aPixbuf16 = gtk.gdk.pixbuf_new_from_file(anIconFile16)
        aPixbuf32 = gtk.gdk.pixbuf_new_from_file(anIconFile32)

        theWidget=self[ self.__class__.__name__ ]
        if theWidget!=None:
            theWidget.set_icon_list( aPixbuf16, aPixbuf32 )

    def setTitle( self, aTitle ):
        """
        sets title
        aTitle  --  a title to save (str)
        """
        self.theTitle = aTitle
        if self.theRootWidget != None and \
           isinstance( self.theRootWidget, gtk.Window ):
            self.theRootWidget.set_title( self.theTitle )
        return True

    def getTitle( self ):
        """
        gets title of this Window
        Returns a title (str)
        [Note] This method returs not the title of widget but self.theTitle.
               Because when this method is called after 'destroy' signal,
               all widgets are None.
        """
        return self.theTitle

    def setParent( self, aParent ):
        """
        Sets the parent Window instance
        """
        if self.theParent != None:
            self.theParent.removeChild( self )
        if aParent != None:
            aParent.addChild( self, None )

    def _setParentIntn( self, aParent, aParentWidgetName ):
        self.theParent = aParent
        self.theParentWidgetName = aParentWidgetName

    def getParent( self ):
        """
        Returns a parent Window instance # Not gtk.Window
        """
        return self.theParent

    def addChild( self, aChild, aParentWidgetName ):
        assert aChild.theParent == None
        self.theChildWindowList.append( ( aChild, aParentWidgetName ) )
        aChild._setParentIntn( self, aParentWidgetName )
        if self.exists() and aChild.exists() and aParentWidgetName != None:
            self.getWidget( aParentWidgetName ).add( aChild.theRootWidget )

    def removeChild( self, aTarget ):
        for i in range( 0, len( self.theChildWindowList ) ):
            aChild, aParentWidgetName = self.theChildWindowList[i]
            if aChild == aTarget:
                if self.exists() and aChild.exists():
                    aParentWidget = aChild.theRootWidget.get_parent()
                    if aParentWidget != None:
                        aParentWidget.remove( aChild.theRootWidget )
                aChild._setParentIntn( None, None )
                del self.theChildWindowList[i]
                return

    def getChildren( self ):
        return list( self.theChildWindowList )

    def show( self ):
        if self.theRootWidget == None:
            self.initUI()
        self.theRootWidget.show_all()
        self.shown = True
        if self.theRootWidget.__class__ == gtk.Dialog:
            self.theRootWidget.run()

    def hide( self ):
        if self.theRootWidget == None:
            return
        self.theRootWidget.hide()
        self.shown = False

    def isVisible( self ):
        return self.shown

    def destroy( self ):
        for aCallable in self.theDestroyHandlers:
            aCallable( self )
        if self.theRootWidget == None:
            return
        self.hide()
        for aChild, aParentWidgetName in self.theChildWindowList:
            aChild.hide()
            if aChild.exists():
                aParentWidget = aChild.theRootWidget.get_parent()
                if aParentWidget != None:
                    aParentWidget.remove( aChild.theRootWidget )
            aChild.destroy()
        if self.theParent != None:
            self.theParent.removeChild( self )
        self.theChildWindowList = None
        self.theRootWidget.destroy()
        self.theRootWidget = None
        self.glade = None

    def createGC( self ):
        assert self.theRootWidget != None
        return self.theRootWidget.get_root_window().new_gc()

    def createPixmap( self, width, height, depth = -1 ):
        assert self.theRootWidget != None
        return gtk.gdk.Pixmap(
            self.theRootWidget.get_root_window(), width, height, depth )

    def registerDestroyHandler( self, aCallable ):
        self.theDestroyHandlers.append( aCallable )

    def unregisterDestroyHandler( self, aCallable ):
        try:
            del self.theDestroyHandlers[
                self.theDestroyHandlers.index( aCallable ) ]
        except:
            pass
