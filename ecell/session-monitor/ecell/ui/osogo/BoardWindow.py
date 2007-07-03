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
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#
import gtk
from ecell.ecssupport import *
from OsogoWindow import OsogoWindow
from ConfirmWindow import ConfirmWindow
from utils import *

from numpy import *

FORWARD=0
DOWN=1

UNSELECTED_SHADOW_TYPE=0
SELECTED_SHADOW_TYPE=1
MAXROW=100
MAXCOL=100

MAXWIDTH=1200
MAXHEIGHT=800

class BoardWindow( OsogoWindow ):
    def __init__( self ):
        OsogoWindow.__init__( self )
        self.theX = 0
        self.theY = 0
        self.theRowSize = -1
        self.theColSize = 3
        self.theRow = -1
        self.theCol = -1
        self.thePluginMap = {}
        self.theSelectedPluginFrame = None
        self.alignmentIsForward = True

    def initUI(self):
        OsogoWindow.initUI( self )
        self.addHandlers( {
            'on_close_button_clicked': self.handleDeleteEvent, 
            } )

        self.addHandlersAuto()

        self['board_table'].resize( MAXROW, MAXCOL )
        self['size_spinbutton'].set_value(
            self.alignmentIsForward and self.theColSize or \
                                        self.theRowSize )
        self.theScrolledWindow = self['scrolledwindow1']
        self.theViewPort = self['viewport1']

    def destroy( self ):
        OsogoWindow.destroy( self )
        self.thePluginMap = None
        self.theScrolledWindow = None
        self.theViewPort = None

    def addPluginWindows( self, aPluginWindowType, aRawFullPNList ):
        if aPluginWindowType == 'TracerWindow':
            aButton = self.attachPluginWindow(
                aPluginWindowType, aRawFullPNList )
        else:
            for aRawFullPN in aRawFullPNList:
                aButton = self.attachPluginWindow(
                    aPluginWindowType, [ aRawFullPN ] )
        return aButton

    def getWidget( self, aWidgetName ):
        aWidget = OsogoWindow.getWidget( self, aWidgetName )
        if aWidget != None:
            return aWidget
        if self.thePluginMap.has_key( aWidgetName ):
            return self.thePluginMap[ aWidgetName ][0]
        return None

    def attachPluginWindow( self, aPluginWindowType, aRawFullPNList ):
        aPluginWindow = self.theSession.openPluginWindow(
            aPluginWindowType, aRawFullPNList )

        r,c = self.getNextPosition()
        aPluginFrame = gtk.Frame()
        aCellWidgetName = "board(%d,%d)" % ( r, c )
        aPluginFrame.set_name( aCellWidgetName )
        aPluginFrame.set_label( aPluginWindow.getTitle() )
        aPluginFrame.set_shadow_type( UNSELECTED_SHADOW_TYPE )
        aPluginFrame.connect('set_focus_child',
            lambda w, c: self.selectCell( w ) )
        self.attachPluginFrame( ( aPluginFrame, aPluginWindow ), r, c )
        self.addChild( aPluginWindow, aCellWidgetName )
        aPluginFrame.show_all()
        self.updatePositions()
        return aPluginWindow

    def handleSessionEvent( self, event ):
        if event.type == 'plugin_instance_removed':
            for anElement in self.thePluginMap.values():
                if anElement[1] == event.instance:
                    self.detachPluginFrame( anElement )
                    self.updatePositions()
                    return
        elif event.type == 'plugin_window_title_changed':
            for anElement in self.thePluginMap.values():
                if anElement[1] == event.instance:
                    anElement[0].set_label( event.instance.getTitle() )
                    return

    def doDeletePluginWindow( self ):
        if self.theSelectedPluginFrame != None:
            self.removePluginFrame(
                self.thePluginMap[ self.theSelectedPluginFrame.get_name() ] )
            self.theSelectedPluginFrame = None
            self['title_entry'].set_text('')

    def removePluginFrame( self, anElement ):
        anElement[1].destroy()

    def findPluginFrameByTitle( self, aTitle ):
        for anElement in self.thePluginMap.values():
            if anElement[1].getTitle() == aTitle:
                return anElement
        return None

    def selectCell( self, aCellWidget ):
        if self.thePluginMap == None:
            return
        aTitle = self.thePluginMap[ aCellWidget.get_name() ][1].getTitle()
        self['title_entry'].set_text( aTitle )

        if self.theSelectedPluginFrame != None:
            self.theSelectedPluginFrame.set_shadow_type(
                UNSELECTED_SHADOW_TYPE )

        aCellWidget.set_shadow_type( SELECTED_SHADOW_TYPE )
        self.theSelectedPluginFrame = aCellWidget

    def attachPluginFrame( self, anElement, aRow, aCol ):
        ( aPluginFrame, aPluginWindow ) = anElement
        self.thePluginMap[ aPluginFrame.get_name() ] = anElement
        self['board_table'].attach(
            aPluginFrame, aCol, aCol+1, aRow, aRow+1,\
            xoptions = gtk.EXPAND | \
                       gtk.SHRINK,
            yoptions = gtk.EXPAND | \
                       gtk.SHRINK )
        if len( self['board_table'].get_children() ) == 1:
            self.selectCell( self['board_table'].get_children()[0] )

    def detachPluginFrame( self, anElement ):
        ( aPluginFrame, aPluginWindow ) = anElement
        aPluginFrame.get_parent().remove( aPluginFrame )
        del self.thePluginMap[ anElement[0].get_name() ]
        return aPluginWindow

    def getNextPosition( self ):
        # when row and col are in initial position.
        if self.theRow == -1 and self.theCol == -1:
            self.theRow = 0
            self.theCol = 0
            return ( self.theRow, self.theCol )
        # when the row size is not specified.
        elif self.theRowSize == -1:
            if self.theCol < self.theColSize-1:
                self.theCol += 1
            else:
                self.theCol = 0
                self.theRow += 1
        # when the col size is not specified.
        else:
            if self.theRow < self.theRowSize-1:
                self.theRow += 1
            else:
                self.theRow = 0
                self.theCol += 1

        return ( self.theRow, self.theCol )

    def __initializePosition( self ):
        self.theRow = -1
        self.theCol = -1
        self.theWidth = 0
        self.theHeigth = 0

    def updatePositions( self ):
        anElementList = self.thePluginMap.values()
        aRequisitionList = []
        theRowHeights = []
        theColumnWidths = []

        for anElement in anElementList:
            self.detachPluginFrame( anElement )

        #calculate dimensions
        if self.theRowSize == -1:
            rowsize = int( ceil( len( anElementList ) / self.theColSize ) )
            colsize = self.theColSize
        else:
            rowsize = self.theRowSize
            colsize = int( ceil( len( anElementList ) / self.theRowSize ) )
        #init requisitionlist
        matrixsize = max(rowsize, colsize)
        for i in range(0 , matrixsize * matrixsize ):
            aRequisitionList.append(None)
        
        for i in range(0, matrixsize ):
            theRowHeights.append( 0 )
            theColumnWidths.append( 0 )
    
        self.__initializePosition()
        
        for anElement in anElementList:
            r, c = self.getNextPosition()
            self.attachPluginFrame( anElement, r, c )
            aRequisitionList[r * matrixsize + c ] = anElement[0].size_request()

    def setAlignmentAndTableSize( self, anAlignmentType, aSize = None ):
        self.alignmentIsForward = anAlignmentType
        if anAlignmentType:
            self.theRowSize = -1
            self.theColSize = aSize
            self['size_label'].set_text('Cols :')
        else:
            self.theRowSize = aSize
            self.theColSize = -1
            self['size_label'].set_text('Rows :')
        self['size_spinbutton'].set_text( str(aSize) )

    def doSetTableSize( self, aSize ):
        self.setAlignmentAndTableSize(
            self.alignmentIsForward, int( aSize ) )
        self.updatePositions()

    def doToggleAlignment( self, state ):
        self.setAlignmentAndTableSize( state,
            self.alignmentIsForward and self.theColSize or self.theRowSize )
        self.updatePositions()

    def doSetTitleEntry( self, aNewTitle ):
        aNewTitle = string.strip( aNewTitle )
    
        if len(aNewTitle) == 0:
            anErrorMessage='\nError text field is blank.!\n'
            aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,"!")
            return None

        self.thePluginMap[ self.theSelectedPluginFrame.get_name() ][1].setTitle( aNewTitle )
        self.theSession.updateUI()

    def shrink_to_fit(self):
        self.updatePositions()

    def resize( self, width, heigth ):
        """resizes this window according to width and heigth.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """
        if self.exists():
            self[self.__class__.__name__].resize( width, heigth)
            self.updatePositions()

    def setPackDirectionForward( self, aDirection ):
        """ sets direction of packing PluginWindows  
            aDirection:
            True : Forward
            False : Down
        """
        if aDirection:
            self['forward_radiobutton'].set_active( True )
        if not aDirection:
            self['forward_radiobutton'].set_active( False )

        self.changeAlignment( None )                

