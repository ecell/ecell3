#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from OsogoUtil import *

from OsogoWindow import *
import gobject

from os import *

from ecell.ecssupport import *
import ConfirmWindow

from Numeric import *

FORWARD=0
DOWN=1

UNSELECTED_SHADOW_TYPE=0
SELECTED_SHADOW_TYPE=1
MAXROW=100
MAXCOL=100

MAXWIDTH=1200
MAXHEIGHT=800

class BoardWindow(OsogoWindow):


	def __init__( self, aSession ): 

		OsogoWindow.__init__( self, aSession, 'BoardWindow.glade' )

		self.thePluginManager = aSession.thePluginManager
		self.theX = 0
		self.theY = 0
		self.theRowSize = -1
		self.theColSize = 3
		self.theRow = -1
		self.theCol = -1
		self.thePluginMap = {}
		self.theSelectedPluginFrame = None
		self.theSession = aSession
	# end of the __init__

	def openWindow(self):

		OsogoWindow.openWindow( self )

		self.addHandlers({ \
		                   #'on_close_button_clicked' : self.attachPluginWindow, 
		                   'on_close_button_clicked' : self.deleted, 
		                   'on_delete_button_clicked' : self.deletePluginWindow, 
		                   'on_forward_radiobutton_toggled' : self.changeAlignment, 
		                   'on_size_spinbutton_value_changed' : self.changeTableSize,
		                   'on_title_entry_activate' : self.changeTitle,
			})

		self['board_table'].resize( MAXROW, MAXCOL )
		self['size_spinbutton'].set_text( str(self.theColSize) )
		#self[self.__class__.__name__].connect('delete_event',self.hide)
		self.theScrolledWindow = self['scrolledwindow1']
		self.theViewPort=self['viewport1']
		self.deleteScroll()
		for anEntityWindow in self.theSession.getWindow('EntityListWindow'):
			anEntityWindow.checkBoardExists()


	def close( self ):
		# delete all pluginwindows
		for aPlugin in self.thePluginMap.keys():
			self.deletePluginWindowByTitle( self.thePluginMap[aPlugin].getTitle() )

		# delete all references to widgets:
		self.theScrolledWindow = None
		self.theViewPort = None

		
		# call bas class
		OsogoWindow.close( self )


	def displayScroll(self):
		if self.theViewPort.get_child() == None:
			board_table=self['vbox3']
			self['hbox6'].remove(board_table)
			self.theViewPort.add(self['vbox3'])
			self['hbox6'].add(self.theScrolledWindow)
	
	def deleteScroll(self):
		if self.theViewPort.get_child() != None:
			board_table=self['vbox3']
			self.theViewPort.remove(board_table)
			self['hbox6'].remove(self.theScrolledWindow)
			self['hbox6'].add(self['vbox3'])


	# end of openWindow


	def addPluginWindows( self, aPluginWindowType, aRawFullPNList ):
		if aPluginWindowType == 'TracerWindow':
			aButton=self.attachPluginWindow( aPluginWindowType, aRawFullPNList )
		else:
			for aRawFullPN in aRawFullPNList:
				aButton=self.attachPluginWindow( aPluginWindowType, [ aRawFullPN ] )
		return aButton

	def attachPluginWindow( self, aPluginWindowType, aRawFullPNList ):

		aButton = self.thePluginManager.createInstance( \
		    aPluginWindowType, aRawFullPNList, self )

		aTopFrame = aButton['top_frame']

		aPluginFrame = gtk.Frame()
		aPluginFrame.add(aTopFrame)
		aPluginFrame.set_shadow_type(UNSELECTED_SHADOW_TYPE)
		aPluginFrame.connect('set_focus_child',self.__radioButtonToggled)

		self.thePluginMap[ str(aPluginFrame) ] = aButton

		r,c = self.__getNextPosition()
		self.__appendPluginFrame( aPluginFrame, r, c )
		aPluginFrame.show_all()
		self.updatePositions()
		return aButton

	def deletePluginWindow( self, *arg ):
		if self.theSelectedPluginFrame != None:
			aTitle = self.thePluginMap[str(self.theSelectedPluginFrame)].getTitle()
			del self.thePluginMap[ str(self.theSelectedPluginFrame) ]
			self['board_table'].remove( self.theSelectedPluginFrame )
			self.theSelectedPluginFrame = None
			self['title_entry'].set_text('')
			self.thePluginManager.removeInstanceByTitle(aTitle)
			self.updatePositions()




	def deletePluginWindowByTitle( self, aTitle ):
		if self.exists() == TRUE:
			for aPluginFrame in self['board_table'].get_children():
				if self.thePluginMap[str(aPluginFrame)].getTitle() == aTitle:
					self.theSelectedPluginFrame = aPluginFrame
					self.deletePluginWindow()
					self.updatePositions()
					break
		# When this window is not created, does nothing.
		else:
			pass

	def __radioButtonToggled( self, *arg ):

		aTitle = self.thePluginManager.thePluginTitleDict[ self.thePluginMap[str(arg[0])] ]
		self['title_entry'].set_text( aTitle )

		if self.theSelectedPluginFrame != None:
			self.theSelectedPluginFrame.set_shadow_type(UNSELECTED_SHADOW_TYPE)
		self.theSelectedPluginFrame = arg[0]
		self.theSelectedPluginFrame.set_shadow_type(SELECTED_SHADOW_TYPE)

		arg[0].set_shadow_type(1)

	def __appendPluginFrame( self, aPluginFrame, aRow, aCol ):
		self['board_table'].attach(aPluginFrame,aCol,aCol+1,aRow,aRow+1,\
		                           xoptions=gtk.EXPAND,yoptions=gtk.EXPAND)

		if len( self['board_table'].get_children() ) == 1:
			self.__radioButtonToggled( self['board_table'].get_children()[0] )
			self.theSelectedPluginFrame = self['board_table'].get_children()[0] 

	def __getNextPosition( self ):

		# --------------------------------------------
		# when row and col are in initial position.
		# --------------------------------------------
		if self.theRow == -1 and self.theCol == -1:
			self.theRow = 0
			self.theCol = 0
			return ( self.theRow, self.theCol )

		# --------------------------------------------
		# when the row size is not specified.
		# --------------------------------------------
		elif self.theRowSize == -1:
			if self.theCol < self.theColSize-1:
				self.theCol += 1
			else:
				self.theCol = 0
				self.theRow += 1

		# --------------------------------------------
		# when the col size is not specified.
		# --------------------------------------------
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

	def changeTableSize( self, *arg ):
		self.changeAlignment()

	def __updateSize( self ):

		aSize = string.atoi( self['size_spinbutton'].get_text() )
		# When 'forward' is toggled.
		if self['forward_radiobutton'].get_active() == TRUE:
			self.theRowSize = -1
			self.theColSize = aSize

		# When 'down' is toggled.
		else:
			self.theRowSize = aSize
			self.theColSize = -1


	def updatePositions( self, *arg ):
		anElementList = []
		aRequisitionList = []
		theRowHeights = []
		theColumnWidths = []

		for anElement in self['board_table'].get_children():
			anElementList.insert( 0, anElement )
			self['board_table'].remove( anElement )

		#calculate dimensions
		if self.theRowSize == -1:
			__rowsize = int(ceil( float(len( anElementList )) / self.theColSize ) )
			__colsize = self.theColSize
		else:
			__rowsize = self.theRowSize
			__colsize = int(ceil( ceil(len(anElementList )) / self.theRowSize ) )


		#init requisitionlist
		__matrixsize = max (__rowsize, __colsize)
 		for i in range(0 , __matrixsize * __matrixsize ):
			aRequisitionList.append(None)
		
		for i in range(0, __matrixsize ):
			theRowHeights.append( 0 )
			theColumnWidths.append( 0 )
	
		self.__initializePosition()
		
		for anElement in anElementList:
			r,c = self.__getNextPosition()
			self.__appendPluginFrame( anElement, r, c )
			aRequisitionList[r * __matrixsize + c ] = anElement.size_request()

		#shrink to fit
		theTableHeight = self['board_table'].size_request()[1]
		theTableWidth = self['board_table'].size_request()[0]

		#calculate window dimensions
		__hboxrequisition = self['hbox1'].size_request()
		theWindowHeight = theTableHeight + __hboxrequisition[1]
		theWindowWidth = max ( theTableWidth, __hboxrequisition[0] )
		scroll_needed = False
		if theWindowHeight > MAXHEIGHT:
			theWindowHeight = MAXHEIGHT
			scroll_needed = True
			theWindowWidth += 10
		if theWindowWidth > MAXWIDTH:
			theWindowWidth = MAXWIDTH
			scroll_needed = True
			theWindowHeight += 10

		self['BoardWindow'].resize( theWindowWidth, theWindowHeight )		
		if scroll_needed:
			self.displayScroll()
		else:
			self.deleteScroll()


	def changeAlignment( self, *arg ):
		# --------------------------------------------
		# When 'forward' is toggled.
		# --------------------------------------------
		if self['forward_radiobutton'].get_active() == TRUE:
			self['size_label'].set_text('Cols :')
			self.__updateSize()
			self.updatePositions()

		# --------------------------------------------
		# When 'down' is toggled.
		# --------------------------------------------
		else:
			self['size_label'].set_text('Rows :')
			self.__updateSize()
			self.updatePositions()

	def changeTitle( self, *arg ):
		
		aNewTitle = self["title_entry"].get_text() 
		aNewTitle = string.strip( aNewTitle )
	
		if len(aNewTitle) == 0:
			anErrorMessage='\nError text field is blank.!\n'
			#self.theSession.message( anErrorMessage )
			aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,"!")
			return None

		aTitle = self.thePluginMap[str(self.theSelectedPluginFrame)].getTitle()

		self.thePluginManager.editInstanceTitle( aTitle, aNewTitle )
		self.theSession.updateWindows()


	def shrink_to_fit(self):
		self.updatePositions()

	# ========================================================================
	def resize( self, width, heigth ):
		"""resizes this window according to width and heigth.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""
		if self.exists():
			self[self.__class__.__name__].resize( width, heigth)
			self.updatePositions()

	# ========================================================================
	def setPackDirectionForward( self, aDirection ):
		""" sets direction of packing PluginWindows  
			aDirection:
			True : Forward
			False : Down
		"""
		if aDirection:
			self['forward_radiobutton'].set_active( gtk.TRUE )
		if not aDirection:
			self['forward_radiobutton'].set_active( gtk.FALSE )

		self.changeAlignment( None )				
	
	# ========================================================================
	def setTableSize( self, aSize ):
		""" sets row or column size of BoardWindow 
		   depening on packing strategy
		"""
		self['size_spinbutton'].set_text( str(aSize) )
		self.changeAlignment()


	# end of editTitle

                    
