#!/usr/bin/env python

from OsogoWindow import *
from gtk import *
import gobject
from ecell.ecssupport import *
from string import *

# Constants for plugin instance list
TITLE  = 0
CLASS  = 1
FULLPN = 2


class InterfaceWindow( OsogoWindow ):
	"""InterfaceWindow
	- displays the list of plugin instance
	- A title of plugin instance can be changed.
	- A plugin instance can be deleted.
	- A plugin instance can be moved to the top of desktop.
	"""
    
	# ==========================================================================
	def __init__( self, aSession ):
		"""Constructor
		aSession   ---  a reference to Session (Session)
		"""

		# calls superclass's constructor
		OsogoWindow.__init__( self, aSession )

		# saves reference to PluginManager
		self.thePluginManager = aSession.thePluginManager

		# initializes row selection attributes
		self.theSelectedRow = None


	# ==========================================================================
	def openWindow( self ):
		"""overwrites superclass's method
		Returns None
		"""

		# calls superclass's method
		OsogoWindow.openWindow(self)

		# sets up plugin instance list
		self.theInterfaceListWidget = self[ 'InterfaceCList' ]
		aListStore = gtk.ListStore( gobject.TYPE_STRING,\
					    gobject.TYPE_STRING,\
					    gobject.TYPE_STRING )
		self.theInterfaceListWidget.set_model( aListStore )
		column=gtk.TreeViewColumn('Title',gtk.CellRendererText(),text=TITLE)
		column.set_resizable(gtk.TRUE)
		self.theInterfaceListWidget.append_column(column)
		column=gtk.TreeViewColumn('Class',gtk.CellRendererText(),text=CLASS)
		column.set_resizable(gtk.TRUE)
		self.theInterfaceListWidget.append_column(column)
		column=gtk.TreeViewColumn('FullPN',gtk.CellRendererText(),text=FULLPN)
		column.set_resizable(gtk.TRUE)
		self.theInterfaceListWidget.append_column(column)

		# initialize row selection
		self.theSelectedRow = None

		# appends signal handers
		self.addHandlers( { 'interfacelist_select_row' : self.rowSelected,
		                    'ShowButton_clicked'       : self.showWindow,
		                    'SaveButton_clicked'       : self.editTitle,
		                    'DeleteButton_clicked'     : self.deleteWindow } )


	# ==========================================================================
	def update( self ):
		"""overwrites superclass's method
		Returns None
		"""
	
		try:
			aModel = self.theInterfaceListWidget.get_model()
		except:
			return None

		if aModel != None:
			aModel.clear()

		# gets
		anInstanceList = self.theSession.thePluginManager.theInstanceList
		for anInstance in anInstanceList:
			aTitle = anInstance.getTitle()
			aClass =  anInstance.__class__.__name__
			aFullPN = createFullPNString( anInstance.theFullPN() )
			#aList = ( aTitle , aClass , aFullPN )

			if self.exists() == TRUE:

				anIter = aModel.append()
				aModel.set( anIter,  \
				        TITLE, aTitle,\
					    CLASS, aClass,\
					    FULLPN, aFullPN )


	# ==========================================================================
	def editTitle( self , *arg ):
		"""edits title of plugin list
		Returns None
		"""
		
		# if no data is selected, show error message.
		if self.theSelectedRow == None:
			anErrorMessage='\nNo data is selected!\n'
			aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,"Error!")
			return None

		# if a data is selected, then remove it.
		else:

			# gets inputted title
			aNewTitle = self[ "SelectedTitle" ].get_text() 
			aNewTitle = strip( aNewTitle )
	
			# checks the length of inputted title.
			if len(aNewTitle) == 0:
				anErrorMessage='\nTitle is empty!\n'
				aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,"Error!")
				return None

			# gets current title
			aTitle =  self['InterfaceCList'].get_model().get_value(self.theSelectedRow,TITLE)

			# sets new title 
			self.theSession.thePluginManager.editInstanceTitle( aTitle, aNewTitle )

		# updates all fundamental windows
		self.theSession.updateWindows()

                    
	# ==========================================================================
	def rowSelected( self , arg ):
		"""selectes one row.
		Returns None
		"""

		# gets selected row
		row=self['InterfaceCList'].get_selection().get_selected()[1]
		self.theSelectedRow = row
		
		# updates bottom text field
		aText =  self['InterfaceCList'].get_model().get_value( row, TITLE )
		self[ "SelectedTitle" ].set_text( aText )


	# ==========================================================================
	def showWindow( self , *arg ):
		"""move a plugin instance to the top of desktop
		Returns None
		"""

		# -------------------------------------------
		# show an Instance
		# -------------------------------------------

		# if no data is selected, show error message.
		if self.theSelectedRow == None:
			anErrorMessage='\nNothing is selected!\n'
			aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,"Error!")
			return None

		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_model().get_value(self.theSelectedRow,TITLE)
			aTitleDict = self.theSession.thePluginManager.thePluginTitleDict

			anInstanceList = self.theSession.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					self.theSession.thePluginManager.showPlugin( anInstance )
			

	# end of showWindow

                    
	# ==========================================================================
	def deleteWindow( self , *arg ):
		"""delete a plugin instance
		Returns None
		"""

		# -------------------------------------------
		# delete an Instance
		# -------------------------------------------

		# if no data is selected, show error message.
		if self.theSelectedRow == None:
			anErrorMessage='\nNothing is selected.!\n'
			#self.theSession.message( anErrorMessage )
			aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
			return None

		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_model().get_value( self.theSelectedRow ,0 )
			self.theSession.getWindow('BoardWindow').deletePluginWindowByTitle( aTitle )
			self.thePluginManager.removeInstanceByTitle( aTitle )

			# clear select status
			self.theSelectedRow = None

		# -------------------------------------------
		# update list
		# -------------------------------------------
		self.update()

	# end of deleteWindow



if __name__ == "__main__":


	def mainLoop():
		gtk.mainloop()

	def main():
			aWindow = InterfaceWindow( 'InterfaceWindow.glade' )
			mainLoop()

	main()


