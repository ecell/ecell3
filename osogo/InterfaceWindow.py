#!/usr/bin/env python

from OsogoWindow import *
from gtk import *
import gobject
from ecell.ecssupport import *
from string import *

class InterfaceWindow( OsogoWindow ):
    
	# -------------------------------------------------------------
	# Constructor
	# -------------------------------------------------------------
	def __init__( self, aMainWindow ):

		OsogoWindow.__init__( self, aMainWindow )
		self.theSelectedRow = None

	# end of __init__

	# -------------------------------------------------------------
	# openWindow
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def openWindow( self ):
		OsogoWindow.openWindow(self)

		self.theInterfaceListWidget = self[ 'InterfaceCList' ]
		aListStore = gtk.ListStore( gobject.TYPE_STRING,\
					    gobject.TYPE_STRING,\
					    gobject.TYPE_STRING )
		self.theInterfaceListWidget.set_model( aListStore )
		column=gtk.TreeViewColumn('Title',gtk.CellRendererText(),text=0)
		self.theInterfaceListWidget.append_column(column)
		column=gtk.TreeViewColumn('Class',gtk.CellRendererText(),text=1)
		self.theInterfaceListWidget.append_column(column)
		column=gtk.TreeViewColumn('FullPN',gtk.CellRendererText(),text=2)
		self.theInterfaceListWidget.append_column(column)
		self.theSelectedRow = None
		self.addHandlers( { 'interfacelist_select_row' : self.rowSelected,
		                    'ShowButton_clicked'       : self.showWindow,
		                    'SaveButton_clicked'       : self.editTitle,
		                    'DeleteButton_clicked'     : self.deleteWindow } )

	# end of openWindow

	
	# -------------------------------------------------------------
	# update
	#    - update CList
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def update( self ):
	
		aModel = self.theInterfaceListWidget.get_model()

		#		if self.theExist != 0:
		if aModel != None:
			aModel.clear()

		# gets
		anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
		for anInstance in anInstanceList:
			aTitle = anInstance.getTitle()
			aClass =  anInstance.__class__.__name__
			aFullPN = createFullPNString( anInstance.theFullPN() )
			aList = ( aTitle , aClass , aFullPN )

			if self.theExist != 0:

				anIter = aModel.append()
				aModel.set( anIter,\
					    0, aList[0],\
					    1, aList[1],\
					    2, aList[2] )

	# end of update


	# -------------------------------------------------------------
	# update
	#    - update CList
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def editTitle( self , obj ):
		
		# if no data is selected, show error message.
		if self.theSelectedRow == None:
			anErrorMessage='\nNo data is selected.!\n'
			self.theMainWindow.printMessage( anErrorMessage )
			aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
			return None

		# if a data is selected, then remove it.
		else:

			aNewTitle = self[ "SelectedTitle" ].get_text() 
			aNewTitle = strip( aNewTitle )
	
			if len(aNewTitle) == 0:
				anErrorMessage='\nError text field is blank.!\n'
				self.theMainWindow.printMessage( anErrorMessage )
				aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
				return None

			aTitle =  self['InterfaceCList'].get_model().get_value( self.theSelectedRow ,0 )
			aTitleDict = self.theMainWindow.thePluginManager.thePluginTitleDict

			anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					self.theMainWindow.thePluginManager.editModuleTitle( anInstance, aNewTitle )
					break

		self.theMainWindow.updateFundamentalWindows()

	# end of editTitle

                    
	# -------------------------------------------------------------
	# update
	#    - update CList
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def rowSelected( self , obj ):

		row=self['InterfaceCList'].get_selection().get_selected()[1]
		self.theSelectedRow = row
		
		aText =  self['InterfaceCList'].get_model().get_value( row,0 )
		self[ "SelectedTitle" ].set_text( aText )


	# -------------------------------------------------------------
	# updateWindow
	#    - shows selected instance of plug-in window
	#
	# anObject : dammy object
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def showWindow( self , anObject ):

		# -------------------------------------------
		# show an Instance
		# -------------------------------------------

		# if no data is selected, show error message.
		if self.theSelectedRow == None:
			anErrorMessage='\nNo data is selected.!\n'
			self.theMainWindow.printMessage( anErrorMessage )
			aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
			return None

		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_model().get_value( self.theSelectedRow ,0 )
			aTitleDict = self.theMainWindow.thePluginManager.thePluginTitleDict

			anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					self.theMainWindow.thePluginManager.showPlugin( anInstance )
			
			#	if aTitle == aTitleDict[ anInstance ]:
			#		anInstance[ anInstance.__class__.__name__ ].hide()
			#		anInstance[ anInstance.__class__.__name__ ].show_all()
			#		break

	# end of showWindow

                    
	# -------------------------------------------------------------
	# deleteWindow
	#    - deletes selected instance of plug-in window
	#
	# anObject : dammy object
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def deleteWindow( self , anObject ):

		# -------------------------------------------
		# delete an Instance
		# -------------------------------------------

		# if no data is selected, show error message.
		if self.theSelectedRow == None:
			anErrorMessage='\nNo data is selected.!\n'
			self.theMainWindow.printMessage( anErrorMessage )
			aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
			return None

		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_model().get_value( self.theSelectedRow ,0 )
			aTitleDict = self.theMainWindow.thePluginManager.thePluginTitleDict

			anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					self.theMainWindow.thePluginManager.removeInstance( anInstance )
					break

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


