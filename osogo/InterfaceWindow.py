#!/usr/bin/env python

from OsogoWindow import *
from gtk import *
from ecell.ecssupport import *
from string import *

class InterfaceWindow( OsogoWindow ):
    
	# -------------------------------------------------------------
	# Constructor
	# -------------------------------------------------------------
	def __init__( self, aMainWindow ):

		OsogoWindow.__init__( self, aMainWindow )
		self.theSelectedRow = -1

	# end of __init__

	# -------------------------------------------------------------
	# openWindow
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def openWindow( self ):
		OsogoWindow.openWindow(self)
		self.theSelectedRow = -1
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
	
		if self.theExist != 0:
			self[ 'InterfaceCList' ].clear()

		# gets
		anInstanceList = self.theMainWindow.thePluginManager.theInstanceList

		for anInstance in anInstanceList:
			aTitle = anInstance.getTitle()
			aClass =  anInstance.__class__.__name__
			aFullPN = createFullPNString( anInstance.theFullPN() )
			aList = ( aTitle , aClass , aFullPN )

			if self.theExist != 0:
				self['InterfaceCList'].append( aList )

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
		if self.theSelectedRow == -1:
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

			aTitle =  self['InterfaceCList'].get_text( self.theSelectedRow ,0 )
			aTitleDict = self.theMainWindow.thePluginManager.thePluginTitleDict

			anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					self.theMainWindow.thePluginManager.editModuleTitle( anInstance, aNewTitle )
					break

		self.theMainWindow.updateBasicWindows()

	# end of editTitle

                    
	# -------------------------------------------------------------
	# update
	#    - update CList
	#
	# return -> None
	# This method is throwable exception
	# -------------------------------------------------------------
	def rowSelected( self , obj , row , column , data3 ):

		self.theSelectedRow = row
		aText =  self['InterfaceCList'].get_text( row,0 )
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
		if self.theSelectedRow == -1:
			anErrorMessage='\nNo data is selected.!\n'
			self.theMainWindow.printMessage( anErrorMessage )
			aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
			return None

		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_text( self.theSelectedRow ,0 )
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
		if self.theSelectedRow == -1:
			anErrorMessage='\nNo data is selected.!\n'
			self.theMainWindow.printMessage( anErrorMessage )
			aWarningWindow = ConfirmWindow(0,anErrorMessage,"!")
			return None

		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_text( self.theSelectedRow ,0 )
			aTitleDict = self.theMainWindow.thePluginManager.thePluginTitleDict

			anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					self.theMainWindow.thePluginManager.removeInstance( anInstance )
					break

			# clear select status
			self.theSelectedRow = -1

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


