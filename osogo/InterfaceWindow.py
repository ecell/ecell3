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
		OsogoWindow.openWindow(self)

		#self.theMainWindow = aMainWindow

		self.theSelectedRow = -1

		self.addHandlers( { 'interfacelist_select_row' : self.rowSelected,
		                    'ShowButton_clicked'       : self.showWindow,
		                    'SaveButton_clicked'       : self.editTitle,
		                    'DeleteButton_clicked'     : self.deleteWindow } )

		self.noSelectedMessage = " Nothing is selected. "

	# end of __init__
	
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
		aInstanceList = self.theMainWindow.thePluginManager.theInstanceList

		for anInstance in aInstanceList:
			aTitle = anInstance.getTitle()
			aClass =  anInstance.__class__.__name__
			aFullPN = createFullPNString( anInstance.theFullPN() )
			aList = ( aTitle , aClass , aFullPN )

			if self.theExist != 0:
				self['InterfaceCList'].append( aList )

	# end of update

	def editTitle( self , obj ):
		
		# if no data is selected, show error message.
		if self.theSelectedRow == -1:
			self.theMainWindow.printMessage( self.noSelectedMessage )
		# if a data is selected, then remove it.
		else:

			aNewTitle = self[ "SelectedTitle" ].get_text() 
			aNewTitle = strip( aNewTitle )
	
			if len(aNewTitle) == 0:
				self.theMainWindow.printMessage(" Error text field is blank. ")
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

                    
	def rowSelected( self , obj , row , column , data3 ):

		self.theSelectedRow = row
		aText =  self['InterfaceCList'].get_text( row,0 )
		self[ "SelectedTitle" ].set_text( aText )

	def showWindow( self , obj ):

		# -------------------------------------------
		# show an Instance
		# -------------------------------------------

		# if no data is selected, show error message.
		if self.theSelectedRow == -1:
			self.theMainWindow.printMessage( self.noSelectedMessage )
		# if a data is selected, then remove it.
		else:

			aTitle =  self['InterfaceCList'].get_text( self.theSelectedRow ,0 )
			aTitleDict = self.theMainWindow.thePluginManager.thePluginTitleDict

			anInstanceList = self.theMainWindow.thePluginManager.theInstanceList
			for anInstance in anInstanceList:
				if aTitle == aTitleDict[ anInstance ]:
					#self.theMainWindow.thePluginManager.showPlugin( anInstance )
					anInstance[ anInstance.__class__.__name__ ].hide()
					anInstance[ anInstance.__class__.__name__ ].show_all()
					break

	# end of showWindow

                    
	def deleteWindow( self , obj ):

		# -------------------------------------------
		# delete an Instance
		# -------------------------------------------

		# if no data is selected, show error message.
		if self.theSelectedRow == -1:
			self.theMainWindow.printMessage( self.noSelectedMessage )

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


