#!/usr/bin/env python

#from Window import *
from OsogoWindow import *
from gtk import *
from ecell.ecssupport import *
from string import *

#class InterfaceWindow( Window ):
class InterfaceWindow( OsogoWindow ):
    
	def __init__( self, aMainWindow ):

		#Window.__init__( self )
		OsogoWindow.__init__( self )
		OsogoWindow.openWindow(self)

		self.theMainWindow = aMainWindow
		self.theInterfaceList = self[ 'InterfaceCList' ]
		self.theInterfaceListBuffer = []
		self.thePluginWindowsNoDict = {}
		self.thePluginWindowsDict = {}
		self.theCount = 0
		self.theSelectedRow = -1
		self.theTitle = ""

		# __init__

		self.addHandlers( { 'interfacelist_select_row' : self.rowSelected,
		                    'ShowButton_clicked'       : self.showWindow,
		                    'SaveButton_clicked'       : self.editTitle,
		                    'DeleteButton_clicked'     : self.deleteWindow } )
	

	def update( self ):
	
		self.theInterfaceList.clear()
		self[ 'InterfaceCList' ].clear()

		for aList in self.theInterfaceListBuffer:
			#self.theInterfaceList.append( aList )
			self[ 'InterfaceCList' ].append( aList )

	def addNewRecord( self , classname , data ):
        
		if len(data) != 0:

			aClassName = classname[ : -6 ]
			aFullPN = createFullPNString( data[0] )
			self.theTitle = self.createNewTitle( aClassName )
			self.thePluginWindowsNoDict[ aClassName ] += 1
        
			alist = ( self.theTitle , aClassName , aFullPN )
			#self.theInterfaceList.append( alist )
			self.theInterfaceListBuffer.append( alist )
			self.thePluginWindowsDict[ self.theTitle ] = self.theCount
			self.theCount += 1

		self.update()

	def createNewTitle( self , aClassName ):

		aTitle = aClassName + str( self.thePluginWindowsNoDict[ aClassName ] + 1 )
		return aTitle

	def removeRecord( self , obj ):
		
		aWindowName = obj.getWidget( obj.__class__.__name__ )[ 'title' ]

		aRowNum = self.thePluginWindowsDict[ aWindowName ]

		for aKey in self.thePluginWindowsDict.keys():
			if self.thePluginWindowsDict[ aKey ] > aRowNum:
				self.thePluginWindowsDict[ aKey ] -= 1
			elif self.thePluginWindowsDict[ aKey ] == aRowNum:
				del self.thePluginWindowsDict[ aKey ]

		self.theInterfaceList.remove( aRowNum )
		self.theCount -= 1
		self[ "SelectedTitle" ].set_text( "" )

	def editTitle( self , obj ):
        
		try:
			if self[ "SelectedTitle" ].get_text() !=  "" and self[ "SelectedTitle" ].get_text() != self.theInterfaceList.get_text( self.theSelectedRow , 0 ):
				for aKey in self.thePluginWindowsDict.keys():
					if aKey == self[ "SelectedTitle" ].get_text():
						aSameNameFlag = 1    #aSameNameFlag = 1 if title already exists 
						break
					else:
						aSameNameFlag = 0

				if aSameNameFlag == 0:
					for aKey in self.thePluginWindowsDict.keys():
						if self.thePluginWindowsDict[ aKey ] == self.theSelectedRow:
							del self.thePluginWindowsDict[ aKey ]
			
					self.thePluginWindowsDict[ self[ "SelectedTitle" ].get_text() ] = self.theSelectedRow
					self.theInterfaceList.set_text( self.theSelectedRow , 0 , self[ "SelectedTitle" ].get_text() )
					self.theMainWindow.thePluginManager.getModule( self.theSelectedRow , self[ "SelectedTitle" ].get_text())

				else:
					self[ "SelectedTitle" ].set_text( self.theInterfaceList.get_text( self.theSelectedRow, 0 ) )

		except ValueError:
			pass
			
	def rowSelected( self , obj , row , column , data3 ):

		self.theSelectedRow = row
		self[ "SelectedTitle" ].set_text( self.theInterfaceList.get_text( self.theSelectedRow , 0 ))

	def showWindow( self , obj ):

		if self.theSelectedRow != -1:
			self.theMainWindow.thePluginManager.showPlugin( self.theSelectedRow , obj )
                    
	def deleteWindow( self , obj ):

		if self.theSelectedRow != -1:
			self.theMainWindow.thePluginManager.deleteModule( self.theSelectedRow , obj )

if __name__ == "__main__":

	def mainLoop():
		gtk.mainloop()

	def main():
			aWindow = InterfaceWindow( 'InterfaceWindow.glade' )
			mainLoop()

	main()


