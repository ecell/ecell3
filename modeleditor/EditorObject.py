from ComplexShape import *
from Constants import *
from LayoutCommand import *
import gtk
from EntityCommand import *

class EditorObject:


	def __init__(self, aLayout, objectID,  x, y, parentSystem ):
		self.theLayout = aLayout
		self.theID = objectID
		self.parentSystem = parentSystem
		self.thePropertyMap = {}
		self.thePropertyMap[ OB_POS_X ] = x
		self.thePropertyMap[ OB_POS_Y ] = y
		self.thePropertyMap[ OB_HASFULLID ] = False

		self.theCanvas = None
		self.theShape = None
		# default colors
		self.thePropertyMap [ OB_OUTLINE_COLOR ] = self.theLayout.graphUtils().getRRGByName("black")
		self.thePropertyMap [ OB_FILL_COLOR ] = self.theLayout.graphUtils().getRRGByName("white")
		self.thePropertyMap [ OB_TEXT_COLOR ] = self.theLayout.graphUtils().getRRGByName("blue")
		# default outline
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 1
		self.parentSystem.registerObject( self )
		self.theSD = None
		self.isSelected = False



	def destroy(self):
		if self.theShape != None:
			self.theShape.delete()

	def hide( self ):
		# deletes it from canvas
		pass

	def doSelect( self ):
		if not self.isSelected:
			self.theLayout.selectRequest( self.theID )

	def selected( self ):
		if not self.isSelected:
			self.isSelected = True
			self.theShape.selected()
			self.theShape.outlineColorChanged()


	def unselected( self ):
		if self.isSelected:
			self.isSelected = False
			self.theShape.unselected()
			self.theShape.outlineColorChanged()



	def showMenu( self, anEvent ):
		aMenuDict = self.getMenuItems()
		if aMenuDict.keys() == []:
			return
		aMenu = gtk.Menu()

		for aMenuName in aMenuDict.keys():
			menuItem = gtk.MenuItem( aMenuName )
			menuItem.connect( 'activate', aMenuDict[ aMenuName ] )
			aMenu.add(menuItem)
		self.theMenu = aMenu
		aMenu.show_all()
		aMenu.popup(None, None, None, anEvent.button, anEvent.time)



	def outlineDragged( self, deltax, deltay, x, y ):
		# in most of the cases object are not resizeable, only system is resizeable and it will override this
		self.objectDragged( )
		pass

	def objectDragged( self, deltax, deltay ):
		#  parent system boundaries should be watched here!!!
		#get new positions:
		# currently move out of system is not supported!!!
		if self.parentSystem.__class__.__name__ == 'Layout':
			#rootsystem cannot be moved!!!
			return

		newx = self.getProperty( OB_POS_X ) + deltax
		newy = self.getProperty( OB_POS_Y ) + deltay
		if (newy + self.getProperty( OB_DIMENSION_Y ) > self.parentSystem.getProperty( SY_INSIDE_DIMENSION_Y )) and deltay> 0:
			newy = self.getProperty( OB_POS_Y )
		if ( newx + self.getProperty( OB_DIMENSION_X ) > self.parentSystem.getProperty( SY_INSIDE_DIMENSION_X  ) ) and deltax > 0:
			newx = self.getProperty( OB_POS_X )
		if newx <0 and deltax< 0:
			newx = self.getProperty( OB_POS_X )
		if newy <0 and deltay< 0:
			newy = self.getProperty( OB_POS_Y )
		

		if newx==0 and newy == 0:
			return
		aCommand = MoveObject( self.theLayout, self.theID, newx, newy, None )
		self.theLayout.passCommand( [ aCommand ] )


	def popupEditor( self ):

		self.theLayout.popupObjectEditor( self.theID )		


	def setLimits( self, x0, y0, x1, y1 ):
		pass


	def setCanvas( self, aCanvas ):
		self.theCanvas = aCanvas


	def show( self ):
		self.theShape = ComplexShape( self, self.theCanvas, self.thePropertyMap[ OB_POS_X ], self.thePropertyMap[ OB_POS_Y ], self.thePropertyMap[ OB_DIMENSION_X ], self.thePropertyMap[ OB_DIMENSION_Y ] )
		self.theShape.show()


	def setProperty( self, aPropertyName, aPropertyValue ):
		# if fullID property is changed, change legend, too
		pass


	def getProperty( self, aPropertyName ):
		if aPropertyName in self.thePropertyMap.keys():
			return self.thePropertyMap[aPropertyName]
		else:
			raise Exception("Unknown property %s for object %s"%(aPropertyName, self.theID ) )


	def getPropertyList( self ):
		return self.thePropertyMap.keys()


	def getAbsolutePosition( self ):
		( xpos, ypos ) = self.parentSystem.getAbsoluteInsidePosition()
		return ( xpos + self.thePropertyMap[ OB_POS_X ], ypos + self.thePropertyMap[ OB_POS_Y ] )


	def getPropertyMap( self ):
		return self.thePropertyMap

	def setPropertyMap( self, aPropertyMap ):
		self.thePropertyMap = aPropertyMap

	def getID( self ):
		return self.theID
		
	def getLayout( self ):
		return self.theLayout


	def getParent( self ):
		return self.parentSystem

	def getGraphUtils( self ):
		return self.theLayout.graphUtils()

	def getModelEditor( self ):
		return self.theLayout.theLayoutManager.theModelEditor

	def getCursorType( self, aFunction, x, y, buttonPressed ):
		if aFunction in [ SD_FILL, SD_TEXT ] and buttonPressed:
			return CU_MOVE
		elif aFunction  in [ SD_FILL, SD_TEXT ] and not buttonPressed:
			return CU_POINTER
		return CU_POINTER

	def move( self, deltax, deltay ):
		self.thePropertyMap[ OB_POS_X ] += deltax
		self.thePropertyMap[ OB_POS_Y ] += deltay
		self.theShape.move( deltax, deltay )

	def getMenuItems( self, aSubMenu = None ):
		menuDict = {}
		if self.parentSystem.__class__.__name__ != 'Layout':
			menuDict [ 'delete from layout'] = self.__userDeleteObject 
			if self.getProperty( OB_HASFULLID ):
				menuDict['delete_from_model'] = self.__userDeleteEntity 
		return menuDict

	def __userDeleteObject( self, *args ):
		self.theMenu.destroy()
		aCommand = DeleteObject( self.theLayout, self.theID )
		self.theLayout.passCommand( [ aCommand ] )


	def __userDeleteEntity ( self, *args ):

		self.theMenu.destroy()
		aModelEditor = self.theLayout.theLayoutManager.theModelEditor
		aFullID = self.getProperty( OB_FULLID )
		aCommand = DeleteEntityList( aModelEditor, [aFullID ] )
		self.theLayout.passCommand( [ aCommand ] )


