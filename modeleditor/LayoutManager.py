
from ModelEditor import *
from Layout import *
from LayoutBufferFactory import *
from PathwayCanvas import *
from GraphicalUtils import *

class LayoutManager:

	def __init__( self, aModelEditor):
		self.theModelEditor = aModelEditor
		self.theLayoutMap = {}
		self.theLayoutBufferFactory = LayoutBufferFactory( self.theModelEditor, self )
		self.theLayoutBufferPaster = LayoutBufferPaster( self.theModelEditor, self )
		self.theGraphicalUtils = GraphicalUtils( self.theModelEditor.theMainWindow )


	def createLayout( self, aLayoutName ):
		# create and show
		if aLayoutName in self.theLayoutMap.keys():
			raise Exception("Layout %s already exists!"%aLayoutName )
		newLayout = Layout ( self, aLayoutName )
		self.theLayoutMap[ aLayoutName ] = newLayout
		self.showLayout( aLayoutName )


	def deleteLayout( self, aLayoutName ):
		aLayout = self.theLayoutMap[ aLayoutName ]
		editorWindow = aLayout.getCanvas().getParentWindow()
		aLayout.detachFromCanvas()
		editorWindow.close()


	def showLayout( self, aLayoutName ):
		aLayout = self.theLayoutMap[ aLayoutName ]
		if aLayout.getCanvas() != None:
			return
		# create new pathwayeditor
		anEditorWindow = self.theModelEditor.createPathwayEditor( aLayout )



	def saveLayouts( self, aFileName ):
		# LATER
		pass


	def loadLayouts( self, aFileName ):
		# LATER
		pass


	def update( self, aType = None, anID = None ):
		# i am not sure this is necessary!!!
		pass


	def getUniqueLayoutName( self, tryThisName = None ):
		if tryThisName == None:
			tryThisName = 'Layout'
		nameList = self.theLayoutMap.keys()
		counter = 0
		layoutName = tryThisName
		while layoutName in nameList:
			layoutName = tryThisName + str( counter )
			counter += 1
		return layoutName


	def getLayoutNameList( self ):
		return self.theLayoutMap.keys()


	def getLayout( self, aLayoutName ):
		return self.theLayoutMap[ aLayoutName ]
		

	def doesLayoutExist( self, aLayoutName ):
		return aLayoutName in self.theLayoutMap.keys()
	

	def renameLayout( self, oldLayoutName, newLayoutName ):
		aLayout = self.theLayoutMap[ oldLayoutName ]
		if self.doesLayoutExist( newLayoutName ):
			raise Exception("%s layout already exists!"%newLayoutName )
		aLayout.rename( newLayoutName )
		self.theLayoutMap[ newLayoutName ] = aLayout
		self.theLayoutMap.__delitem__( oldLayoutName )
		
	
	def createObjectIterator( self ):
		return ObjectIterator( self)





class ObjectIterator:
	# cannot handle modifications to the layouts

	def __init__( self, aLayoutManager):
		self.theLayoutManager = aLayoutManager
		self.filterList = []
		self.reset()
		

	def reset( self ):
		self.layoutList = self.theLayoutManager.getLayoutNameList()
		self.objectList = []
		self.currentLayoutID = None
		self.currentObjectID = None
	

	def deleteFilters( self ):
		self.filterList = []
		self.reset()
	
	def filterByFullID( self, aFullID ):
		filterList.append( [ "FULLID", aFullID ] )
		
	def filterByProperty( self, aPropertyName, aPropertyValue ):
		filterList.append( [ "CUSTOMPROPERTY", aPropertyName, aPropertyValue ] )
	
	def filterByID( self, objectID ):
		filterList.append( "ID", objectID )


	def getNextObject( self ):
		# return first matching object self
		while self.__getNextObject != None:
			theLayout = self.theLayoutManager.getLayout( self.currentLayoutID )
			theObject = theLayout.getObject( self.currentObjectID )
			if self.doesComply( theObject ):
				return theObject	
		return None
	
	
	def doesComply( self, anObject ):
		complied = 0
		propertyList = anObject.getPropertyList()
		for aFilter in self.filterList:
			if aFilter[0] == "FULLID" and OB_FULLID in propertyList:
				if aFilter[1] == anObject.getProperty( OB_FULLID ):
					complied += 1
			elif aFilter[0] == "CUSTOMPROPERTY" and aFilter[1] in propertyList:
				if aFilter[2] == anObject.getProperty( aFilter[1] ):
					complied += 1
			elif aFilter[0] == "ID" :
				if aFilter[1] == anObject.getID( ):
					complied += 1
		return complied == len( self.filterList )


	
	def __getNextObject( self ):
		# if no objectlist or currentobject at the end of objectlist, get next layout
		if self.objectList != []:
			curpos = self.objectList.find( currentObject ) 
			if curpos != len( self.objectList ) - 1:
				curpos += 1
				self.currentObjectID = self.objectList[ curpos ]
				return self.currentObjectID
		self.__getNextLayout()
		if self.currentLayout != None:
			curpos = 0
			self.currentObjectID = self.objectList[ curpos ]
			return self.currentObjectID
		else:
			return None
					

		
	def __getNextLayout( self ):
		# get a layout that contains at least one object
		# if last layout return None
		# fill in objectlist
		if self.layoutList == []:
			self.currentLayout = None
			return None
		if self.currentLayoutID == None:
			curpos = 0
		else:
			curpos = self.layoutList.find( self.currentLayoutID )
			curpos += 1
		while curpos < len( self.layoutList ):
			self.currentLayoutID = self.layoutList[0]
			layout = self.theLayoutManager.getLayout( self.currentLayoutID )
			self.objectList = layout.getObjectList()
			if self.objectList != []:
				return None
			curpos += 1
		self.currentLayout = None
		return None
