
from Constants import *
import os
import gtk
from ConfirmWindow import *
from Command import *
from CommandQueue import *
from EntityCommand import *
from StepperCommand import *
import string
import traceback
import sys
from ModelStore import *
import time

from LayoutManagerWindow import *
from MainWindow import *
from StepperWindow import *
from EntityListWindow import *
from DMInfo import *
from PopupMenu import *
from PathwayEditor import *
from LayoutManager import *
from AboutModelEditor import *


from CommandMultiplexer import *
from ObjectEditorWindow import *
from ConnectionObjectEditorWindow import *
from LayoutEml import *
from GraphicalUtils import *

from Error import *

RECENTFILELIST_FILENAME = '.modeleditor' + os.sep + '.recentlist'
RECENTFILELIST_DIRNAME = '.modeleditor'
RECENTFILELIST_MAXFILES = 10


class ModelEditor:

	"""
	class for holding together Model, Command, Buffer classes
	loads and saves model
	handles undo queue
	display messages
	"""

	def __init__( self, aFileName = None ):
		self.copyBuffer = None

		# set up load and save dir
		self.loadDirName = os.getcwd()
		self.saveDirName = os.getcwd()

		self.theDMInfo = DMInfo()

		# create variables
		self.__loadRecentFileList()

		self.__theModel = None
		self.theModelName = ''
		self.theModelFileName = ''

		self.theStepperWindowList = []
		self.theEntityListWindowList = []
		self.thePathwayEditorList = []
		self.theLayoutManagerWindow=None 
		self.theObjectEditorWindow = None
		self.theConnObjectEditorWindow = None
		self.theAboutModelEditorWindow = None
		self.theFullIDBrowser = None
		self.thePopupMenu = PopupMenu( self )
		self.theMainWindow = MainWindow( self )
		self.changesSaved = True
		self.openAboutModelEditor = False
		# create untitled model
		self.__createModel()

		# set up windows
		self.theMainWindow.openWindow()
		self.theGraphicalUtils = GraphicalUtils( self.theMainWindow )
		
		# load file
		if aFileName != None:
			self.loadModel( aFileName )


	def quitApplication( self ):
		if not self.__closeModel():
			return False
		gtk.mainquit()
		return True

	
	def createNewModel( self ):
		""" 
		in:
		returns True if successful, false if not
		"""

		# create new Model
		self.__createModel()
		self.updateWindows()


	def validateModel( self):
		self.printMessage("Sorry, not implemented!", ME_ERROR )


	def loadModel ( self, aFileName ):
		"""
		in: nothing
		returns nothing
		"""
		# check if it is dir
		if os.path.isdir( aFileName ):
			self.loadDirName = aFileName.rstrip('/')
			return

		# check if it is eml fileself.printMessage( "Only one Layout Window allowed", ME_PLAINMESSAGE )
		if not os.path.isfile( aFileName ):
			self.printMessage("%s file cannot be found!"%aFileName, ME_ERROR )
			return
		self.loadDirName = os.path.split( aFileName )[0]
		if self.loadDirName != '':
			os.chdir(self.loadDirName )


		# create new model
		self.__createModel()

		# tries to parse file
		try:
			aFileObject = open( aFileName, 'r' )
			anEml = Eml( aFileObject )


			# calls load methods
			self.__loadStepper( anEml )
			self.__loadEntity( anEml )
			self.__loadProperty( anEml )
	
		except:
			#display message dialog
			
			self.printMessage( "Error loading file %s"%aFileName, ME_ERROR )
			
			if str(sys.exc_type)=="Error.ClassNotExistError":
				anErrorMessage= string.join( sys.exc_value )
			else:
				anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
					sys.exc_traceback), '\n' )
			
			self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
			self.__createModel()
			return
		# load layouts
		try:
			self.loadLayoutEml( os.path.split( aFileName ) )
		except:
			#display message dialog
			self.printMessage( "Error loading layout information from file %s"%aFileName, ME_ERROR )
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
					sys.exc_traceback), '\n' )
			self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
		# log to recent list
		self.__addToRecentFileList ( aFileName )
		self.theModelFileName = aFileName
		self.theModelName = os.path.split( aFileName )[1]

		self.modelHasName = True
		self.changesSaved = True
		self.printMessage("Model %s loaded successfully."%aFileName )
		self.updateWindows()




	def saveModel ( self,  aFileName = None ):
		"""
		in: bool saveAsFlag
		returns nothing
		"""
		# check if it is dir
		if os.path.isdir( aFileName ):
			self.saveDirName = aFileName.rstrip('/')
			return

		# check if it is eml file
		self.saveDirName = os.path.split( aFileName )[0]

		anEml = Eml()


		self.__saveStepper( anEml )

		# creates root entity
		anEml.createEntity('System', 'System::/')
		
		# calls save methods
		self.__saveEntity( anEml )
		self.__saveProperty( anEml )

		# if the type is string

		# add comment
		aCurrentInfo = '''<!-- created by ecell ModelEditor
 date: %s

-->
<eml>
''' % time.asctime( time.localtime() )
   		aString = anEml.asString()
		anEml.destroy()
		aBuffer = aString #+ aCurrentInfo
		try:
			aFileObject = open( aFileName, 'w' )
			aFileObject.write( aBuffer )
			aFileObject.close()
		except:
			#display message dialog
			self.printMessage( "Error saving file %s"%aFileName, ME_ERROR )
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
					sys.exc_traceback), '\n' )
			self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
			return
		# save layout eml
		self.saveLayoutEml(os.path.split( aFileName ))

		# log to recent list
		self.__addToRecentFileList ( aFileName )
		self.theModelFileName = aFileName
		self.printMessage("Model %s saved successfully."%aFileName )
		self.theModelName = os.path.split( aFileName )[1]
		self.modelHasName = True
		self.changesSaved = True
		self.updateWindows()

	def lemlExist(self,fileName):
		if os.path.isfile(fileName):
			return True
		else:
			return False

	def loadLayoutEml( self, (emlPath, emlName) ):
		fileName = os.path.join( emlPath, self.__getLemlFromEml( emlName ) )
		if not self.lemlExist(fileName):
			return
		fileObject = open(fileName, "r")
		aLayoutEml = LayoutEml(fileObject)

		aLayoutNameList =  aLayoutEml.getLayoutList()
		# create layouts

		for aLayoutName in  aLayoutNameList:
			self.theLayoutManager.createLayout(aLayoutName)
			aLayout = self.theLayoutManager.getLayout(aLayoutName)
		# create layoutproperties
		
			propList = aLayoutEml.getLayoutPropertyList(aLayoutName)
			for aProp in propList:
				aPropValue =aLayoutEml.getLayoutProperty(aLayoutName,aProp)
                                if aPropValue == "False":
                                        aPropValue=False
                                elif aPropValue == "True":
                                        aPropValue=True
				aLayout.setProperty(aProp,aPropValue)
				
			aRootID = aLayout.getProperty( LO_ROOT_SYSTEM )
			self.__loadObject( aLayout, aRootID, aLayoutEml, None)
			aConnObjectList = aLayoutEml.getObjectList(OB_TYPE_CONNECTION, aLayoutName)
			# finally read and create connections for layout
			for aConnID in aConnObjectList:
				aProAttachedID = aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_PROCESS_ATTACHED)
				aVarAttachedID =aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_VARIABLE_ATTACHED)
				aProRing = aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_PROCESS_RING)
				aVarRing = aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_VARIABLE_RING)
				aVarrefName =  aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_NAME)
				aLayout.createConnectionObject(aConnID, aProAttachedID, aVarAttachedID,  aProRing, aVarRing,PROCESS_TO_VARIABLE, aVarrefName )
				anObject =aLayout.getObject(aConnID)
				propList = aLayoutEml.getObjectPropertyList(aLayoutName, aConnID )
				for aProp in propList:
					if aProp in (CO_PROCESS_ATTACHED, CO_VARIABLE_ATTACHED,CO_PROCESS_RING,CO_VARIABLE_RING, CO_NAME):
						continue
					aPropValue = aLayoutEml.getObjectProperty(aLayoutName, aConnID,aProp)
					if aPropValue == None:
						continue
                                        if aPropValue=="False":
                                                aPropValue=False
                                        elif aPropValue=="True":
                                                aPropValue=True
					anObject.setProperty(aProp,aPropValue)


		

	def __loadObject( self, aLayout, anObjectID,  aLayoutEml, parentSys ):
		aLayoutName = aLayout.getName()
		propList = aLayoutEml.getObjectPropertyList(aLayoutName, anObjectID )

		objectType = aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_TYPE)
		aFullID = aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_FULLID)
		if objectType != OB_TYPE_CONNECTION:
			x=aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_POS_X)
			y=aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_POS_Y)

			aLayout.createObject(anObjectID, objectType, aFullID,x, y, parentSys )
			anObject =aLayout.getObject(anObjectID)
			for aProp in propList:
				if aProp in ( OB_TYPE,OB_FULLID,OB_POS_X,OB_POS_Y) :
					continue
				aPropValue = aLayoutEml.getObjectProperty(aLayoutName, anObjectID,aProp)
				if aPropValue == None :
					continue
                                if aPropValue == "False":
                                        aPropValue = False
                                elif aPropValue == "True":
                                        aPropValue = True
				anObject.setProperty(aProp,aPropValue)
		# create subobjects
		if objectType == OB_TYPE_SYSTEM:
			anObjectList = self.__getObjectList(aLayoutEml,aLayoutName, anObjectID)
			for anID in anObjectList:
				self.__loadObject( aLayout, anID, aLayoutEml,anObject )
		
		
	def __getObjectList(self,aLayoutEml,aLayoutName,aParentID):
		anObjectList=[]
		aProObjectList = aLayoutEml.getObjectList(OB_TYPE_PROCESS, aLayoutName, aParentID)
		aVarObjectList = aLayoutEml.getObjectList(OB_TYPE_VARIABLE, aLayoutName, aParentID)
		aSysObjectList = aLayoutEml.getObjectList(OB_TYPE_SYSTEM, aLayoutName, aParentID)
		aTextObjectList = aLayoutEml.getObjectList(OB_TYPE_TEXT, aLayoutName, aParentID)
		anObjectList = aProObjectList + aVarObjectList + aSysObjectList+ aTextObjectList
		return anObjectList



	def saveLayoutEml( self, (emlPath, emlName) ):
		fileName = os.path.join( emlPath, self.__getLemlFromEml( emlName ) )
		aLayoutEml = LayoutEml()
		# create layouts
		for aLayoutName in self.theLayoutManager.getLayoutNameList():
			self.__saveLayout( aLayoutName, aLayoutEml )

		#aCurrentInfo = '''<!-- created by ecell ModelEditor
# date: %s
#
#-->
#<leml>
#''' % time.asctime( time.localtime() )
   		aString = aLayoutEml.asString()
		aLayoutEml.destroy()
#		aBuffer = string.join( string.split(aString, '<leml>\n'), aCurrentInfo)
		aBuffer = aString
		try:
			aFileObject = open( fileName, 'w' )
			aFileObject.write( aBuffer )
			aFileObject.close()
		except:
			#display message dialog
			self.printMessage( "Error saving file %s"%fileName, ME_ERROR )
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
					sys.exc_traceback), '\n' )
			self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
			



	def __saveLayout( self, aLayoutName, aLayoutEml ):
		aLayoutEml.createLayout( aLayoutName )
		# save properties
		aLayoutObject = self.theLayoutManager.getLayout( aLayoutName )
		propList = aLayoutObject.getPropertyList()
		for aProperty in propList:
			aValue = aLayoutObject.getProperty( aProperty)
			aLayoutEml.setLayoutProperty( aLayoutName, aProperty, aValue )
		# save objects
		aRootID = aLayoutObject.getProperty( LO_ROOT_SYSTEM )
		self.__saveObject( aLayoutObject, aRootID, aLayoutEml )
		# save connections
		conList = aLayoutObject.getObjectList( OB_TYPE_CONNECTION )
		for aConID in conList:
			self.__saveObject( aLayoutObject, aConID, aLayoutEml )


	def __saveObject( self, aLayoutObject, anObjectID,  aLayoutEml, parentID = 'layout' ):
		aLayoutName = aLayoutObject.getName()
		# create object
		anObject = aLayoutObject.getObject( anObjectID )
		aType = anObject.getProperty( OB_TYPE )
		aLayoutEml.createObject(  aType, aLayoutName, anObjectID, parentID )

		# save properties
		propList = anObject.getPropertyList()
		for aProperty in propList:
			if aProperty in ( PR_CONNECTIONLIST, VR_CONNECTIONLIST ):
				continue
			aValue = anObject.getProperty( aProperty)

			aLayoutEml.setObjectProperty( aLayoutName, anObjectID, aProperty, aValue )

		# save subobjects
		if aType == OB_TYPE_SYSTEM:
			for anID in anObject.getObjectList():
				self.__saveObject( aLayoutObject, anID, aLayoutEml, anObjectID )
			
		


	def __getLemlFromEml( self, emlName ):
		trunk = emlName.split('.')
		if trunk[ len(trunk)-1] == 'eml':
			trunk.pop()
		return '.'.join( trunk ) + '.leml'
	

	def closeModel ( self ):
		"""
		in: nothing
		return: nothing
		"""
		# call close model
		self.__closeModel()



	def getUniqueEntityName ( self, aType, aSystemPath ):
		"""
		in: string aType, string aSystemPath
		returns string entityname or none if atype or asystempath doesnot exists
		"""

		# get entitylist in system
		anEntityNameList  = self.theModelStore.getEntityList( aType, aSystemPath )

		# call getUniqueName
		return self.__getUniqueName( aType, anEntityNameList )
		



	def getUniqueStepperName (self ):
		"""
		in: nothing
		return string stepper name 
		"""
		# get stepper list
		aStepperIDList = self.theModelStore.getStepperList()

		# call getUniquename
		return self.__getUniqueName( 'Stepper', aStepperIDList )




	def getUniqueEntityPropertyName ( self, aFullID ):
		"""
		in: string aFullID
		returns new unique propertyname
		"""
		# get Propertylist
		aPropertyList = self.theModelStore.getEntityPropertyList( aFullID )

		# call getUniquename
		return self.__getUniqueName( 'Property', aPropertyList )


	def getUniqueStepperPropertyName ( self, anID ):
		"""
		in: string anIDon_LayoutButton_toggled
		returns new unique propertyname
		"""
		# get Propertylist
		aPropertyList = self.theModelStore.getStepperPropertyList( anID )

		# call getUniquename
		return self.__getUniqueName( 'Property', aPropertyList )


	def getDMInfo( self ):
		return self.theDMInfo


	def getDefaultStepperClass( self ):
		return self.theDMInfo.getClassList( ME_STEPPER_TYPE )[0]


	def getDefaultProcessClass( self ):
		return self.theDMInfo.getClassList( ME_PROCESS_TYPE )[0]


	def getRecentFileList (self):
		"""
		in: nothing
		returns list of recently saved/loaded files
		"""
		return self.__theRecentFileList



	def getModel ( self ):
		return self.theModelStore



	def printMessage( self, aMessageText, aType = ME_PLAINMESSAGE ):
		"""
		Types:
		confirmation
		plain message
		alert
		"""
		if aType == ME_PLAINMESSAGE:
			if self.theMainWindow.exists():
				self.theMainWindow.displayMessage( aMessageText )
			else:
				print aMessageText
		elif aType == ME_OKCANCEL:
			msgWin = ConfirmWindow( OKCANCEL_MODE, aMessageText, 'Message')
			return msgWin.return_result()
		elif aType == ME_YESNO:
			msgWin = ConfirmWindow( OKCANCEL_MODE, aMessageText, 'Message')
			return msgWin.return_result()
		elif aType == ME_WARNING:
			msgWin = ConfirmWindow( OK_MODE, aMessageText, 'Warning!')
		elif aType == ME_ERROR:
			msgWin = ConfirmWindow( OK_MODE, aMessageText, 'ERROR!')

	def createPopupMenu( self, aComponent, anEvent ):
		self.setLastUsedComponent( aComponent )

		self.thePopupMenu.open(  anEvent )



	def doCommandList( self, aCommandList ):
		undoCommandList = []
		aCommandList = self.theMultiplexer.processCommandList( aCommandList )
		for aCommand in aCommandList:
			# execute commands
			aCommand.execute()

			( aType, anID ) = aCommand.getAffectedObject()
			
			self.updateWindows( aType, anID )
			( aType, anID ) = aCommand.getSecondAffectedObject()
			if aType == None and anID == None:
				pass
			else:
				self.updateWindows( aType, anID )
	
			# get undocommands
			undoCommand = aCommand.getReverseCommandList()
			
			# put undocommands into undoqueue
			if undoCommand != None:
				undoCommandList.extend( undoCommand )

			# reset commands put commands into redoqueue
			aCommand.reset()

		if undoCommandList != []:
			self.theUndoQueue.push ( undoCommandList )
			self.theRedoQueue.push ( aCommandList )
			self.changesSaved = False
		self.theMainWindow.update()
		


	def canUndo( self ):
		if self.theUndoQueue == None:
			return False
		return self.theUndoQueue.isPrevious()

	def canRedo( self ):
		if self.theRedoQueue == None:
			return False
		return self.theRedoQueue.isNext()


	def undoCommandList( self ):
		aCommandList = self.theUndoQueue.moveback()
		self.theRedoQueue.moveback()
		cmdList = aCommandList[:]
		cmdList.reverse()
		for aCommand in cmdList:
			# execute commands
			aCommand.execute()
			( aType, anID ) = aCommand.getAffectedObject()
			self.updateWindows( aType, anID )
			( aType, anID ) = aCommand.getSecondAffectedObject()
			if aType == None and anID == None:
				pass
			else:
				self.updateWindows( aType, anID )

			aCommand.reset()
		self.changesSaved = False


	def redoCommandList( self ):
		aCommandList = self.theRedoQueue.moveforward()
		self.theUndoQueue.moveforward()
		for aCommand in aCommandList:
			# execute commands
			#try:
			aCommand.execute()

			#except:
			#	self.printMessage( 'Error in operation.', ME_ERROR )
			#	anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
			#		sys.exc_traceback), '\n' )
			#	self.printMessage( anErrorMessage, ME_PLAINMESSAGE )

			( aType, anID ) = aCommand.getAffectedObject()
			self.updateWindows( aType, anID )
			( aType, anID ) = aCommand.getSecondAffectedObject()
			if aType == None and anID == None:
				pass
			else:
				self.updateWindows( aType, anID )

			aCommand.reset()
		self.changesSaved = False


	def createStepperWindow( self ):
		newWindow = StepperWindow( self )
		newWindow.openWindow()
		self.theStepperWindowList.append( newWindow )

	def createEntityWindow( self ):
		newWindow = EntityListWindow( self )
		newWindow.openWindow()
		self.theEntityListWindowList.append( newWindow )


	def createPathwayEditor( self, aLayout ):
		
		newWindow = PathwayEditor( self, aLayout )
		newWindow.openWindow()
		self.thePathwayEditorList.append( newWindow )
		return newWindow
		
		pass
#		newWindow = PathwayEditor( self, args )
#		newWindow.openWindow()
#		self.thePathwayEditorList.append( newWindow )

	def toggleOpenLayoutWindow(self,isOpen):
		self.openLayoutWindow=isOpen

	def createLayoutWindow( self ):
		if not self.openLayoutWindow:
			newWindow = LayoutManagerWindow( self )
			self.theLayoutManagerWindow=newWindow
			newWindow.openWindow()
			self.openLayoutWindow=True
		else:		
			self.theLayoutManagerWindow.present()

	def createObjectEditorWindow(self, aLayoutName, anObjectID ):
		if not self.openObjectEditorWindow:
			ObjectEditorWindow(self, aLayoutName, anObjectID)
		else:
			self.theObjectEditorWindow.setDisplayObjectEditorWindow( aLayoutName, anObjectID)
		
	def toggleObjectEditorWindow(self,isOpen,anObjectEditor):
		self.theObjectEditorWindow=anObjectEditor
		self.openObjectEditorWindow=isOpen

	def deleteObjectEditorWindow( self ):
		self.theObjectEditorWindow = None

	def createConnObjectEditorWindow(self, aLayoutName, anObjectID ):
		if not self.openConnObjectEditorWindow:
			ConnectionObjectEditorWindow(self, aLayoutName, anObjectID)
		else:
			self.theConnObjectEditorWindow.setDisplayConnObjectEditorWindow( aLayoutName, anObjectID)
		
	def toggleConnObjectEditorWindow(self,isOpen,aConnObjectEditor):
		self.theConnObjectEditorWindow=aConnObjectEditor
		self.openConnObjectEditorWindow=isOpen


	def createAboutModelEditor(self):
		if not self.openAboutModelEditor:
			AboutModelEditor(self)

	def toggleAboutModelEditor(self,isOpen,anAboutModelEditorWindow):
		self.theAboutModelEditorWindow = anAboutModelEditorWindow
		self.openAboutModelEditor=isOpen
			
		
	def copy(self ):
		self.theLastComponent.copy()


	def cut( self ):
		self.theLastComponent.cut()


	def paste(self ):
		self.theLastComponent.paste()


	def getCopyBuffer( self ):
		return self.copyBuffer



	def setCopyBuffer( self, aBuffer ):
		self.copyBuffer = aBuffer



	def setLastUsedComponent( self, aComponent ):
		self.theLastComponent = aComponent
		self.theMainWindow.update()



	def getLastUsedComponent( self  ):
		return self.theLastComponent 


	
	def getADCPFlags( self ):
		if self.theLastComponent == None:
			return [ False, False, False, False ]
		else:
			if self.copyBuffer == None:
				aType = None
			else:
				aType = self.copyBuffer.getType()
				
			return self.theLastComponent.getADCPFlags( aType )

	def setFullIDBrowser( self, aBrowser):
		self.theFullIDBrowser = aBrowser


	def updateWindows( self, aType = None, anID = None ):
		
		# aType None means nothing to be updated
		for aStepperWindow in self.theStepperWindowList:
			# anID None means all for steppers
			aStepperWindow.update( aType, anID )
		for anEntityWindow in self.theEntityListWindowList:
			anEntityWindow.update( aType, anID )
		if self.theObjectEditorWindow!=None:
			self.theObjectEditorWindow.update(aType, anID)
		if self.theConnObjectEditorWindow!=None:
			self.theConnObjectEditorWindow.update(aType, anID)
		for aPathwayEditor in self.thePathwayEditorList:
			aPathwayEditor.update( aType, anID )
		if self.theFullIDBrowser != None:
			self.theFullIDBrowser.update( aType,anID )
		if self.theMainWindow.exists():
			self.theMainWindow.update()
		#self.theLayoutManager.update( aType, anID )
		if self.theLayoutManagerWindow!=None:
			self.theLayoutManagerWindow.update()
		

	def __closeModel ( self ):
		""" 
		in: Nothing
		returns True if successful, False if not
		"""
		if not self.changesSaved:
			if self.printMessage("Changes are not saved.\n Are you sure you want to close %s?"%self.theModelName,ME_OKCANCEL)  == ME_RESULT_CANCEL:
			
				return False

		# close ModelStore
		self.theModelStore = None
		self.theLayoutManager = None
		self.theMultiplexer = None

		# set name 
		self.theModelName = ''
		self.theUndoQueue = None
		self.theRedoQueue = None
		self.changesSaved = True
		self.theLayoutManager = None
		self.__closeWindows()
		#self.updateWindows()
		return True


	def __closeWindows( self ):
		for aStepperWindow in self.theStepperWindowList:
			# anID None means all for steppers
			aStepperWindow.close( )
		for anEntityWindow in self.theEntityListWindowList:
			anEntityWindow.close( )

		for aPathwayEditor in self.thePathwayEditorList:
			aPathwayEditor.close( )
		if self.theConnObjectEditorWindow!=None:
			self.theConnObjectEditorWindow.destroy()
		if self.theObjectEditorWindow!=None:
			self.theObjectEditorWindow.destroy()
		if self.theLayoutManagerWindow!=None:
			self.theLayoutManagerWindow.close()
			self.theLayoutManagerWindow=None
			self.toggleOpenLayoutWindow(False)


	def __createModel (self ):
		""" 
		in: nothing
		out nothingfrom EntityListWindow import *
		"""
		self.__closeModel()
		# create new model
		self.theModelStore = ModelStore()
		self.theLayoutManager = LayoutManager( self )
		self.theMultiplexer = CommandMultiplexer( self, self.theLayoutManager )
		self.theUndoQueue = CommandQueue(MAX_REDOABLE_COMMAND)
		self.theRedoQueue = CommandQueue(MAX_REDOABLE_COMMAND)
		
		# initDM

		# set name
		self.theModelName = 'Untitled'
		self.modelHasName = False
		self.changesSaved = True
		self.openLayoutWindow = False
		self.openObjectEditorWindow = False
		self.openConnObjectEditorWindow = False
		self.theLastComponent = None




	def __loadRecentFileList ( self ):
		"""
		in: nothing
		returns nothing
		"""
		self.__theRecentFileList = []
		aFileName = self.__getHomeDir() + os.sep +  RECENTFILELIST_FILENAME

		# if not exists create it
		if not os.path.isfile( aFileName ):
			self.__saveRecentFileList()
		else:
			# if exists open it
			aFileObject = open ( aFileName, 'r' )

			# parse lines
			aLine = aFileObject.readline()
			while aLine != '':
				self.__theRecentFileList.append( aLine.strip( '\n' ) )
				aLine = aFileObject.readline()
				
			# close file
			aFileObject.close()


	def __addToRecentFileList (self, aFileName ):
		"""
		in: string aFileName (including directory, starting from root)
		returns nothing
		"""

		# check wheter its already on list
		for aFile in self.__theRecentFileList:
			if aFile == aFileName:
				return		

		# insert new item as first
		self.__theRecentFileList.insert ( 0, aFileName )

		# if length is bigger than max delete last 
		if len(self.__theRecentFileList) > RECENTFILELIST_MAXFILES:
			self.__theRecentFileList.pop()

		# save to file whole list
		self.__saveRecentFileList()

		
	def __getHomeDir ( self ):
		"""
		in: nothing
		returns the Home directory of user
		workaround for a bug in Python os.path.expanduser in Windows
		"""
		
		aHomeDir = os.path.expanduser( '~' )
		if aHomeDir not in ( '~', '%USERPROFILE%' ):
			return aHomeDir
		if os.name == 'nt' and aHomeDir == '%USERPROFILE%':
			return os.environ['USERPROFILE']


	def __saveRecentFileList ( self ):
		"""
		in: nothing
		returns nothing
		"""
		# creates recentfiledir if does not exist
		aDirName = self.__getHomeDir() + os.sep + RECENTFILELIST_DIRNAME
		if not os.path.isdir( aDirName ):
			os.mkdir( aDirName )

		# creates file 
		aFileName = self.__getHomeDir() + os.sep + RECENTFILELIST_FILENAME
		aFileObject = open( aFileName, 'w' )
		for aLine in self.__theRecentFileList:
			# writes lines
			aFileObject.write( aLine + '\n' )

		# close file
		aFileObject.close()


	def __convertPropertyValueList( self, aValueList ):
		
		aList = list()

		for aValueListNode in aValueList:

			if type( aValueListNode ) in (type([]), type( () ) ):

				isList = False
				for aSubNode in aValueListNode:
					if type(aSubNode) in (type([]), type( () ) ):
						isList = True
						break
				if isList:
					aConvertedList = self.__convertPropertyValueList( aValueListNode )
				else:
					aConvertedList = map(str, aValueListNode)

				aList.append( aConvertedList )
			else:
				aList.append( str (aValueListNode) )

		return aList




	def __getUniqueName( self, aType, aStringList ):
		"""
		in: string aType, list of string aStringList
		returns: string newName		
		"""
		i = 1
		while True:
			# tries to create 'aType'1
			newName = 'new' + aType + str(i)
			if not aStringList.__contains__( newName ):
				return newName
			# if it already exists in aStringList, creates aType2, etc..
			i += 1


	def __plainMessage( self, theMessage ):
		"""
		in: string theMessage
		returns None
		"""

		print theMessage



	def __loadStepper( self, anEml ):
		"""stepper loader"""

		aStepperList = anEml.getStepperList()


		for aStepper in aStepperList:

			aClassName = anEml.getStepperClass( aStepper )
			self.theModelStore.createStepper( str( aClassName ),\
				str( aStepper ) )

			aPropertyList = anEml.getStepperPropertyList( aStepper )
			
			for aProperty in aPropertyList:

				aValue = anEml.getStepperProperty( aStepper, aProperty )
				self.theModelStore.loadStepperProperty( aStepper,\
						   aProperty,\
						   aValue )

											 
	def __loadEntity( self, anEml, aSystemPath='/' ):

		aVariableList = anEml.getEntityList( DM_VARIABLE_CLASS, aSystemPath )
		aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
		aSubSystemList = anEml.getEntityList( 'System',	aSystemPath )

		self.__loadEntityList( anEml, 'Variable', aSystemPath, aVariableList )
		self.__loadEntityList( anEml, 'Process',  aSystemPath, aProcessList )
		self.__loadEntityList( anEml, 'System',   aSystemPath, aSubSystemList )

		for aSystem in aSubSystemList:
			aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
			self.__loadEntity( anEml, aSubSystemPath )


	def __loadProperty( self, anEml, aSystemPath='' ):
		# the default of aSystemPath is empty because
		# unlike __loadEntity() this starts with the root system

		aVariableList  = anEml.getEntityList( 'Variable',  aSystemPath )
		aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
		aSubSystemList = anEml.getEntityList( 'System',	aSystemPath )

		self.__loadPropertyList( anEml, 'Variable',\
						 aSystemPath, aVariableList )
		self.__loadPropertyList( anEml, 'Process',  aSystemPath, aProcessList )
		self.__loadPropertyList( anEml, 'System',\
						 aSystemPath, aSubSystemList )

		for aSystem in aSubSystemList:
			aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
			self.__loadProperty( anEml, aSubSystemPath )

	def __loadPropertyList( self, anEml, anEntityTypeString,\
						aSystemPath, anIDList ):

		for anID in anIDList:

			aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID

			aPropertyList = anEml.getEntityPropertyList( aFullID )

			aFullPNList = map( lambda x: aFullID + ':' + x, aPropertyList ) 
			aValueList = map( anEml.getEntityProperty, aFullPNList )
			map( self.theModelStore.loadEntityProperty,
				 aFullPNList, aValueList )

#####################################################################################################################	
	def __loadEntityList( self, anEml, anEntityTypeString,\
						  aSystemPath, anIDList ):
		
		aPrefix = anEntityTypeString + ':' + aSystemPath + ':'
		aMessage=''

		aFullIDList = map( lambda x: aPrefix + x, anIDList )
		aClassNameList = map( anEml.getEntityClass, aFullIDList )
		map( self.theModelStore.createEntity, aClassNameList, aFullIDList )
		"""
		if len(self.theModelStore.getNotExistClass())>0:
			aMessage='There is no .desc file for '
			for aClass in self.theModelStore.getNotExistClass():
				aMessage=aMessage + aClass + ', '
			self.printMessage( aMessage[:-1], ME_PLAINMESSAGE )
			self.theModelStore.setNotExistClass()
		"""


	def __saveStepper( self , anEml ):
		"""stepper loader"""

		aStepperList = self.theModelStore.getStepperList()

		for aStepper in aStepperList:

			aClassName = self.theModelStore.getStepperClassName( aStepper )
			anEml.createStepper( str( aClassName ), str( aStepper ) )

			aPropertyList = self.theModelStore.getStepperPropertyList( aStepper )

			for aProperty in aPropertyList:
				
				anAttributeList = self.theModelStore.getStepperPropertyAttributes( aStepper, aProperty)

				# check get attribute 
				if anAttributeList[ME_SAVEABLE_FLAG]:
									
					aValue = self.theModelStore.saveStepperProperty( aStepper, aProperty )
					if aValue == '' or aValue == []:
						continue

					aValueList = list()
					if type( aValue ) != type([]):
						aValueList.append( str(aValue) )
					else:
						aValueList = self.__convertPropertyValueList( aValue )

					anEml.setStepperProperty( aStepper, aProperty, aValueList )
	
	def __saveEntity( self, anEml, aSystemPath='/' ):

		aVariableList = self.theModelStore.getEntityList(  'Variable', aSystemPath )
		aProcessList   = self.theModelStore.getEntityList( 'Process', aSystemPath )
		aSubSystemList = self.theModelStore.getEntityList( 'System', aSystemPath )
		
		self.__saveEntityList( anEml, 'System',   aSystemPath, aSubSystemList )
		self.__saveEntityList( anEml, 'Variable', aSystemPath, aVariableList )
		self.__saveEntityList( anEml, 'Process',  aSystemPath, aProcessList )

		for aSystem in aSubSystemList:
			aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
			self.__saveEntity( anEml, aSubSystemPath )

			
	def __saveEntityList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):

	   for anID in anIDList:
		   
			aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
			aClassName = self.theModelStore.getEntityClassName( aFullID )

			if aClassName == 'System::/':
				pass
			else:
				anEml.createEntity( aClassName, aFullID )
			


	def __saveProperty( self, anEml, aSystemPath='' ):
		# the default of aSystemPath is empty because
		# unlike __loadEntity() this starts with the root system

		aVariableList  = self.theModelStore.getEntityList( 'Variable',\
														  aSystemPath )
		aProcessList   = self.theModelStore.getEntityList( 'Process',\
														  aSystemPath )
		aSubSystemList = self.theModelStore.getEntityList( 'System',\
														  aSystemPath )

		self.__savePropertyList( anEml, 'Variable', aSystemPath, aVariableList )
		self.__savePropertyList( anEml, 'Process', aSystemPath, aProcessList )
		self.__savePropertyList( anEml, 'System', aSystemPath, aSubSystemList )

		for aSystem in aSubSystemList:
			aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
			self.__saveProperty( anEml, aSubSystemPath )



	def __savePropertyList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):

		for anID in anIDList:

			aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
			aPropertyList = self.theModelStore.getEntityPropertyList( aFullID )

			for aProperty in aPropertyList:
				aFullPN = aFullID + ':' + aProperty
				
				anAttributeList = self.theModelStore.getEntityPropertyAttributes(aFullPN)

				# check savable
				if anAttributeList[ME_SAVEABLE_FLAG] :
					
					aValue = self.theModelStore.saveEntityProperty(aFullPN)

					if aValue != '' and aValue != []:

						aValueList = list()
						if type( aValue ) != type([]):
							aValueList.append( str(aValue) )

						else:
							# ValueList convert into string for eml
							aValueList = self.__convertPropertyValueList( aValue )

							
						anEml.setEntityProperty( aFullID, aProperty, aValueList )
