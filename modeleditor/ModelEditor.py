
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

from MainWindow import *
from StepperWindow import *
from EntityListWindow import *
#from PathwayEditor import *
from DMInfo import *
from PopupMenu import *
from PathwayEditor import *

from ecell.eml import *


RECENTFILELIST_FILENAME = '~/.modeleditor/.recentlist'
RECENTFILELIST_DIRNAME = '~/.modeleditor'
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
		self.theFullIDBrowser = None
		self.thePopupMenu = PopupMenu( self )
		self.theMainWindow = MainWindow( self )

		self.changesSaved = True

		# create untitled model
		self.__createModel()
		self.theLastComponent = None

		# set up windows
		self.theMainWindow.openWindow()
		
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
	

	def validateModel( ):
		printMessage("Sorry, not implemented!", ME_ERROR )


	def loadModel ( self, aFileName ):
		"""
		in: nothing
		returns nothing
		"""

		# check if it is dir
		if os.path.isdir( aFileName ):
			self.loadDirName = aFileName.rstrip('/')
			return

		# check if it is eml file
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
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
					sys.exc_traceback), '\n' )
			self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
			return

		self.__closeWindows()

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
		aBuffer = string.join( string.split(aString, '<eml>\n'), aCurrentInfo)
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
		
		# log to recent list
		self.__addToRecentFileList ( aFileName )
		self.theModelFileName = aFileName
		self.printMessage("Model %s saved successfully."%aFileName )
		self.theModelName = os.path.split( aFileName )[1]
		self.modelHasName = True
		self.changesSaved = True
		self.updateWindows()


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
		in: string anID
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
		for aCommand in aCommandList:
			# execute commands
			aCommand.execute()

			( aType, anID ) = aCommand.getAffectedObject()
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
		for aCommand in aCommandList:
			# execute commands
			aCommand.execute()
			( aType, anID ) = aCommand.getAffectedObject()
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


	def createPathwayEditor( self ):
		newWindow = PathwayEditor( self )
		newWindow.openWindow()
		self.thePathwayEditorList.append( newWindow )

		pass
#		newWindow = PathwayEditor( self, args )
#		newWindow.openWindow()
#		self.thePathwayEditorList.append( newWindow )


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

		for aPathwayEditor in self.thePathwayEditorList:
			aPathwayEditor.update( aType, anID )
		if self.theFullIDBrowser != None:
			self.theFullIDBrowser.update( aType,anID )
		if self.theMainWindow.exists():
			self.theMainWindow.update()
		#self.theLayoutManager.update( aType, anID )


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

		# set name 
		self.theModelName = ''
		self.theUndoQueue = None
		self.theRedoQueue = None
		self.changesSaved = True
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



	def __createModel (self ):
		""" 
		in: nothing
		out nothing
		"""

		self.__closeModel()
		# create new model
		self.theModelStore = ModelStore()
		self.theUndoQueue = CommandQueue(MAX_REDOABLE_COMMAND)
		self.theRedoQueue = CommandQueue(MAX_REDOABLE_COMMAND)
		
		# initDM

		# set name
		self.theModelName = 'Untitled'
		self.modelHasName = False
		self.changesSaved = True




	def __loadRecentFileList ( self ):
		"""
		in: nothing
		returns nothing
		"""
		self.__theRecentFileList = []
		aFileName = os.path.expanduser( RECENTFILELIST_FILENAME )

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



	def __saveRecentFileList ( self ):
		"""
		in: nothing
		returns nothing
		"""
		# creates recentfiledir if does not exist
		aDirName = os.path.expanduser( RECENTFILELIST_DIRNAME )
		if not os.path.isdir( aDirName ):
			os.mkdir( aDirName )

		# creates file 
		aFileName = os.path.expanduser( RECENTFILELIST_FILENAME )
		aFileObject = open( aFileName, 'w' )

		for aLine in self.__theRecentFileList:
			# writes lines
			aFileObject.write( aLine + '\n' )

		# close file
		aFileObject.close()


	def __convertPropertyValueList( self, aValueList ):
		
		aList = list()

		for aValueListNode in aValueList:

			if type( aValueListNode ) == type([]):

				if type( aValueListNode[0] ) == type([]):
					aConvertedList = self.__convertPropertyValueList( aValueListNode )
				else:
					aConvertedList = map(str, aValueListNode)

				aList.append( aConvertedList )

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

			
	def __loadEntityList( self, anEml, anEntityTypeString,\
						  aSystemPath, anIDList ):
		
		aPrefix = anEntityTypeString + ':' + aSystemPath + ':'

		aFullIDList = map( lambda x: aPrefix + x, anIDList )
		aClassNameList = map( anEml.getEntityClass, aFullIDList )
		map( self.theModelStore.createEntity, aClassNameList, aFullIDList )




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
				if anAttributeList[ME_SETTABLE_FLAG] and anAttributeList[ME_GETTABLE_FLAG]:
									
					aValue = self.theModelStore.saveStepperProperty( aStepper, aProperty )
					if aValue == '':
						continue

					aValueList = list()
					if type( aValue ) != type([]):
						aValueList.append( (aValue) )
					else:
						aValueList = aValue

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
				if anAttributeList[ME_SETTABLE_FLAG] and anAttributeList[ME_GETTABLE_FLAG]:
					
					aValue = self.theModelStore.saveEntityProperty(aFullPN)

					if aValue != '':

						aValueList = list()
						if type( aValue ) != type([]):
							aValueList.append( (aValue) )
						else:
							# ValueList convert into string for eml
							aValueList = self.__convertPropertyValueList( aValue )

							
						anEml.setEntityProperty( aFullID, aProperty, aValueList )
