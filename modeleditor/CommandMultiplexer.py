from Constants import *
from LayoutCommand import *
from EntityCommand import *
from Utils import *


class CommandMultiplexer:

	def __init__( self, aModelEditor, aLayoutManager ):
		self.theLayoutManager = aLayoutManager
		self.theModelEditor = aModelEditor


	def processCommandList( self, aCommandList ):
		returnCmdList = []

		for aCmd in aCommandList:
			if aCmd.doMultiplex:
				returnCmdList.extend(self.processCommand( aCmd ))
		return returnCmdList


	def processCommand( self, aCmd ):
		cmdType = aCmd.__class__.__name__
		returnCmdList = []
		anIterator = self.theLayoutManager.createObjectIterator()
		returnCmdList.append( aCmd )
		if cmdType == "DeleteEntityList":
			fullIDList = aCmd.theArgs[ aCmd.IDLIST ]
			returnCmdList = []
			for aFullID in fullIDList:
				returnCmdList .extend( self.__deleteObjectsByFullID( aFullID) )
			returnCmdList.append( aCmd )				

		elif cmdType == "RenameEntity":
			oldFullID = aCmd.theArgs[ aCmd.OLDID ]
			newFullID = aCmd.theArgs[ aCmd.NEWID ]
			anIterator.filterByFullID( oldFullID )
			while True:
				anObject = anIterator.getNextObject()
				if anObject == None:
					break
				renameCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_FULLID, newFullID )
				returnCmdList.append( renameCommand )

		elif cmdType == "CutEntityList":
			fullIDList = aCmd.theArgs[ aCmd.IDLIST ]
			for aFullID in fullIDList:
				returnCmdList.extend( self.__deleteObjectsByFullID( aFullID) )
				
			# elif cmdType == "DeleteStepperList":
			#	stepperIDList = aCmd.theArgs[ aCmd.IDLIST ]
			#	for aStepperID in stepperIDList:
			#		anIterator.deleteFilters():
			#		anIterator.filterByProperty( OB_STEPPERID, aStepperID )
			#		while True:
			#			anObject = anIterator.getNextObject()
			#			if anObject == None:
			#				break
			#			changeCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, '' )
			#			returnCmdList.append( changeCommand )

		elif cmdType == "RenameStepper":
			oldID = aCmd.theArgs[ aCmd.OLDID ]
			newID = aCmd.theArgs[ aCmd.NEWID ]
			anIterator.filterByProperty( OB_STEPPERID, oldID )
			while True:
				anObject = anIterator.getNextObject()
				if anObject == None:
					break
				changeCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, newID )
				returnCmdList.append( changeCommand )

		elif cmdType == "ChangeEntityProperty":
			chgdFullPN = aCmd.theArgs[ aCmd.FULLPN ]
			newValue = aCmd.theArgs[ aCmd.VALUE ]
			chgdFullID = getFullID( chgdFullPN )
			chgdProperty = getPropertyName( chgdFullPN )
			if chgdProperty in [ MS_PROCESS_STEPPERID, MS_SYSTEM_STEPPERID ]:
				returnCmdList.extend( self.__changeStepperID(chgdFullID, newValue ) )
			elif chgdProperty == MS_PROCESS_VARREFLIST:
				returnCmdList.extend( self.__changeVariableReferenceList( chgdFullID, newValue ) )
		elif cmdType == "PasteEntityPropertyList":
			# get buffer
			chgdFullID = aCmd.theArgs[ aCmd.FULLID ]
			aBuffer = aCmd.theArgs[ aCmd.BUFFER ]
			propertyList = aBuffer.getPropertyList()
			for aProperty in propertyList:
				if aProperty in [ MS_PROCESS_STEPPERID, MS_SYSTEM_STEPPERID ]:
					newValue = aBuffer.getProperty( aProperty )
					returnCmdList.extend( self.__changeStepperID(chgdFullID, newValue ) )
				elif aProperty == MS_PROCESS_VARREFLIST:
					newValue = aBuffer.getProperty( aProperty )
					returnCmdList.extend( self.__changeVariableReferenceList( chgdFullID, newValue ) )
		elif cmdType == "RelocateEntity":
			targetSystemID = aCmd.theArgs[ aCmd.TARGETSYSTEM ]
			aFullID = aCmd.theArgs[ aCmd.ENTITYID ]
			anIterator.filterByFullID( aFullID )
			while True:
				anObject = anIterator.getNextObject()
				if anObject == None:
					break
				thePackingStrategy = anObject.getLayout().thePackingStrategy
				relocateCommand = thePackingStrategy.autoMoveObject( targetSystemID, anObject.getID() )
				returnCmdList.extend( relocateCommand )
		elif cmdType == "CreateObject":
			# if type is not system or process or variable pass
			objectType = aCmd.theArgs[ aCmd.TYPE ]
			objectFullID = aCmd.theArgs[ aCmd.FULLID ]
			if objectType in [ME_VARIABLE_TYPE, ME_PROCESS_TYPE, ME_SYSTEM_TYPE ]:
				if not self.theModelEditor.getModel().isEntityExist( objectFullID ):
					if objectType == ME_PROCESS_TYPE:
						newClass = self.theModelEditor.getDefaultProcessClass()
					elif objectType == ME_VARIABLE_TYPE:
						newClass = DM_VARIABLE_CLASS
					elif objectType == ME_SYSTEM_TYPE:
						newClass = DM_SYSTEM_CLASS

					createCmd = CreateEntity( self.theModelEditor, objectFullID, newClass )
					returnCmdList.append( createCmd )
		elif cmdType == "SetObjectProperty":
			chgdProperty = aCmd.theArgs[ aCmd.PROPERTYNAME ]
			chgdValue = aCmd.theArgs[ aCmd.NEWVALUE ]
			chgdID = aCmd.theArgs[ aCmd.OBJECTID ]
			theLayout = aCmd.theReceiver
			theObject = theLayout.getObject ( chgdID )
			objectType = theObject.getProperty( OB_TYPE )
			if chgdProperty in GLOBALPROPERTYSET:
				if objectType == OB_TYPE_CONNECTION:
					processObject = theObject.getProperty( CO_PROCESS_ATTACHED )
					processFullID = processObject.getProperty( OB_FULLID )
					aVarrefName =  theObject.getProperty( CO_NAME )
					objectIter = self.theLayoutManager.createObjectIterator()
					objectIter.filterByFullID( aProcessFullID )
					while True:
						anObject = objectIter.getNextObject()
						connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
						for conObject in connectionList:
							if conObject.getProperty( CO_NAME ) == aVarrefName and conObject.getID() != chgdID:
								chgCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), chgdProperty, chgdValue )
								cmdList.append( chgCommand )
				else:
					objectFullID = theObject.getProperty( OB_FULLID )
					objectIter = self.theLayoutManager.createObjectIterator()
					objectIter.filterByFullID( aProcessFullID )
					while True:
						anObject = objectIter.getNextObject()
						if anObject.getID() != chgdID:
							chgCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), chgdProperty, chgdValue )
							cmdList.append( chgCommand )
			# here comes the shaky part
			# Commands that require Issuing modelcommands ( like renaming connections, changing coefficients)
			if chgdProperty == CO_COEF:
				processObject = theObject.getProperty( CO_PROCESS_ATTACHED )
				processFullID = processObject.getProperty( OB_FULLID )
				aVarrefName =  theObject.getProperty( CO_NAME )
				aFullPN = createFullPN( processFullID, aVarrefName )
				oldValue = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
				for aVarref in oldValue:
					if aVarref[ MS_VARREF_NAME ] == aVarrefName:
						aVarref[ MS_VARREF_COEF ] = chgdValue
						break
				chgCmd = ChangeEntityProperty( aFullPN, oldValue )
				cmdList.append(chgCmd)
			elif chgdProperty == CO_NAME:
				processObject = theObject.getProperty( CO_PROCESS_ATTACHED )
				processFullID = processObject.getProperty( OB_FULLID )
				aVarrefName =  theObject.getProperty( CO_NAME )
				aFullPN = createFullPN( processFullID, aVarrefName )
				oldValue = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
				for aVarref in oldValue:
					if aVarref[ MS_VARREF_NAME ] == aVarrefName:
						aVarref[ MS_VARREF_NAME ] = chgdValue
						break
				chgCmd = ChangeEntityProperty( aFullPN, oldValue )
				cmdList.append(chgCmd)

		elif cmdType == "MoveObject":
			# decide whether relocate or move
			aLayout = aCmd.theReceiver
			objectID = aCmd.theArgs[ aCmd.OBJECTID ]
			newX = aCmd.theArgs[ aCmd.NEWX ]
			newY = aCmd.theArgs[ aCmd.NEWY ]
			newParent = aCmd.theArgs[ aCmd.NEWPARENT ]
			if newParent != None:
				anObject = aLayout.getObject( objectID )
				oldParent = anObject.getParent()
				if oldParent != newParent:
					# issue relocate command
					targetSystemID = newParent.getProperty( OB_FULLID )
					aFullID = anObject.getProperty( OB_FULLID )
					relCmd = RelocateEntity( self.theModelEditor, aFullID, targetSystemID )
					returnCmdList.insert( len( returnCmdList)-1, relCmd )
					anIterator.filterByFullID( aFullID )
					while True:
						anObject = anIterator.getNextObject()
						if anObject == None:
							break
						if anObject.getID() == objectID:
							continue
						thePackingStrategy = anObject.getLayout().thePackingStrategy
						relocateCommand = thePackingStrategy.autoMoveObject( targetSystemID, anObject.getID() )
						returnCmdList.extend( relocateCommand )
#		elif cmdType == "DeleteObject":
#			aLayout = aCmd.theReceiver
#			objectID = aCmd.theArgs[ aCmd.OBJECTID ]
#			anObject = aLayout.getObject( objectID )
#			returnCmdList.extend( self.__deleteObjectByID( anObject ) )
		elif cmdType == "PasteObject":
			# construct real fullid
			aBuffer = aCmd.theBuffer
			aParent = aCmd.theParent
			objectType = aBuffer.getProperty ( OB_TYPE )
			if objectType in [ OB_TYPE_PROCESS, OB_TYPE_VARIABLE, OB_TYPE_SYSTEM]:
				objectFullID = aBuffer.getProperty( OB_FULLID )
				parentFullID = aParent.getProperty( OB_FULLID )
				# modify buffer fullid
				# create it from parent 
				parentSysPath = convertSysIDToSysPath( parentFullID )
				fullIDList = objectFullID.split(':')
				fullIDList[1] = parentSysPath
				newFullID = ':'.join( fullIDList )
				# if fullid exists do nothing
				# else create new fullid
				if not self.theModelEditor.getModel().isEntityExist( newFullID ):
					entityBuffer = aBuffer.getEntityBuffer()
					pasteCmd = PasteEntity( self.theModelEditor, entityBuffer, parentSysPath )
					returnCmdList.insert( len( returnCmdList) - 1, pasteCmd )

		elif cmdType == "CreateConnection":
			# get processfullid
			aProcessObject = aCmd.theReceiver.getObject( aCmd.processObjectID )
			processFullID = aProcessObject.getProperty( OB_FULLID )
			aVarrefName = aCmd.varrefName
			aFullPN = createFullPN ( processFullID, MS_PROCESS_VARREFLIST )
			# create command for modifying variablereferencelist if not exist
			aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
			foundFlag = False
			for aVarref in aVarrefList:
				if aVarref[MS_VARREF_NAME] == aVarrefName:
					foundFlag = True
					break
			if not foundFlag:
				variableObject = aCmd.theReceiver.getObject( aCmd.variableObjectID )
				variableFullID = variableObject.getProperty( OB_FULLID )
				variableFullID = variableFullID.lstrip( variableFullID, ME_VARIABLE_TYPE )
				aCoef = 1
				if aCmd.direction == VARIABLE_TO_PROCESS:
					aCoef = -1
				newVarref = [ aVarrefName, variableFullID, aCoef ]
				aVarrefList.append ( newVarref )
				addCmd = ChangeEntityProperty( self.theModelEditor, aFullPN, aVarrefList )
				returnCmdList.insert( len(returnCmdList) -1, addCmd )
			
		elif cmdType == "RedirectConnection":
			# get changed and new endpoint
			connectionObject = aCmd.theReceiver.getObject( aCmd.objectID )
			aVarrefName = connectionObject.getProperty( CO_NAME )
			oldProcessObject = connectionObject.getProperty( CO_PROCESS_ATTACHED )
			oldProcessFullID = oldProcessObject.getProperty( OB_FULLID )
			oldVariableObject = connectionObject.getProperty( CO_VARIABLE_ATTACHED )
			oldVariableFullID = oldVariableObject.getProperty( OB_FULLID )
			newVarrefName = aCmd.newVarrefName
			if newVarrefName == None:
				newVarrefName = aVarrefName
			# if processID is changed, delete from varreflist, insert into new varreflist
			if aCmd.newProcessObjectID != None:
				# first delete
				newProcessFullID = aCmd.newProcessObject.getProperty( OB_FULLID )

				oldVarrefFullPN = createFullPN ( oldProcessFullID, MS_PROCESS_VARREFLIST )
				aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( oldVarrefFullPN ) )

				for aVarref in aVarrefList:
					if aVarref[ MS_VARREF_NAME ] == aVarrefName:
						foundVarref = aVarref
						break
				theCoef = foundVarref[MS_VARREF_COEF]
				isRelative = isRelativeReference( foundVarref[ MS_VARREF_FULLID ] )
				aVarrefList.remove( foundVarref )
				delCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
				returnCmdList.insert( len(returnCmdList) -1, delCmd )

				# then insert
				newVarrefFullPN = createFullPN ( newProcessFullID, MS_PROCESS_VARREFLIST )
				aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( newVarrefFullPN ) )
				if isRelative:
					newVariableFullID = getRelativeReference( newProcessFullID, oldVariableFullID )
				newVariableFullID = newVariableFullID.lstrip( ME_VARIABLE_TYPE )
				newVarref = [ newVarrefName, newVariableFullID, theCoef ]
				aVarrefList.add( newVarref )
				addCmd = ChangeEntityProperty( self.theModelEditor, newVarrefFullPN, aVarrefList )
				returnCmdList.insert( len(returnCmdList) -1, addCmd )

			# if varrefID is changed, change varreflist
			else:
				newVariableObject = aCmd.theReceiver.getObject(newVariableObjectID)
				newVariableFullID = newVariableObject.getProperty( OB_FULLID )
				oldVarrefFullPN = createFullPN ( oldProcessFullID, MS_PROCESS_VARREFLIST )
				aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( oldVarrefFullPN ) )

				for aVarref in aVarrefList:
					if aVarref[ MS_VARREF_NAME ] == aVarrefName:
						foundVarref = aVarref
						break
				isRelative = isRelativeReference( foundVarref[ MS_VARREF_FULLID ] )
				if isRelative:
					newVariableFullID = getRelativeReference( oldProcessFullID, newVariableFullID )
				newVariableFullID = newVariableFullID.lstrip( ME_VARIABLE_TYPE )
				aVarref[MS_VARREF_FULLID] = newVariableFullID
				chgCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
				returnCmdList.insert( len(returnCmdList) - 1, chgCmd )
				
			# go through all objectID with same FullID on different layouts
			# check if andpoint is present and create commands with autoconnect
			objectIter = self.theLayoutManager.createObjectIterator()
			objectIter.filterByFullID(oldProcessFullID )
			while True:
				anObject = objectIter.getNextObject()
				if anObject == None:
					break
				connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
				for conObject in connectionList:
					if conObject.getProperty( CO_NAME ) == aVarrefName:
						if conObject.getID()== aCmd.objectID:
							continue
						newProcessObjectID = None
						newVariableObjectID = None
						aLayout = conObject.getLayout()
						if aCmd.newProcessObjectID != None:
							processObjectList = aLayout.getObjectList( OB_TYPE_PROCESS )
							for aProcessID in processObjectList:
								aProcessObject = aLayout.getObject( aProcessID )
								if aProcessObject.getProperty( OB_FULLID ):
									newProcessObjectID = aProcessObject.getID()
							if newProcessObjectID == None:
								continue
							newVariableObjectID = conObject.getProperty( CO_VARIABLE_ATTACHED ).getID()
						else:
							variableObjectList = aLayout.getObjectList( OB_TYPE_VARIABLE )
							for aVariableID in variableObjectList:
								aVariableObject = aLayout.getObject( aVariableID )
								if aVariableObject.getProperty( OB_FULLID ):
									newVariableObjectID = aVariableObject.getID()
							if  newVariableObjectID == None:
								continue
							newProcessObjectID = conObject.getProperty( CO_PROCESS_ATTACHED ).getID()

						redirCmd = RedirectConnection( aLayout, conObject.getID(), newProcessObjectID, newVariableObjectID, None, None, None ) 
						returnCmdList.append( redirCmd )


		for aCommand in returnCmdList:
			aCommand.doNotMultiplex()
			aCommand.doNotMultiplexReverse()
		return returnCmdList


	def __deleteObjectsByFullID( self, aFullID ):
		cmdList = []
		anIterator = self.theLayoutManager.createObjectIterator()
		anIterator.filterByFullID( aFullID )

		while True:
			anObject = anIterator.getNextObject()
			if anObject == None:
				break
			cmdList.extend( self.__deleteObjectByID( anObject) )
		return cmdList
	
	
	def __deleteObjectByID( self, anObject):
		# issue command for deleting object, 
		cmdList =[]
		deleteCommand = DeleteObject( anObject.getLayout(), anObject.getID() )
		cmdList.append( deleteCommand )
		# issue commands for deleting connections

#		objectType = anObject.getProperty( OB_TYPE )
#		if objectType == ME_VARIABLE_TYPE:
#			IDProperty = VR_CONNECTIONLIST
#		elif objectType == ME_PROCESS_TYPE:
#			IDProperty = PR_CONNECTIONLIST
#		else:
#			return cmdList
#		connectionList = anObject.getProperty( IDProperty )
#
#		for aConnection in connectionList:
#			deleteCommand = DeleteObject( aConnection.getLayout(), aConnection.getID() )
#			cmdList.append( deleteCommand )
		return cmdList



	def __changeStepperID( self, aFullID, newStepperID ):
		cmdList = []
		anIterator = self.theLayoutManager.createObjectIterator()
		anIterator.filterByFullID( aFullID )
		while True:
			anObject = anIterator.getNextObject()
			if anObject == None:
				break
			chgCmd = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, newStepperID)
			cmdList.append( chgCmd )
		return cmdList
		

		
	def __changeVariableReferenceList( self, aProcessFullID, newVarrefList ):
		cmdList = []
		# get old VariablereferenceList
		aFullPN = createFullPN( aProcessFullID, MS_PROCESS_VARREFLIST )
		oldVarrefList = self.theModelEditor.getModel().getEntityProperty( aFullPN )
		# find deleted
		oldVarrefNames = []
		for anOldVarref in oldVarrefList:
			foundInNew = False
			oldVarrefNames.append( anOldVarref [ ME_VARREF_NAME ] )
			for aNewVarref in newVarrefList:
				if aNewVarref[ ME_VARREF_NAME ] == anOldVarref [ ME_VARREF_NAME ]:
					foundInNew = True
					break
				if not foundInNew:
					cmdList.extend( self.__deleteConnections( aProcessFullID, anOldVarref [ MS_VARREF_NAME ] ) )
	
		# cycle through all new varrefs
		for aNewVarref in newVarrefList:
			# process insertions
			if aNewVarref[ MS_VARREF_NAME ] not in oldVarrefNames:	
				# do nothing
				continue
			else:
				for anOldVarref in oldVarrefList:
					if anOldVarref [ ME_VARREF_NAME ] == aNewVarref[ ME_VARREF_NAME ]:
						break
			if anOldVarref[ ME_VARREF_COEF ] != aNewVarref[ ME_VARREF_COEF ]:
				# process redirections
				oldVariable = getAbsoluteReference( aProcessFullID, anOldVarref[ ME_VARREF_FULLID ] )
				newVariable = getAbsoluteReference( aProcessFullID, aNewVarref[ ME_VARREF_FULLID ] )
 
				cmdList.extend( self.__redirectVarref( aProcessFullID, anOldVarref [ ME_VARREF_NAME ], oldVariable, newVariable ) )
			if anOldVarref[ ME_VARREF_FULLID ] != aNewVarref[ ME_VARREF_FULLID ]:
				# process coef changes
				cmdList.extend( self.__changeCoef( aProcessFullID, anOldVarref [ ME_VARREF_NAME ], anOldVarref[ ME_VARREF_COEF ] ) )

		return cmdList

		
	def __deleteConnections( self, aProcessID, aVarrefName ):
		cmdList = []
		objectIter = self.theLayoutManager.createObjectIterator()
		objectIter.filterByFullID(aProcessFullID )
		while True:
			anObject = objectIter.getNextObject()
			connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
			for conObject in connectionList:
				if conObject.getProperty( CO_NAME ) == aVarrefName:
					deleteCommand = DeleteObject( conObject.getLayout(), conObject.getID() )
					cmdList.append( deleteCommand )
		return cmdList


	def __redirectVarref( self, aProcessFullID, aVarrefName, oldVariable, newVariable ):
		cmdList = []
		objectIter = self.theLayoutManager.createObjectIterator()
		objectIter.filterByFullID( aProcessFullID )
		while True:
			anObject = objectIter.getNextObject()
			if anObject == None:
				break
			connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
			for conObject in connectionList:
				if conObject.getProperty( CO_NAME ) == aVarrefName:
					deleteCommand = DeleteObject( conObject.getLayout(), conObject.getID() )
					cmdList.append( deleteCommand )

		return cmdList


	def __changeCoef( self, aProcessFullID, aVarrefName, newCoef ):
		cmdList = []
		objectIter = self.theLayoutManager.createObjectIterator()
		objectIter.filterByFullID( aProcessFullID )
		while True:
			anObject = objectIter.getNextObject()
			if anObject == None:
				break

			connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
			for conObject in connectionList:
				if conObject.getProperty( CO_NAME ) == aVarrefName:
					deleteCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), CO_COEF, newCoef )
					cmdList.append( deleteCommand )

		return cmdList

