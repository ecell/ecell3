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
				returnCmdList.extend( self.__deleteObjectsByFullID( aFullID) )
			returnCmdList.append( aCmd )


		elif cmdType == "RenameEntity":
			oldFullID = aCmd.theArgs[ aCmd.OLDID ]
			newFullID = aCmd.theArgs[ aCmd.NEWID ]
			anIterator.filterByFullID( oldFullID )
			aType = getFullIDType( newFullID )
			while True:
				anObject = anIterator.getNextObject()
				if anObject == None:
					break
				renameCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_FULLID, newFullID )
				returnCmdList.append( renameCommand )
				if aType == ME_SYSTEM_TYPE:
					returnCmdList.extend( self.__changeFullID( anObject, newFullID ) )
					#check broken connections!
					returnCmdList.extend( self.__checkBrokenConnections( oldFullID, newFullID ) )

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

		#elif cmdType == "RenameStepper":
		#	oldID = aCmd.theArgs[ aCmd.OLDID ]
		#	newID = aCmd.theArgs[ aCmd.NEWID ]
		#	anIterator.filterByProperty( OB_STEPPERID, oldID )
		#	while True:
		#		anObject = anIterator.getNextObject()
		#		if anObject == None:
		#			break
		#		changeCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, newID )
		#		returnCmdList.append( changeCommand )

		elif cmdType == "ChangeEntityProperty":
			chgdFullPN = aCmd.theArgs[ aCmd.FULLPN ]
			newValue = aCmd.theArgs[ aCmd.VALUE ]
			chgdFullID = getFullID( chgdFullPN )
			chgdProperty = getPropertyName( chgdFullPN )
			#if chgdProperty in [ MS_PROCESS_STEPPERID, MS_SYSTEM_STEPPERID ]:
			#	returnCmdList.extend( self.__changeStepperID(chgdFullID, newValue ) )
			if chgdProperty == MS_PROCESS_VARREFLIST:
				chgCom = returnCmdList.pop()
				returnCmdList.extend( self.__changeVariableReferenceList( chgdFullID, newValue ) )
				returnCmdList.append( chgCom )
		elif cmdType == "PasteEntityPropertyList":
			# get buffer
			chgdFullID = aCmd.theArgs[ aCmd.FULLID ]
			aBuffer = aCmd.theArgs[ aCmd.BUFFER ]
			propertyList = aBuffer.getPropertyList()
			for aProperty in propertyList:
				#if aProperty in [ MS_PROCESS_STEPPERID, MS_SYSTEM_STEPPERID ]:
				#	newValue = aBuffer.getProperty( aProperty )
				#	returnCmdList.extend( self.__changeStepperID(chgdFullID, newValue ) )
				if aProperty == MS_PROCESS_VARREFLIST:
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
					returnCmdList.insert(0, createCmd )
		elif cmdType == "SetObjectProperty":
			chgdProperty = aCmd.theArgs[ aCmd.PROPERTYNAME ]
			chgdValue = aCmd.theArgs[ aCmd.NEWVALUE ]
			chgdID = aCmd.theArgs[ aCmd.OBJECTID ]
			theLayout = aCmd.theReceiver
			theObject = theLayout.getObject ( chgdID )
			objectType = theObject.getProperty( OB_TYPE )
			if chgdProperty in GLOBALPROPERTYSET:
				if objectType == OB_TYPE_CONNECTION:
					aProcessID = theObject.getProperty( CO_PROCESS_ATTACHED )
					processObject = theLayout.getObject( aProcessID )
					aProcessFullID = processObject.getProperty( OB_FULLID )
					aVarrefName =  theObject.getProperty( CO_NAME )
					objectIter = self.theLayoutManager.createObjectIterator()
					objectIter.filterByFullID( aProcessFullID )
					while True:
						anObject = objectIter.getNextObject()
						if anObject == None:
							break
						connectionList = anObject.getProperty( PR_CONNECTIONLIST )
						for aConID in connectionList:
							conObject = theLayout.getObject( aConID )
							if conObject.getProperty( CO_NAME ) == aVarrefName and aConID!= chgdID:
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
				cmdList.insert(0,chgCmd)
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
				cmdList.insert(0,chgCmd)

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
			foundFlag = False
			# get processfullid
			processObjectID = aCmd.theArgs[ aCmd.PROCESSOBJECTID ]
			variableObjectID = aCmd.theArgs[ aCmd.VARIABLEOBJECTID ]
			cmdVarrefName = aCmd.theArgs[ aCmd.VARREFNAME ]
			cmdDirection = aCmd.theArgs[ aCmd.DIRECTION ]
			validFlag = True
			aProcessObject = aCmd.theReceiver.getObject( processObjectID )
			processFullID = aProcessObject.getProperty( OB_FULLID )
			aVarrefName = cmdVarrefName
			aFullPN = createFullPN ( processFullID, MS_PROCESS_VARREFLIST )
			# create command for modifying variablereferencelist if not exist
			aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
			for aVarref in aVarrefList:
				if aVarref[MS_VARREF_NAME] == aVarrefName:
					foundFlag = True
					break
			if not foundFlag:
				if  type( variableObjectID ) == type ( [] ):
					variableFullID = ":.:___NOTHING"
				else:
					processObject =  aCmd.theReceiver.getObject( processObjectID )
					processFullID = processObject.getProperty( OB_FULLID )
					variableObject = aCmd.theReceiver.getObject( variableObjectID )
					variableFullID = variableObject.getProperty( OB_FULLID )

					variableFullID = getRelativeReference (processFullID, variableFullID)


					
				aCoef = 1
				if cmdDirection == VARIABLE_TO_PROCESS:
					aCoef = -1
				newVarref = [ aVarrefName, variableFullID, aCoef ]
				aVarrefList.append ( newVarref )
				addCmd = ChangeEntityProperty( self.theModelEditor, aFullPN, aVarrefList )
				returnCmdList.insert( len(returnCmdList) -1, addCmd )
			
		elif cmdType == "RedirectConnection":
			# get changed and new endpoint
			cmdObjectID = aCmd.theArgs[ aCmd.OBJECTID ]
			cmdNewProcessObjectID = aCmd.theArgs[aCmd.NEWPROCESSOBJECTID ]
			cmdNewVariableObjectID = aCmd.theArgs[aCmd.NEWVARIABLEOBJECTID ]
			connectionObject = aCmd.theReceiver.getObject( cmdObjectID )
			aVarrefName = connectionObject.getProperty( CO_NAME )
			oldProcessFullID = connectionObject.getProperty( CO_PROCESS_ATTACHED )
			if type( oldProcessFullID ) != type( [] ):
				oldProcessObject = aCmd.theReceiver.getObject( oldProcessFullID )
			else:
				oldProcessObject = oldProcessFullID
			oldVariableFullID = connectionObject.getProperty( CO_VARIABLE_ATTACHED )
			if type( oldVariableFullID ) != type( [] ):
				oldVariableObject = aCmd.theReceiver.getObject( oldVariableFullID )
			else:
				oldVariableObject = oldVariableFullID
			newVarrefName = aCmd.theArgs[ aCmd.NEWVARREFNAME ]
			if newVarrefName == None:
				newVarrefName = aVarrefName
			theCoef = connectionObject.getProperty( CO_COEF )
			isRelative = connectionObject.getProperty( CO_ISRELATIVE )

			oldVarrefFullPN = createFullPN ( oldProcessFullID, MS_PROCESS_VARREFLIST )
			aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( oldVarrefFullPN ) )
			for aVarref in aVarrefList:
				if aVarref[ MS_VARREF_NAME ] == aVarrefName:
					foundVarref = aVarref
					break

			# if processID is changed, it can only be changes in the rings so no changes are necessary
			if cmdNewProcessObjectID != None:
				pass

			# if varrefID is changed, handle four cases
			elif cmdNewVariableObjectID != None:
				doDetachment = type( oldVariableObject) == type( [] )
				doAttachment = type( cmdNewVariableObjectID) == type([])
				#no detachment, no attachment - do nothing
				if not doDetachment and not doAttachment:
					entityCmd = None
					objCommands = []

				# just detachment,  first change entity property, then detach other layouts
				elif doDetachment and not doAttachment:
					foundVarref[ MS_VARREF_FULLID ] = ""
					objCommands = []
					objectIter = self.theLayoutManager.createObjectIterator()
					objectIter.filterByFullID(aProcessFullID )
					while True:
						anObject = objectIter.getNextObject()
						if anObject == None:
							break
						connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
						for conObject in connectionList:
							if conObject.getProperty( CO_NAME ) == aVarrefName and conObject.getID() != cmdObjectID:
								attachmentPoints = copyValue(conObject.getProperty( CO_ENDPOINT2 ) )
								attachmentPoints[0] += 20
								attachmentPoints[1] += 20
								redirCommand = RedirectConnection( conObject.getLayout(), conObject.getID(), None, attachmentPoints, None, None, None )
								objCommands.append( redirCommand )
					

				# just attachment, first change entity varref, then attach other layouts
				elif doAttachment:
					#FIXME do Attachments with autoconnect
					objCommands = []
					aVariableObject = aCmd.theReceiver.getObject( cmdNewVariableFullID )
					foundVarref[ MS_VARREF_FULLID ] = aVariableObject.getProperty( OB_FULLID )
					entityCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
				if entityCmd != None:
					returnCmdList.insert( len(returnCmdList)-1, entityCmd )
				returnCmdList.extend( objCommands )

				

			elif cmdNewVarrefName != None:
				# do rename in entity list and other places
				foundVarref[ MS_VARREF_NAME ] = cmdNewVarrefName
				objCommands = []
				objectIter = self.theLayoutManager.createObjectIterator()
				objectIter.filterByFullID(aProcessFullID )
				while True:
					anObject = objectIter.getNextObject()
					if anObject == None:
						break
					connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
					for conObject in connectionList:
						if conObject.getProperty( CO_NAME ) == aVarrefName and conObject.getID() != cmdObjectID:
							redirCommand = RedirectConnection( conObject.getLayout(), conObject.getID(), None, None, None, None, cmdNewVarrefName )
							objCommands.append( redirCommand )
				entityCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
				returnCmdList.extend( objCommands )

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
		deleteVarrefList = []
		for anOldVarref in oldVarrefList:
			oldVarrefNames.append( anOldVarref [ ME_VARREF_NAME ] )
		for anOldVarref in oldVarrefList:
			foundInNew = False
			changedTo = None
			for aNewVarref in newVarrefList:
				if aNewVarref[ ME_VARREF_NAME ] == anOldVarref [ ME_VARREF_NAME ]:
					foundInNew = True
				elif aNewVarref[ ME_VARREF_COEF ] == anOldVarref[ ME_VARREF_COEF ]:
					if aNewVarref[ ME_VARREF_COEF ] == anOldVarref[ ME_VARREF_COEF ] and aNewVarref[ ME_VARREF_NAME ] not in oldVarrefNames:
						changedTo = aNewVarref[ ME_VARREF_NAME ]
			if not foundInNew:
				if changedTo != None:
					cmdList.extend( self.__changeVarrefName( aProcessFullID, anOldVarref [ MS_VARREF_NAME ], changedTo ) )
				else:
					cmdList.extend( self.__deleteConnections( aProcessFullID, anOldVarref [ MS_VARREF_NAME ] ) )

		# cycle through all new varrefs
		for aNewVarref in newVarrefList:
			# process insertions
			if aNewVarref[ MS_VARREF_NAME ] not in oldVarrefNames:	
				# new varref inserted - find whether insertion is rename
				continue
			else:
				for anOldVarref in oldVarrefList:
					if anOldVarref [ ME_VARREF_NAME ] == aNewVarref[ ME_VARREF_NAME ]:
						break
			if anOldVarref[ ME_VARREF_FULLID ] != aNewVarref[ ME_VARREF_FULLID ]:
				# process redirections
				oldVariable = getAbsoluteReference( aProcessFullID, anOldVarref[ ME_VARREF_FULLID ] )
				newVariable = getAbsoluteReference( aProcessFullID, aNewVarref[ ME_VARREF_FULLID ] )
				cmdList.extend( self.__redirectVarref( aProcessFullID, anOldVarref [ ME_VARREF_NAME ], oldVariable, newVariable ) )
			if anOldVarref[ ME_VARREF_COEF ] != aNewVarref[ ME_VARREF_COEF ]:
				# process coef changes
				cmdList.extend( self.__changeCoef( aProcessFullID, anOldVarref [ ME_VARREF_NAME ], aNewVarref[ ME_VARREF_COEF ] ) )

		return cmdList


	def __deleteConnections( self, aProcessFullID, aVarrefName, doNotTouch = None ):
		cmdList = []
		objectIter = self.theLayoutManager.createObjectIterator()
		objectIter.filterByFullID(aProcessFullID )
		while True:
			anObject = objectIter.getNextObject()
			if anObject == None:
				break
			connectionList = anObject.getProperty( PR_CONNECTIONLIST )
			aLayout = anObject.getLayout()
			for conID in connectionList:
				conObject = aLayout.getObject( conID )
				if conObject.getProperty( CO_NAME ) == aVarrefName and conObject.getID()!= doNotTouch:
					deleteCommand = DeleteObject( aLayout, conObject.getID() )
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
			aLayout = anObject.getLayout()	
			for conID in connectionList:
				conObject = aLayout.getObject( conID )
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
			aLayout = anObject.getLayout()
			connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
			for conID in connectionList:
				conObject = aLayout.getObject( conID )
				if conObject.getProperty( CO_NAME ) == aVarrefName:
					deleteCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), CO_COEF, newCoef )
					cmdList.append( deleteCommand )

		return cmdList

	def __changeVarrefName(self, aProcessFullID, oldVarrefName, newVarrefName ):
		cmdList = []
		objectIter = self.theLayoutManager.createObjectIterator()
		objectIter.filterByFullID( aProcessFullID )
		while True:
			anObject = objectIter.getNextObject()
			if anObject == None:
				break
			aLayout = anObject.getLayout()
			connectionList = anObject.getProperty( PR_CONNECTIONLIST )	
			for conID in connectionList:
				conObject = aLayout.getObject( conID )
				if conObject.getProperty( CO_NAME ) == oldVarrefName:
					renameCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), CO_NAME, newVarrefName )
					cmdList.append( renameCommand )

		return cmdList
		
	def __changeFullID( self, aSystemObject, newFullID ):
		returnCmdList = []
		newPath = convertSysIDToSysPath( newFullID )
		aLayout = aSystemObject.getLayout()
		for anID in aSystemObject.getObjectList():
			anObject = aLayout.getObject( anID )
			if not anObject.getProperty( OB_HASFULLID ):
				continue
			aFullID = anObject.getProperty( OB_FULLID )
			( aType, aPath, aName ) = aFullID.split(':')
			aFullID = ':'.join( [aType, newPath, aName ] )
			renameCommand = SetObjectProperty( aLayout, anID, OB_FULLID, aFullID )
			returnCmdList.append( renameCommand )
			if aType == ME_SYSTEM_TYPE:
				returnCmdList.extend( self.__changeFullID( anObject, aFullID  ) )
		return returnCmdList

	def __checkBrokenConnections(self, oldFullID, newFullID ):
		# should find whether
		return []
