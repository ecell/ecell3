
import string
from DMInfo import *
from Constants import *
from Utils import *


class ModelStore:


	
	def __init__( self ):
		"""make ModelStore"""

		self.__theModel = [] 
		self.__theStepper = {}
		self.__theEntity = {}
		self.__theDM = DMInfo()
		self.__theEntity[MS_SYSTEM_ROOT] = [ DM_SYSTEM_CLASS, None, None, [],  [], [], 'RootSystem' ]
		self.__theEntity[MS_SYSTEM_ROOT][MS_ENTITY_PROPERTYLIST] = self.__createPropertyList( DM_SYSTEM_CLASS )

	##---------------------------------------------
	## Methods for Stepper
	##---------------------------------------------

	def createStepper( self, aClass, anID ):
		"""create a stepper"""

		if anID in self.__theStepper.keys():
			raise Exception("Stepper %s already exists!"%anID)
		if aClass not in self.__theDM.getClassList( ME_STEPPER_TYPE ):
			raise Exception("Unknown class name %s!"%aClass)

		aPropertyList = self.__createPropertyList( aClass )

		self.__theStepper[anID] = [ aClass, aPropertyList, "%s class"%aClass ]

		# find referring processes systems
		for anEntityID in self.__theEntity.keys():
			anEntityType = getFullIDType( anEntityID )
			if  anEntityType not in [ ME_PROCESS_TYPE, ME_SYSTEM_TYPE ]:
				continue
			aStepperID = self.__getEntityProperty( anEntityID, ME_STEPPERID )[MS_PROPERTY_VALUE]
			if aStepperID == anID:
				self.__registerEntityToStepper( anEntityID, aStepperID )





	def deleteStepper( self, anID ):
		"""delete a stepper"""
		# check if Stepper exists
		if anID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%anID )

		# delete attached systems
		#aStepper = self.__theStepper[anID]
		#aSystemList = aStepper[MS_STEPPER_PROPERTYLIST][MS_STEPPER_SYSTEMLIST][MS_PROPERTY_VALUE]
		#for aSystem in aSystemList :
		#	self.setEntityProperty( createFullPN( aSystem, MS_SYSTEM_STEPPERID ), '' )

		# delete attached processes
		#aProcessList = aStepper[MS_STEPPER_PROPERTYLIST][MS_STEPPER_PROCESSLIST][MS_PROPERTY_VALUE]
		#for aProcess in aProcessList:
		#	self.setEntityProperty( createFullPN( aProcess, MS_PROCESS_STEPPERID ), '' )

		del self.__theStepper[anID]


	def getStepperList( self):

		return self.__theStepper.keys()


	def getStepperClassName( self, aStepperID ):
		# check if Stepper exists
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )
		return self.__theStepper[aStepperID][MS_STEPPER_CLASS]
	


	def getStepperPropertyList( self, aStepperID ):
		# check if Stepper exists
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )
		return copyValue ( self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST].keys() )



	def createStepperProperty( self, aStepperID, aPropertyName, aValue, aType = DM_PROPERTY_STRING, anAttribute = (1,1,1,1,1) ):
		"""create a Stepper property"""

		# check if Stepper exists
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )

		aClass = self.getStepperClassName( aStepperID )
		if not self.__theDM.getClassInfo( aClass, DM_ACCEPTNEWPROPERTY ):
			raise Exception("Cannot create new property for stepper %s"%aStepperID )
		
		if aPropertyName in self.getStepperPropertyList( aStepperID ):
			raise Exception("Property %s of stepper %s already exists!"%(aPropertyName,aStepperID))


		convertValue = copyValue( DMTypeCheck( aValue, aType ) )
		if convertValue == None:
			raise Exception("Invalid value for Property %s of stepper %s!"%(aPropertyName, aStepperID) )

		# create property
		self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][aPropertyName] = [convertValue, copyValue(anAttribute), aType]



	def __getStepperProperty( self, aStepperID, aPropertyName ):
		# check if Stepper exists
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )
		if aPropertyName not in self.getStepperPropertyList( aStepperID ):
			raise Exception("Property %s of stepper %s does not exist!"%(aPropertyName,aStepperID))

		return self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][aPropertyName]

	

	def loadStepperProperty( self, aStepperID, aPropertyName, aValue, aType = DM_PROPERTY_STRING, anAttribute = (1,1,1,1,1) ):
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )
		if aPropertyName not in self.getStepperPropertyList( aStepperID ):
			self.createStepperProperty( aStepperID, aPropertyName, aValue, aType, anAttribute )
		else:
			self.setStepperProperty( aStepperID, aPropertyName, aValue )


	def saveStepperProperty( self, aStepperID, aPropertyName ):
		return self.getStepperProperty( aStepperID, aPropertyName )


	def setStepperProperty( self, aStepperID, aPropertyName, aValue ):
		"""set a Stepper property"""
		aProperty = self.__getStepperProperty( aStepperID, aPropertyName )

		# check settable flag settable 
		if aProperty[MS_PROPERTY_FLAGS][MS_SETTABLE_FLAG]:
			aType = aProperty[MS_PROPERTY_TYPE]
			convertValue = DMTypeCheck( aValue, aType )
			if convertValue != None:
				aProperty[MS_PROPERTY_VALUE] = copyValue( aValue )
			else:
				raise Exception("Invalid value for Property %s of stepper %s!"%(aPropertyName, aStepperID) )
		else:
			raise Exception("Property %s of stepper %s is not settable!"%(aPropertyName, aStepperID) )



	def getStepperProperty( self, aStepperID, aPropertyName ):
		"""get a Stepper property"""

		aProperty = self.__getStepperProperty( aStepperID, aPropertyName )

		# check settable flag settable 
		if aProperty[MS_PROPERTY_FLAGS][MS_GETTABLE_FLAG]:
			return aProperty[MS_PROPERTY_VALUE]
		else:
			raise Exception("Property %s not gettable"%aPropertyName )




	def deleteStepperProperty( self, aStepperID, aPropertyName ):
		"""delete a Stepper property"""

		aProperty = self.__getStepperProperty( aStepperID, aPropertyName )

		# check settable flag settable 
		if aProperty[MS_PROPERTY_FLAGS][MS_DELETEABLE_FLAG]:
			del self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][aPropertyName]
		else:
			raise Exception("Property %s of stepper %s is not deleteable!"%(aPropertyName, aStepperID) )



	def getStepperPropertyAttributes( self, aStepperID, aPropertyName ):
		""" get a Stepper property attribute"""
		return self.__getStepperProperty( aStepperID, aPropertyName )[MS_PROPERTY_FLAGS]



	def getStepperPropertyType( self, aStepperID, aPropertyName ):
		""" candidates are:
		DM_PROPERTY_STRING,  DM_PROPERTY_MULTILINE, etc.
		"""
		return self.__getStepperProperty( aStepperID, aPropertyName )[MS_PROPERTY_TYPE]



	def setStepperInfo( self, aStepperID, anInfoStrings ):
		# check if Stepper exists
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )

		self.__theStepper[aStepperID][MS_STEPPER_INFO] = str( anInfoStrings )



	def getStepperInfo( self, aStepperID ):
		# check if Stepper exists
		if aStepperID not in self.__theStepper.keys():
			raise Exception("Stepper %s does not exist!"%aStepperID )
		return self.__theStepper[aStepperID][MS_STEPPER_INFO]

	

		
	##---------------------------------------------
	## Methods for Entity
	##---------------------------------------------

	def __getEntityType( self, aFullID ):

		aFullIDParts = string.split( aFullID, ':' )
		return aFullIDParts[0]



	def __getEntityPath( self, aFullID ):

		theEntityPath = string.split( aFullID, ':' )
		return theEntityPath[1]



	def __getFullID( self, aFullPNString ):
		"""get FullID from FullPN"""

		aFullPNStringParts = string.split( aFullPNString, ':')
		aFullID = string.join( aFullPNStringParts[:3], ':')
		return aFullID



	def __getPropertyName( self, aFullPNString ):
		"""get property name from FullPN"""

		aFullPNStringParts = string.split( aFullPNString, ':')
		return aFullPNStringParts[3]


	def __getEntity( self, aFullID ):
		# check if Entity exists
		if aFullID not in self.__theEntity.keys():
			return None
		else:
			return self.__theEntity[ aFullID ]


	def __createPropertyList( self, aClass ):
		aPropertyList = {}
		aPropertyNames = self.__theDM.getClassInfo( aClass, DM_PROPERTYLIST )
		for aPropertyName in aPropertyNames:
			if aClass == DM_SYSTEM_CLASS and ( aPropertyName ==MS_STEPPER_PROCESSLIST or aPropertyName == MS_STEPPER_SYSTEMLIST ):
				aDefaultValue = []
				aType = DM_PROPERTY_NESTEDLIST
				aFlags = ( False, False, False, False, False )
			else:
				aDefaultValue = self.__theDM.getClassPropertyInfo( aClass, aPropertyName, DM_PROPERTY_DEFAULTVALUE )
				aType = self.__theDM.getClassPropertyInfo( aClass, aPropertyName, DM_PROPERTY_TYPE )

				settable = self.__theDM.getClassPropertyInfo( aClass, aPropertyName, DM_PROPERTY_SETTABLE_FLAG )
				gettable = self.__theDM.getClassPropertyInfo( aClass, aPropertyName, DM_PROPERTY_GETTABLE_FLAG )
				deleteable = self.__theDM.getClassPropertyInfo( aClass, aPropertyName, DM_PROPERTY_DELETEABLE_FLAG )
				aFlags = ( settable, gettable, False, False, deleteable )
			aPropertyList[aPropertyName] = [ aDefaultValue, aFlags, aType ]

		if aClass == DM_VARIABLE_CLASS:
			aPropertyList[MS_VARIABLE_PROCESSLIST] = [ [], (False, True, False, False, False), DM_PROPERTY_NESTEDLIST ]
		return aPropertyList



	def __getEntityProperty( self, anEntity, aPropertyName ):
		# check if Entity exists
		if anEntity not in self.__theEntity.keys():
			raise Exception("Stepper %s does not exist!"%anEntity )
		if aPropertyName not in self.getEntityPropertyList( anEntity ):
			raise Exception("Property %s of entity %s does not exist!"%(aPropertyName,anEntity))

		return self.__theEntity[anEntity][MS_ENTITY_PROPERTYLIST][aPropertyName]



	def createEntity( self, aClass, aFullID ):

		if aFullID in self.__theEntity.keys():
			raise Exception("Entity %s already exists!"%aFullID )

		anEntityType = self.__getEntityType( aFullID )

		if aClass not in self.__theDM.getClassList( anEntityType ):
			raise Exception("Unknown class name %s!"%aClass)

		aParentFullID = convertSysPathToSysID( self.__getEntityPath( aFullID ) )
		aParentSystem = self.__getEntity( aParentFullID )
		if aParentSystem == None:
			raise Exception( "Parent system of %s doesnot exist!"%aFullID )

		aPropertyList = self.__createPropertyList( aClass )

		self.__theEntity[aFullID] = [ aClass, aPropertyList, aParentSystem, [],  [], [], 'A System' ]
		if anEntityType == ME_SYSTEM_TYPE:
			#aParentSystem[MS_ENTITY_CHILD_SYSTEMLIST].append( self.__theEntity[aFullID] )
			pass
		elif anEntityType == ME_PROCESS_TYPE:
			pass
			#aParentSystem[MS_ENTITY_CHILD_PROCESSLIST].append( self.__theEntity[aFullID] )
		else:
			#aParentSystem[MS_ENTITY_CHILD_VARIABLELIST].append( self.__theEntity[aFullID] )
			# find process referencing this variable
			for aProcessID in self.__theEntity.keys():
				if getFullIDType( aProcessID ) != ME_PROCESS_TYPE:
					continue
				aVarrefProperty = self.__getEntityProperty( aProcessID, MS_PROCESS_VARREFLIST )[MS_PROPERTY_VALUE]
				for aVarref in aVarrefProperty:
					aVariableRef = aVarref[MS_VARREF_FULLID]
					aVariableFullID =  getAbsoluteReference( aProcessID, aVariableRef )
					if aVariableFullID == aFullID:
						self.__registerProcessToVariable( aProcessID, aVarrefProperty )


		


	def deleteEntity( self, aFullID ):
		"""delete an entity"""
		# check if exists

		if aFullID not in self.__theEntity.keys():
			raise Exception("Entity %s does not exist!"%FullID )

		anEntityType = self.__getEntityType( aFullID )

		# delete from parent
		#system
		# delete from stepper.systemlist

		#process
		# delete form stepper.processlist
		# delete from variable.processlist

		aParentFullID = convertSysPathToSysID( self.__getEntityPath( aFullID ) )

		aParentSystem = self.__getEntity( aParentFullID )


		if anEntityType == ME_SYSTEM_TYPE:
			# cascade delete
			aSysPath = convertSysIDToSysPath( aFullID )

			for aType in [ ME_SYSTEM_TYPE, ME_PROCESS_TYPE, ME_VARIABLE_TYPE ]:

			# get subsystems
				anIDList = self.getEntityList(  aType, aSysPath )

				for anID in anIDList:
					newFullID = aType + ':' + aSysPath + ':' + anID
					self.deleteEntity( newFullID )
			


			#aParentSystem[MS_ENTITY_CHILD_SYSTEMLIST].remove( self.__theEntity[aFullID] )
			self.__deregisterEntityFromStepper( aFullID )
		elif anEntityType == ME_PROCESS_TYPE:
			#aParentSystem[MS_ENTITY_CHILD_PROCESSLIST].remove( self.__theEntity[aFullID] )
			self.__deregisterEntityFromStepper( aFullID )
			aVarrefProperty = self.__getEntityProperty( aFullID, MS_PROCESS_VARREFLIST )[MS_PROPERTY_VALUE]
			self.__deregisterProcessFromVariable( aFullID, aVarrefProperty )
		else:
			#aParentSystem[MS_ENTITY_CHILD_VARIABLELIST].remove( self.__theEntity[aFullID] )
			pass

		del self.__theEntity[aFullID]



	def isEntityExist( self, aFullID ):
		return aFullID in self.__theEntity.keys()


	def getEntityClassName( self, aFullID ):
		anEntity = self.__getEntity( aFullID )
		if anEntity == None:
			raise Exception( "Entity %s does not exist!"%aFullID )
		return anEntity[MS_ENTITY_CLASS]

	
	def getEntityList( self, anEntityType, aSystemPath ):

		"""return the Entity list on the SystemPath"""

		theEntityList = []

		allEntityFullID = self.__theEntity.keys()
		for aFullID in allEntityFullID :
			if aSystemPath == self.__getEntityPath( aFullID ) :
				if anEntityType == self.__getEntityType( aFullID ) :
					theEntityList.append( aFullID.split(':')[2] )
		return theEntityList



	def __registerEntityToStepper( self, aFullID, aStepperID ):
		if aStepperID in [None, '' ]:
			return
		if self.__getEntityType( aFullID ) == ME_SYSTEM_TYPE:
			aStepperID = self.__getEntityProperty( aFullID, MS_SYSTEM_STEPPERID )[MS_PROPERTY_VALUE]
			if aStepperID in self.__theStepper.keys():
				self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][MS_STEPPER_SYSTEMLIST][MS_PROPERTY_VALUE].append( aFullID )
		else:
			aStepperID = self.__getEntityProperty( aFullID, MS_PROCESS_STEPPERID )[MS_PROPERTY_VALUE]
			if aStepperID in self.__theStepper.keys():
				self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][MS_STEPPER_PROCESSLIST][MS_PROPERTY_VALUE].append( aFullID )



	def __deregisterEntityFromStepper( self, aFullID ):
		if self.__getEntityType( aFullID ) == ME_SYSTEM_TYPE:
			aStepperID = self.__getEntityProperty( aFullID, MS_SYSTEM_STEPPERID )[MS_PROPERTY_VALUE]
			if aStepperID in self.__theStepper.keys():
				self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][MS_STEPPER_SYSTEMLIST][MS_PROPERTY_VALUE].remove( aFullID )
		else:
			aStepperID = self.__getEntityProperty( aFullID, MS_PROCESS_STEPPERID )[MS_PROPERTY_VALUE]
			if aStepperID in self.__theStepper.keys():
				self.__theStepper[aStepperID][MS_STEPPER_PROPERTYLIST][MS_STEPPER_PROCESSLIST][MS_PROPERTY_VALUE].remove( aFullID )



	def __registerProcessToVariable( self, aProcessID, aVarrefProperty ):
		for aVarref in aVarrefProperty:
			aVariableRef = aVarref[MS_VARREF_FULLID]
			aVariableFullID = getAbsoluteReference( aProcessID, aVariableRef )
			if not self.isEntityExist( aVariableFullID ):
				return

			aVariable = self.__getEntity( aVariableFullID )
			if aProcessID not in aVariable[MS_ENTITY_PROPERTYLIST][MS_VARIABLE_PROCESSLIST][MS_PROPERTY_VALUE]:
				aVariable[MS_ENTITY_PROPERTYLIST][MS_VARIABLE_PROCESSLIST][MS_PROPERTY_VALUE].append( aProcessID )


	def __deregisterProcessFromVariable( self, aProcessID, aVarrefProperty ):

		for aVarref in aVarrefProperty:
			aVariableRef = aVarref[MS_VARREF_FULLID]
			aVariableFullID = getAbsoluteReference( aProcessID, aVariableRef )
			if not self.isEntityExist( aVariableFullID ):
				return
			aVariable = self.__getEntity( aVariableFullID )
			if aProcessID in aVariable[MS_ENTITY_PROPERTYLIST][MS_VARIABLE_PROCESSLIST][MS_PROPERTY_VALUE]:
				aVariable[MS_ENTITY_PROPERTYLIST][MS_VARIABLE_PROCESSLIST][MS_PROPERTY_VALUE].remove( aProcessID )



	def getEntityPropertyList( self, aFullID ):
		"""get an entity property list"""
		anEntity = self.__getEntity( aFullID )
		if anEntity == None:
			raise Exception( "Entity %s does not exist!"%anEntity )
		return copyValue( anEntity[MS_ENTITY_PROPERTYLIST ].keys() )



	def createEntityProperty( self, aFullID, aPropertyName, aValueList, aType = DM_PROPERTY_STRING, anAttribute = (1,1,1,1,1) ):
		"""create an entity property"""
		anEntity = self.__getEntity( aFullID )
		if anEntity == None:
			raise Exception( "Entity %s does not exist!"%anEntity )
		aClass = self.getEntityClassName( aFullID )
		if not self.__theDM.getClassInfo( aClass, DM_ACCEPTNEWPROPERTY ):
			raise Exception("Cannot create new property for stepper %s"%aFullID )

		if aPropertyName in self.getEntityPropertyList( aFullID ):
			raise Exception( "Property %s of entity %s already exists!"%( aPropertyName, anEntity ) )
		convertValue = DMTypeCheck( aValueList, aType )
		if convertValue == None:
			raise Exception("Invalid value %s for property %s"%(aValueList, aPropertyName ) )
		newProperty = [ copyValue( convertValue ), copyValue( anAttribute ), aType ]
		anEntity[MS_ENTITY_PROPERTYLIST][aPropertyName] =  newProperty 
		anEntityType = self.__getEntityType( aFullID )
		if anEntityType == ME_SYSTEM_TYPE:
			#watch for stepperID
			if aPropertyName == MS_SYSTEM_STEPPERID:
				self.__registerEntityToStepper( aFullID, convertValue )
		elif anEntityType == ME_PROCESS_TYPE:
			if aPropertyName == MS_PROCESS_STEPPERID:
				self.__registerEntityToStepper( aFullID, convertValue )
			elif aPropertyName == MS_PROCESS_VARREFLIST:
				self.__registerProcessToVariable( aFullID, convertValue )



	def setEntityProperty( self, aFullPN, aValueList ):
		"""set an entity property"""
		aFullID = self.__getFullID( aFullPN )
		aPropertyName = self.__getPropertyName( aFullPN )
		aProperty = self.__getEntityProperty( aFullID, aPropertyName )
		if not aProperty[MS_PROPERTY_FLAGS][MS_SETTABLE_FLAG]:
			raise Exception( "Property %s is not settable!"%aFullPN )
		aType = aProperty[MS_PROPERTY_TYPE]

		convertValue = DMTypeCheck( aValueList, aType )
		if convertValue == None:
			raise Exception("Invalid value %s for property %s"%(aValueList, aPropertyName ) )
		anEntityType = self.__getEntityType ( aFullID )
		#deregistering
		if anEntityType == ME_SYSTEM_TYPE:
			#watch for stepperID
			if aPropertyName == MS_SYSTEM_STEPPERID:
				self.__deregisterEntityFromStepper( aFullID )
		elif anEntityType == ME_PROCESS_TYPE:
			if aPropertyName == MS_PROCESS_STEPPERID:
				self.__deregisterEntityFromStepper( aFullID )
			elif aPropertyName == MS_PROCESS_VARREFLIST:
				oldValue = aProperty[MS_PROPERTY_VALUE]
				self.__deregisterProcessFromVariable( aFullID, oldValue )
		# set property
		aProperty[MS_PROPERTY_VALUE] = copyValue( convertValue )
		
		# reregistering
		if anEntityType == ME_SYSTEM_TYPE:
			#watch for stepperID
			if aPropertyName == MS_SYSTEM_STEPPERID:
				self.__registerEntityToStepper( aFullID, convertValue )
		elif anEntityType == ME_PROCESS_TYPE:
			if aPropertyName == MS_PROCESS_STEPPERID:
				self.__registerEntityToStepper( aFullID, convertValue )
			elif aPropertyName == MS_PROCESS_VARREFLIST:

				self.__registerProcessToVariable( aFullID, convertValue )



	def loadEntityProperty( self, aFullPN, aValue, aType = DM_PROPERTY_STRING, anAttribute = (1,1,1,1,1) ):
		aFullID = self.__getFullID( aFullPN )
		aPropertyName = self.__getPropertyName( aFullPN )
		if aFullID not in self.__theEntity.keys():
			raise Exception("Entity %s does not exist!"%FullID )
		
		if aPropertyName in self.getEntityPropertyList( aFullID ):

			self.setEntityProperty( aFullPN, aValue )
		else:
			self.createEntityProperty( aFullID, aPropertyName, aValue[0], aType, anAttribute )
		

	def saveEntityProperty( self, aFullPN ):
		return self.getEntityProperty( aFullPN )




	def getEntityProperty( self, aFullPN ):
		"""get an entity property"""
		aFullID = self.__getFullID( aFullPN )
		aPropertyName = self.__getPropertyName( aFullPN )
		aProperty = self.__getEntityProperty( aFullID, aPropertyName )
		if not aProperty[MS_PROPERTY_FLAGS][MS_GETTABLE_FLAG]:
			raise Exception( "Property %s is not gettable!"%aFullPN )
		return copyValue( aProperty[MS_PROPERTY_VALUE] )


		
	def deleteEntityProperty( self, aFullID, aPropertyName ):
		"""delete an entity property"""
		aFullPN = aFullID + ':' + aPropertyName
		aPropertyName = self.__getPropertyName( aFullPN )
		aProperty = self.__getEntityProperty( aFullID, aPropertyName )
		if not aProperty[MS_PROPERTY_FLAGS][MS_DELETEABLE_FLAG]:
			raise Exception( "Property %s is not settable!"%aFullPN )
		# deregister
		anEntityType = getFullIDType( aFullID )
		if anEntityType == ME_SYSTEM_TYPE:
			#watch for stepperID
			if aPropertyName == MS_SYSTEM_STEPPERID:
				self.__deregisterEntityFromStepper( aFullID )
		elif anEntityType == ME_PROCESS_TYPE:
			if aPropertyName == MS_PROCESS_STEPPERID:
				self.__deregisterEntityFromStepper( aFullID )
			elif aPropertyName == MS_PROCESS_VARREFLIST:
				self.__deregisterProcessFromVariable( aFullID, convertValue )
		anEntity = self.__getEntity( aFullID )
		del anEntity[MS_ENTITY_PROPERTYLIST][aPropertyName]



	def getEntityPropertyAttributes( self, aFullPN ):
		"""get an entity property attribute"""
		aFullID = self.__getFullID( aFullPN )
		aPropertyName = self.__getPropertyName( aFullPN )
		aProperty = self.__getEntityProperty( aFullID, aPropertyName )
		return copyValue( aProperty[MS_PROPERTY_FLAGS] )


	def setEntityInfo( self, aFullID, InfoStrings ):
		anEntity = self.__getEntity( aFullID )
		if anEntity == None:
			raise Exception( "Entity %s does not exist!"%anEntity )
		anEntity[MS_ENTITY_INFO] = copyValue( InfoStrings )


	def getEntityInfo( self, aFullID ):
		anEntity = self.__getEntity( aFullID )
		if anEntity == None:
			raise Exception( "Entity %s does not exist!"%anEntity )
		return copyValue( anEntity[MS_ENTITY_INFO] )



	def getEntityPropertyType( self, aFullPN ):
		""" candidates are:
			DM_PROPERTY types
		"""
		aFullID = self.__getFullID( aFullPN )
		aPropertyName = self.__getPropertyName( aFullPN )
		aProperty = self.__getEntityProperty( aFullID, aPropertyName )
		return aProperty[MS_PROPERTY_TYPE]
