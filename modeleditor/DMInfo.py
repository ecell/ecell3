

import ecell._ecs
from Config import *
import os
import os.path
from Constants import *
from Utils import *

class DMInfo:

	def __init__(self ):
		self.theSimulator = ecell._ecs.Simulator()
		
		# create system and variables
		self.theSimulator.createEntity(DM_SYSTEM_CLASS, 'System:/:System' )
		self.theSimulator.createEntity(DM_VARIABLE_CLASS, 'Variable:/:Variable' )



	def getClassList( self, aType ):
		if aType == ME_SYSTEM_TYPE:
			return [DM_SYSTEM_CLASS ]
		if aType == ME_VARIABLE_TYPE:
			return [DM_VARIABLE_CLASS ]

		aList = []
		# get from current directory
		curdir = '.'
		filelist = os.listdir( curdir )
		# get from module directory
		filelist.extend( os.listdir( DM_PATH ) )

		for aFile in filelist:

			if aFile.endswith( aType + '.so' ):
				aList.append( aFile.replace( '.so' , '') )

		return aList



	def getClassInfoList( self, aClass ):
		aFullID = self.__getFullID( aClass )
		return [DM_DESCRIPTION, DM_PROPERTYLIST, DM_ACCEPTNEWPROPERTY]



	def getClassInfo( self, aClass, anInfo ):
		aFullID = self.__getFullID( aClass )
		if anInfo == DM_DESCRIPTION:
			return "blahblah\n blah-blah"
		elif anInfo == DM_PROPERTYLIST:
			if aClass.endswith("Stepper"):
				return self.theSimulator.getStepperPropertyList( aClass )
			else:
				return self.theSimulator.getEntityPropertyList( aFullID )
			
		elif anInfo == DM_ACCEPTNEWPROPERTY:
			return True



	def getClassPropertyInfo( self, aClass, aProperty, anInfo ):
		aFullID = self.__getFullID( aClass )
		if anInfo == DM_PROPERTY_DEFAULTVALUE:
			aType = self.getClassPropertyInfo( aClass, aProperty, DM_PROPERTY_TYPE )
			if aType == DM_PROPERTY_STRING:
				return ''
			elif aType == DM_PROPERTY_NESTEDLIST:
				return []
			return "nothing"

		elif anInfo == DM_PROPERTY_SETTABLE_FLAG:
			if aClass.endswith("Stepper"):
				attr = self.theSimulator.getStepperPropertyAttributes( aClass, aProperty )
			else:
				attr = self.theSimulator.getEntityPropertyAttributes( createFullPN( aFullID, aProperty ) )
				
			return attr[ME_SETTABLE_FLAG]

		elif anInfo == DM_PROPERTY_GETTABLE_FLAG:
			if aClass.endswith(ME_STEPPER_TYPE):
				attr = self.theSimulator.getStepperPropertyAttributes( aClass, aProperty )
			else:
				attr = self.theSimulator.getEntityPropertyAttributes( createFullPN( aFullID, aProperty ) )
				
			return attr[ME_GETTABLE_FLAG]

		elif anInfo == DM_PROPERTY_DELETEABLE_FLAG:
			return False
		elif anInfo == DM_PROPERTY_TYPE:
			if aProperty == MS_PROCESS_VARREFLIST and aClass.endswith( ME_PROCESS_TYPE ):
				return DM_PROPERTY_NESTEDLIST
			elif aClass.endswith(ME_STEPPER_TYPE) and aProperty in [ MS_STEPPER_SYSTEMLIST, MS_STEPPER_PROCESSLIST ]:
				return DM_PROPERTY_NESTEDLIST
			elif aClass.endswith( ME_VARIABLE_TYPE ) and aProperty == MS_VARIABLE_PROCESSLIST:
				return DM_PROPERTY_NESTEDLIST
			return DM_PROPERTY_STRING
			


	def __getFullID( self, aClass):

		# get type
		if aClass.endswith(DM_SYSTEM_CLASS):
			aType = ME_SYSTEM_TYPE
		elif aClass.endswith(DM_VARIABLE_CLASS):
			aType = ME_VARIABLE_TYPE
		elif aClass.endswith('Process'):
			aType = 'Process'
		elif aClass.endswith('Stepper'):
			if aClass not in self.theSimulator.getStepperList():
				self.theSimulator.createStepper( aClass, aClass )
			return aClass
		else:
			raise Exception("Class type unambiguous")

		if aClass not in self.getClassList(aType):
			raise Exception("Unknown class: %s"%aClass)

		aFullID = aType + ':/:' + aClass
		if not self.theSimulator.isEntityExist( aFullID ):
			self.theSimulator.createEntity( aClass, aFullID )
		return aFullID




def DMTypeCheck( aValue, aType ):
	if aType == DM_PROPERTY_STRING:
		if type(aValue) == type([]):
			aValue = aValue[0]
		return str( aValue )
	elif aType == DM_PROPERTY_MULTILINE:
		if type(aValue) == type([]):
			aValue = '/n'.join(aValue)

		return str( aValue )
	elif aType == DM_PROPERTY_NESTEDLIST:
		if type(aValue) in ( type( () ), type( [] ) ):
			return aValue
		else:
			return None
	elif aType == DM_PROPERTY_INTEGER:
		if type(aValue) == type([]):
			aValue = aValue[0]
		try:
			aValue = Int( aValue )
		except:
			return None
		return aValue
	elif aType == DM_PROPERTY_FLOAT:
		if type(aValue) == type([]):
			aValue = aValue[0]
		try:
			aValue = Float( aValue )
		except:
			return None
		return aValue
	else:
		return None

