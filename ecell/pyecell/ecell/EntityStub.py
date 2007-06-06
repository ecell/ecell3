#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
#'Design: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>'
#
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#


from ObjectStub import *


# ---------------------------------------------------------------
# EntityStub -> ObjectStub
#   - provides an object-oriented appearance to the ecs.Simulator's Entity API
#   - does not check validation of each argument.
# ---------------------------------------------------------------
class EntityStub( ObjectStub ):


	# ---------------------------------------------------------------
	# Constructor
	#
	# aSimulator    : a reference to a Simulator 
	# aFullIDString : a FullID of the Entity as a String.
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __init__( self, aSimulator, aFullIDString ):
	
		ObjectStub.__init__( self, aSimulator )
		self.theFullIDString = aFullIDString

	# end of __init__

	def getName( self ):
		return self.theFullIDString

	
	# ---------------------------------------------------------------
	# create
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def create( self, aClassname ):

		self.theSimulator.createEntity( aClassname,
		                                self.theFullIDString ) 

	# end of createEntity

	# ---------------------------------------------------------------
	# delete
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def delete( self ):

		self.theSimulator.deleteEntity( self.theFullIDString ) 

	# end of createEntity


	# ---------------------------------------------------------------
	# getClassname
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def getClassname( self ):

		return self.theSimulator.\
		       getEntityClassName( self.theFullIDString )

	# end of setClassname


	# ---------------------------------------------------------------
	# getPropertyList
	#
	# return -> a list of property names
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def getPropertyList( self ):

		return self.theSimulator.\
		       getEntityPropertyList( self.theFullIDString )

	# end of getPropertyList


	# ---------------------------------------------------------------
	# exists
	#
	# return -> exist:TRUE / not exist:FALSE
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def exists( self ):

		return self.theSimulator.isEntityExist( self.theFullIDString )

	# end of exists

	# ---------------------------------------------------------------
	# setProperty
	#
	# aPropertyName : name of the property to set
	# aValue        : the value to set
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def setProperty( self, aPropertyName, aValue ):

		aFullPN = self.theFullIDString + ':' + aPropertyName	

		self.theSimulator.setEntityProperty( aFullPN, aValue )

	# end of setProperty


	# ---------------------------------------------------------------
	# __setitem__ ( = setProperty )
	#
	# see setProperty().
	#
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __setitem__( self, aPropertyName, aValue ):
	
		self.setProperty( aPropertyName, aValue )

	# end of setProperty


	# ---------------------------------------------------------------
	# getProperty
	#
	# aPropertyName : name of the property to get
	#
	# return -> the property value
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def getProperty( self, aPropertyName ):

		aFullPN = self.theFullIDString + ':' + aPropertyName
		return self.theSimulator.getEntityProperty( aFullPN )

	# end of getProperty


	# ---------------------------------------------------------------
	# __getitem__ ( = getProperty )
	#
	# see getProperty().
	#
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __getitem__( self, aPropertyName ):
	
		return self.getProperty( aPropertyName )

	# end of getProperty


	# ---------------------------------------------------------------
	# getPropertyAttributes
	#
	# aPropertyName : name of the property to get
	#
	# return -> boolean 2-tuple ( setable, getable )
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def getPropertyAttributes( self, aPropertyName ):
	
		aFullPN = self.theFullIDString + ':' + aPropertyName
		return self.theSimulator.getEntityPropertyAttributes( aFullPN )


	# end of getProperty

# end of EntityStub


