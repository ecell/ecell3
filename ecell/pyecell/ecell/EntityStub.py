#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>'
#
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


#from ecell.PropertyInterfaceStub import *
from PropertyInterfaceStub import *


# ---------------------------------------------------------------
# EntityStub -> PropertyInterfaceStub
#   - provides an object-oriented appearance to the ecs.Simulator's Entity API
#   - does not check validation of each argument.
# ---------------------------------------------------------------
class EntityStub( PropertyInterfaceStub ):


	# ---------------------------------------------------------------
	# Constructor
	#
	# aSimulator    : a reference to a Simulator 
	# aClassname    : a classname of the Entity to create.
	# aFullIDString : a FullID of the Entity as a String.
	# aName         : a name of the Entity.
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __init__( self, aSimulator, aClassname, aFullIDString, aName ):
	
		PropertyInterfaceStub.__init__( self, aSimulator )

		self.theClassname = aClassname
		self.theFullIDString = aFullIDString
		self.theName = aName

	# end of __init__


	# ---------------------------------------------------------------
	# setClassname
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def setClassname( self, aClassname ):

		self.theClassname = aClassname

	# end of setClassname


	# ---------------------------------------------------------------
	# getClassname
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def getClassname( self ):

		return self.theClassname 

	# end of setClassname


	# ---------------------------------------------------------------
	# setFullIDString
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def setFullIDString( self, aFullIDString ):

		self.theClassname = aClassname

	# end of setClassname


	# ---------------------------------------------------------------
	# createEntity
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def createEntity( self ):

		self.theSimulator.createEntity( self.theClassname, 
		                                self.theFullIDString, 
		                                self.theName )

	# end of createEntity


	# ---------------------------------------------------------------
	# isExist
	#
	# return -> exist:TRUE / not exist:FALSE
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def isExist( self ):

		return self.theSimulator.isEntityExist( self.theFullIDString )

	# end of isExist

	# ---------------------------------------------------------------
	# setProperty
	#
	# anAttribute : the attribute to set
	# aValue      : the value to set
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def setProperty( self, anAttribute, aValue ):
	
		self.theSimulator.setProperty( self.theFullIDString + ':' + anAttribute,
		                               aValue )

	# end of setProperty


	# ---------------------------------------------------------------
	# __setitem__ ( = setProperty )
	#
	# anAttribute : the attribute to set
	# aValue      : the value to set
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __setitem__( self, anAttribute, aValue ):
	
		self.theSimulator.setProperty( self.theFullIDString + ':' + anAttribute,
		                               aValue )

	# end of setProperty


	# ---------------------------------------------------------------
	# getProperty
	#
	# anAttribute : an attribute 
	#
	# return -> the property value
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def getProperty( self, anAttribute ):
	
		return self.theSimulator.getProperty( self.theFullIDString + ':' + anAttribute )

	# end of getProperty


	# ---------------------------------------------------------------
	# __getitem__ ( = getProperty )
	#
	# anAttribute : an attribute 
	#
	# return -> the property value
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __getitem__( self, anAttribute ):
	
		return self.theSimulator.getProperty( self.theFullIDString + ':' + anAttribute )

	# end of getProperty


# end of EntityStub


