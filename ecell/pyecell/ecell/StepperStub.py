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
# StepperStub -> PropertyInterfaceStub
#   - provides an object-oriented appearance to the ecs.Simulator's Stepper API
#   - does not check validation of each argument.
# ---------------------------------------------------------------
class StepperStub( PropertyInterfaceStub ):


	# ---------------------------------------------------------------
	# Constructor
	#
	# aSimulator : a reference to a Simulator
	# aClassname : a classname of the Stepper
	# anID       : an ID of the Stepper
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------
	def __init__( self, aSimulator, aClassname, anID ):

		PropertyInterfaceStub.__init__( self, aSimulator )

		self.theClassname = aClassname
		self.theID = anID

	# end of __init__



	# ---------------------------------------------------------------
	# createStepper
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def createStepper( self ):

		self.theLogger = self.theSimulator.createStepper( self.theClassname,
		                                                  self.theID )

	# end of getEntity


	# ---------------------------------------------------------------
	# isExist
	#
	# return -> exist:TRUE / not exist:FALSE
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def isExist( self ):

		for anID in self.theSimulator.getStepperList():
			if anID == self.theID:
				return TRUE
		return FALSE

	# end of isExist


	# ---------------------------------------------------------------
	# setStepperProperty
	#
	# aPropertyName : a property name
	# aValue        : a value to set
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def setProperty( self, aPropertyName, aValue ):

		return self.theSimulator.setStepperProperty( self.theID, 
		                                             aPropertyName, 
		                                             aValue )

	# end of setProperty


	# ---------------------------------------------------------------
	# __setitem__ ( = setStepperty )
	#
	# aPropertyName : a property name
	# aValue        : a value to set
	#
	# return -> None
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def __setitem__( self, aPropertyName, aValue ):

		return self.theSimulator.setStepperProperty( self.theID, 
		                                             aPropertyName, 
		                                             aValue )

	# end of setProperty


	# ---------------------------------------------------------------
	# getStepperProperty
	#
	# aPropertyName : a property name
	#
	# return -> the property
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def getProperty( self, aPropertyName ):

		return self.theSimulator.getStepperProperty( self.theID, 
		                                             aPropertyName )

	# end of getProperty


	# ---------------------------------------------------------------
	# __getitem__ ( = getStepperProperty )
	#
	# aPropertyName : a property name
	#
	# return -> the property
	# This method can throw exceptions.
	# ---------------------------------------------------------------

	def __getitem__( self, aPropertyName ):

		return self.theSimulator.getStepperProperty( self.theID, 
		                                             aPropertyName )

	# end of getProperty



# end of LoggerStub


