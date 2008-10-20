#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

#from ecell.ObjectStub import *
from ObjectStub import *

# ---------------------------------------------------------------
# StepperStub -> ObjectStub
#   - provides an object-oriented appearance to the ecs.Simulator's Stepper API
#   - does not check validation of each argument.
# ---------------------------------------------------------------
class StepperStub( ObjectStub ):


    # ---------------------------------------------------------------
    # Constructor
    #
    # aSimulator : a reference to a Simulator
    # anID       : an ID of the Stepper
    #
    # return -> None
    # This method can throw exceptions.
    # ---------------------------------------------------------------
    def __init__( self, aSimulator, anID ):

        ObjectStub.__init__( self, aSimulator )

        self.theID = anID

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

        self.theSimulator.createStepper( aClassname, self.theID )

    # end of create

    # ---------------------------------------------------------------
    # delete
    #
    # return -> None
    # This method can throw exceptions.
    # ---------------------------------------------------------------

    def delete( self ):

        self.theSimulator.deleteStepper( self.theID )

    # end of delete


    # ---------------------------------------------------------------
    # exists
    #
    # return -> exist:TRUE / not exist:FALSE
    # This method can throw exceptions.
    # ---------------------------------------------------------------

    def exists( self ):
        return self.theID in self.theSimulator.getStepperList()

    # end of exists


    # ---------------------------------------------------------------
    # getClassname
    #
    # return -> None
    # This method can throw exceptions.
    # ---------------------------------------------------------------
    def getClassname( self ):

        return self.theSimulator.getStepperClassName( self.theID )

    # end of setClassname

    # ---------------------------------------------------------------
    # getPropertyList
    #
    # return -> a list of property names
    # This method can throw exceptions.
    # ---------------------------------------------------------------

    def getPropertyList( self ):

        return self.theSimulator.getStepperPropertyList( self.theID )

    # end of getPropertyList


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

        return self.setProperty( aPropertyName, aValue )

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
    
        return self.theSimulator.\
               getStepperPropertyAttributes( self.theID,\
                             aPropertyName )


    # end of getProperty


# end of LoggerStub


