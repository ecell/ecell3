#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
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

from ecell.ObjectStub import *

class EntityStub( ObjectStub ):
    """
    EntityStub -> ObjectStub
      - provides an object-oriented appearance to the ecs.Simulator's Entity API
      - does not check validation of each argument.
    """
    def __init__( self, aSimulator, aFullIDString ):
        """
        aSimulator    : a reference to a Simulator 
        aFullIDString : a FullID of the Entity as a String.
        
        return -> None
        This method can throw exceptions.
        """
        ObjectStub.__init__( self, aSimulator )
        self.theFullIDString = aFullIDString

    def getName( self ):
        return self.theFullIDString

    def create( self, aClassname ):
        """ 
        return -> None
        This method can throw exceptions.
        """
        self.theSimulator.createEntity( aClassname,
                                        self.theFullIDString ) 

    def delete( self ):
        """
        return -> None
        This method can throw exceptions.
        """
        self.theSimulator.deleteEntity( self.theFullIDString ) 

    def getClassname( self ):
        """
        return -> None
        This method can throw exceptions.
        """
        return self.theSimulator.\
               getEntityClassName( self.theFullIDString )

    def getPropertyList( self ):
        """
        return -> a list of property names
        This method can throw exceptions.
        """
        return self.theSimulator.\
               getEntityPropertyList( self.theFullIDString )

    def exists( self ):
        """
        return -> exist:TRUE / not exist:FALSE
        This method can throw exceptions.
        """
        return self.theSimulator.isEntityExist( self.theFullIDString )

    def setProperty( self, aPropertyName, aValue ):
        """
        aPropertyName : name of the property to set
        aValue        : the value to set
        
        return -> None
        This method can throw exceptions.
        """
        aFullPN = self.theFullIDString + ':' + aPropertyName	
        self.theSimulator.setEntityProperty( aFullPN, aValue )

    def __setitem__( self, aPropertyName, aValue ):
        """
        see setProperty().
        
        This method can throw exceptions.
        """
        self.setProperty( aPropertyName, aValue )

    def getProperty( self, aPropertyName ):
        """
        aPropertyName : name of the property to get
        
        return -> the property value
        This method can throw exceptions.
        """
        aFullPN = self.theFullIDString + ':' + aPropertyName
        return self.theSimulator.getEntityProperty( aFullPN )

    def __getitem__( self, aPropertyName ):
        """
        see getProperty().
        
        This method can throw exceptions.
        """
        return self.getProperty( aPropertyName )

    def getPropertyAttributes( self, aPropertyName ):
        """
        aPropertyName : name of the property to get
        
        return -> boolean 2-tuple ( setable, getable )
        This method can throw exceptions.
        """
        aFullPN = self.theFullIDString + ':' + aPropertyName
        return self.theSimulator.getEntityPropertyAttributes( aFullPN )

# end of EntityStub


