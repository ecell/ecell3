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
from ecell.ObjectStub import ObjectStub
import ecell.identifiers as identifiers

__all__ = (
    'EntityStub'
    )

class EntityStub( ObjectStub ):
    """
    EntityStub -> ObjectStub
      - provides an object-oriented appearance to the ecs.Simulator's Entity API
      - does not check validation of each argument.
    """
    def __init__( self, aSimulator, aFullID ):
        """
        Constructor
        
        aSimulator    : a reference to a Simulator 
        aFullIDString : a FullID of the Entity as a String.
        
        return -> None
        This method can throw exceptions.
        """
        ObjectStub.__init__( self, aSimulator )
        self.theFullID = identifiers.FullID( aFullID )
        self.theFullIDString = str( self.theFullID )

    def getFullID( self ):
        return self.theFullID

    def getName( self ):
        return self.theFullIDString

    def create( self, aClassname ):
        """
        create
        
        return -> None
        This method can throw exceptions.
        """
        self.theSimulator.createEntity( aClassname,
                                        self.theFullIDString ) 

    def delete( self ):
        """
        delete
        
        return -> None
        This method can throw exceptions.
        """
        self.theSimulator.deleteEntity( self.theFullIDString ) 

    def getClassname( self ):
        """
        getClassname
        
        return -> None
        This method can throw exceptions.
        """
        return self.theSimulator.getEntityClassName( self.theFullIDString )

    def getPropertyList( self ):
        """
        getPropertyList
        
        return -> a list of property names
        This method can throw exceptions.
        """
        return self.theSimulator.getEntityPropertyList( self.theFullIDString )

    def exists( self ):
        """
        exists
        
        return -> exist:TRUE / not exist:FALSE
        This method can throw exceptions.
        """
        return self.theSimulator.isEntityExist( self.theFullIDString )

    def setProperty( self, aPropertyName, aValue ):
        """
        setProperty
        
        aPropertyName : name of the property to set
        aValue        : the value to set
        
        return -> None
        This method can throw exceptions.
        """
        aFullPN = self.theFullID.createFullPN( aPropertyName )
        self.theSimulator.setEntityProperty( str( aFullPN ), aValue )

    def __setitem__( self, aPropertyName, aValue ):
        """
        __setitem__ ( = setProperty )
        
        see setProperty().
        
        This method can throw exceptions.
        """
        self.setProperty( aPropertyName, aValue )

    def getProperty( self, aPropertyName ):
        """
        getProperty
        
        aPropertyName : name of the property to get
        
        return -> the property value
        This method can throw exceptions.
        """
        aFullPN = self.theFullID.createFullPN( aPropertyName )
        return self.theSimulator.getEntityProperty( str( aFullPN ) )

    def __getitem__( self, aPropertyName ):
        """
        __getitem__ ( = getProperty )
        
        see getProperty().
        
        This method can throw exceptions.
        """
        return self.getProperty( aPropertyName )

    def getPropertyAttributes( self, aPropertyName ):
        """
        getPropertyAttributes
        
        aPropertyName : name of the property to get
        
        return -> boolean 2-tuple ( setable, getable )
        This method can throw exceptions.
        """
        aFullPN = self.theFullID.createFullPN( aPropertyName )
        return self.theSimulator.getEntityPropertyAttributes( str( aFullPN ) )


