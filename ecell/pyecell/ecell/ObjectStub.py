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

__all__ = (
    'ObjectStub'
    )

class ObjectStub:
    """
    ObjectStub
      - provides an object-oriented appearance to the ecs.Simulator's API
      - does not check validation of each argument.
    """

    def __init__( self, aSimulator ):
        """
        Constructor (must be called in constructor of subclass)
        
        aSimulator    : a reference to a Simulator 
        
        return -> None
        This method may throw an exception.
        """
        self.theSimulator = aSimulator

    def getName( self ):
        raise NotImplementedError

    def exists( self ):
        """
        exists (abstract method)
        
        return -> exist:TRUE / not exist:FALSE
        This method may throw an exception.
        """
        raise NotImplementedError

    def setSimulator( self, aSimulator ):
        """
        setSimulator
        
        aSimulator    : the reference to Simulator to set
        
        return -> None
        This method may throw an exception.
        """
        self.theSimulator = aSimulator

    def getSimulator( self ):
        """
        getSimulator
        
        return -> the reference to Simulator
        This method may throw an exception.
        """
        return self.theSimulator


