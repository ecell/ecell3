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

__program__ = 'ecell3-eml2em'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Koichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2004 Keio University'
__license__ = 'GPL'

__all__ = (
    'convert'
    )

import ecell.eml
import ecell.em.lexer as emlexer
import ecell.em.parser as emparser
from ecell.identifiers import *
from ecell.util import *

class EMLRenderer:
    def __init__( self, anEml ):
        self.theEml = anEml

    def nodeTypeToDomainClassName( aClass ):
        if isinstance( aClass, emparser.SystemNode ):
            return 'System'
        elif isinstance( aClass, emparser.ProcessNode ):
            return 'Process'
        elif isinstance( aClass, emparser.VariableNode ):
            return 'Variable'
        return None
    nodeTypeToDomainClassName = staticmethod( nodeTypeToDomainClassName )

    def convertToList( anListNode ):
        if not isinstance( anListNode, emparser.ListNode ):
            return anListNode
        return map( EMLRenderer.convertToList, anListNode )
    convertToList = staticmethod( convertToList )

    def renderEntityNode( self, aContainingSystemId, anAst ):
        assert isinstance( anAst, emparser.EntityNode )
        aKind = self.nodeTypeToDomainClassName( anAst )
        aClassName = anAst.identifier[ 0 ]
        aName = anAst.identifier[ 1 ]
        aFullID = FullID( aKind,  FullID( aContainingSystemId ).toSystemPath(),
                aName )
        self.theEml.createEntity( aClassName, aFullID )
        if len( anAst.identifier ) == 3:
            self.theEml.setEntityInfo( aFullID, anAst.identifier[ 2 ] )
        for aListElemNode in anAst[ 1 ]:
            self.theEml.setEntityProperty(
                aFullID, aListElemNode.name,
                self.convertToList( aListElemNode.values ) )

    def renderSystemNode( self, anAst ):
        assert isinstance( anAst, emparser.SystemNode )
        aFullID = SystemPath( anAst.identifier[ 1 ] ).toFullID()
        self.theEml.createEntity( 'System', aFullID )
        if len( anAst.identifier ) == 3:
            self.theEml.setEntityInfo( aFullID, anAst.identifier[ 2 ] )
        for aListElemNode in anAst[ 1 ]:
            if isinstance( aListElemNode, emparser.PropertyNode ):
                self.theEml.setEntityProperty(
                    aFullID, aListElemNode.name,
                    self.convertToList( aListElemNode.values ) )
            elif isinstance( aListElemNode, emparser.EntityNode ):
                self.renderEntityNode( aFullID, aListElemNode )
                
    def renderStepperNode( self, anAst ):
        assert isinstance( anAst, emparser.StepperNode )
        anID = anAst.identifier[ 1 ]
        self.theEml.createStepper( anAst.identifier[ 0 ], anID )
        if len( anAst.identifier ) == 3:
            self.theEml.setStepperInfo( anID, anAst.identifier[ 2 ] )
        for aListElemNode in anAst[ 1 ]:
            if isinstance( aListElemNode, emparser.PropertyNode ):
                self.theEml.setStepperProperty(
                    anID, aListElemNode.name,
                    self.convertToList( aListElemNode.values ) )

    def render( self, anAst ):
        for aStmt in anAst:
            if isinstance( aStmt, emparser.StepperNode ):
                self.renderStepperNode( aStmt )
            elif isinstance( aStmt, emparser.SystemNode ):
                self.renderSystemNode( aStmt )
            elif isinstance( aStmt, emparser.PropertySetterNode ):
                aFullPN = createFullID( aStmt[ 0 ] )
                aPropertyName = aFullPN[ 3 ]
                aFullID = createFullIDString( renderFullPNToFullID( aFullPN ) )
                self.theEml.deleteEntityProperty( aFullID, aPropertyName )
                self.theEml.setEntityProperty( aFullID, aPropertyName, aStmt[ 1 ] )

def convert( str, anEml = None ):
    if anEml == None:
        anEml = ecell.eml.Eml()
    anAst = emparser.createParser().parse( str, lexer = emlexer.createLexer() )
    EMLRenderer( anEml ).render( anAst )

    return anEml

