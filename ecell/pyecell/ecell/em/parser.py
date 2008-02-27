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

"""
A program for converting .em file to EML.
This program is part of E-Cell Simulation Environment Version 3.
"""

__program__ = 'emparser'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Koichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2003 Keio University'
__license__ = 'GPL'

__all__ = (
    'createParser',
    'Node',
    'EntityNode',
    'SystemNode',
    'StepperNode',
    'IdentifierNode',
    'PropertyNode',
    'PropertySetterNode',
    'ListNode',
    )

import os
import ply.yacc as yacc
import ecell.em.lexer as lex

yacctabname = "emparsetab"

tokens = lex.tokens

# may be wrong..
precedence = (
    ( 'right', 'Variable', 'Process', 'System', 'Stepper' ),
    ( 'left', 'identifier' )
    )

class Node( object ):
    __attrs__ = None

    def __init__( self, *nodes ):
        object.__setattr__( self, 'children', list( nodes ) )

    def addChild( self, n ):
        self.children.append( n )

    def removeChild( self, n ):
        i = self.children.find( n )
        if i != None: del self.children[ i ]

    def __getitem__( self, idx ):
        return self.children[ idx ]

    def __setitem__( self, idx, node ):
        assert isinstance( node, Node )
        self.children[ idx ] = node

    def __len__( self ):
        return len( self.children )

    def __iter__( self ):
        return self.children.__iter__()

    def __getattr__( self, name ):
        if not self.__attrs__:
            raise KeyError()
        return self.children[ self.__attrs__[ name ] ]

    def __setattr__( self, name, val ):
        if not self.__attrs__:
            raise KeyError()
        self.children[ self.__attrs__[ name ] ] = val

class EntityNode( Node ):
    __attrs__ = {
        'identifier': 0,
        'properties': 1
    }

class SystemNode( EntityNode ):
    pass

class ProcessNode( EntityNode ):
    pass

class VariableNode( EntityNode ):
    pass

class StepperNode( Node ):
    __attrs__ = {
        'identifier': 0,
        'properties': 1
    }

class IdentifierNode( Node ):
    pass

class PropertyNode( Node ):
    __attrs__ = {
        'name': 0,
        'values': 1
    }

class PropertySetterNode( Node ):
    pass

class ListNode( Node ):
    pass

def p_stmts( p ):
    '''
    stmts : stmts stmt
          | stmt
    '''
    p[0] = buildList( p )

def p_stmt( p ):
    '''
    stmt : stepper_stmt
         | system_stmt
         | ecs
    '''

    p[ 0 ] = p[ 1 ]
    
def p_stepper_stmt( p ):
    '''
    stepper_stmt : Stepper object_decl LBRACE propertylist RBRACE
    '''
    p[ 0 ] = StepperNode( p[ 2 ], p[ 3 ] )
    
def p_system_stmt( p ):
    '''
    system_stmt : System system_object_decl LBRACE property_entity_list RBRACE
    '''
    p[ 0 ] = SystemNode( p[ 2 ], p[ 4 ] )

def p_entity_other_stmt( p ):
    '''
    entity_other_stmt : entity_kind object_decl LBRACE propertylist RBRACE
    '''
    if p[ 1 ] == 'Process':
        p[ 0 ] = ProcessNode( p[ 2 ], p[ 4 ] )
    elif p[ 1 ] == 'Variable':
        p[ 0 ] = VariableNode( p[ 2 ], p[ 4 ] )

def p_ecs( p ):
    '''
    ecs : fullid valuelist SEMI
    '''
    p[ 0 ] = PropertySetterNode( p[ 1 ], p[ 2 ] )

def p_system_object_decl( p ):
    '''
    system_object_decl : name LPAREN systempath RPAREN 
                       | name LPAREN systempath RPAREN info
    '''
    if len( p.slice ) == 6:
        p[ 0 ] = IdentifierNode( p[ 1 ], p[ 3 ], p[ 5 ] )
    else:
        p[ 0 ] = IdentifierNode( p[ 1 ], p[ 3 ] )

    p.containing_system_path = p[ 3 ]

def p_object_decl( p ):
    '''
    object_decl : name LPAREN name RPAREN 
                | name LPAREN name RPAREN info
    '''
    if len( p.slice ) == 6:
        p[ 0 ] = IdentifierNode( p[ 1 ], p[ 3 ], p[ 5 ] )
    else:
        p[ 0 ] = IdentifierNode( p[ 1 ], p[ 3 ] )

def p_info( p ):
    '''
    info : quotedstrings
         | quotedstring
    '''
    p[ 0 ] = p[ 1 ]

def p_entity_kind( p ):
    '''
    entity_kind : Variable
                | Process
    '''
    p[ 0 ] = p[ 1 ]

# property
def p_propertylist( p ):
    '''
    propertylist : propertylist property
                 | property
                 | empty
    '''
    p[ 0 ] = buildList( p )

def p_property( p ):
    '''
    property : name valuelist SEMI
    '''
    if type(p[ 2 ]) == str:
        p[ 2 ] = [ p[ 2 ] ]
    p[ 0 ] = PropertyNode( p[ 1 ], p[ 2 ] )

# property or entity ( for System statement )
def p_property_entity_list( p ):
    '''
    property_entity_list : property_entity_list property_entity
                         | property_entity
                         | empty
    '''
    p[ 0 ] =  buildList( p )

def p_property_entity( p ):
    '''
    property_entity : property
                    | entity_other_stmt
    '''
    p[ 0 ] = p[ 1 ]

# value
def p_value( p ):
    '''
    value : quotedstring
          | number
          | string
          | LBRACKET valuelist RBRACKET
          | quotedstrings
    '''

    if p[ 1 ] == '[':
        p[ 0 ] = p[ 2 ]
    else:
        p[ 0 ] = p[ 1 ]

def p_valuelist( p ):
    '''
    valuelist : valuelist value
              | value
    '''
    p[ 0 ] = buildList( p )

def p_string( p ):
    '''
    string : name
           | fullid
           | systempath
    '''
    p[ 0 ] = p[ 1 ]

def p_name( p ):
    '''
    name : identifier
         | Variable
         | Process
         | System
         | Stepper
    '''
    p[ 0 ] = p[ 1 ]

def p_empty( p ):
    '''
    empty :
    '''    
    p[ 0 ] = None

def p_error( p ):
    if p.errorReporter == None:
        print "Syntax error at line %d in %s. " % ( p.lineno, p.value )
    else:
        p.errorReporter( p )
    yacc.errok()
    
def buildList( p ):
    if len( p.slice ) - 1 == 2:
        aList = p[ 1 ]
        aList.addChild( p[ 2 ] )
        return aList
    elif p[ 1 ] == None:
        return ListNode()
    else:
        return ListNode( p[ 1 ] )

def createParser():
    outputdir = os.path.abspath( os.path.dirname( __file__ )  )
    y = yacc.yacc( tabmodule = yacctabname, outputdir = outputdir, debug = 0 )
    return y
