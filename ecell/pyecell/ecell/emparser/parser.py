#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

import sys
import os
import tempfile

from ecell.eml import convertSystemID2SystemFullID, Eml
from ecell.ecssupport import *

import ply.lex as lex
import ply.yacc as yacc

"""
A program for converting .em file to EML.
This program is part of E-Cell Simulation Environment Version 3.
"""

__program__ = 'emparser'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Koichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2003 Keio University'
__license__ = 'GPL'

LEXTAB = "ecell.emparser.lextab"
PARSERTAB = "ecell.emparser.parsetab"

# List of token names.
tokens = [
    'Stepper',
    'System',
    'Variable',
    'Process',
    'number',
    'identifier',
    'fullid',
    'systempath',
    'quotedstring',
    'quotedstrings',
    'LPAREN', 'RPAREN',
    'LBRACKET', 'RBRACKET',
    'LBRACE', 'RBRACE',
    'SEMI',
    ]

filename = ''

# Delimeters
t_LPAREN   = r'\('
t_RPAREN   = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LBRACE   = r'\{'
t_RBRACE   = r'\}'
t_SEMI     = r';'


def t_Stepper(t):
    r' Stepper[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_System(t):
    r' System[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_Process(t):
    r' Process[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_Variable(t):
    r' Variable[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_number(t):
    r' [+-]?(\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)? '
    return t

def t_fullid(t):
    r'[a-zA-Z]*:[\w/\.]*:\w*'
    return t

def t_identifier(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    return t

def t_systempath(t):
    r'[a-zA-Z_/\.]+[\w/\.]*'
    return t

def t_quotedstrings(t):
    r' """[^"]*""" | \'\'\'[^\']*\'\'\' '
    t.value = t.value[3:-3]
    return t

def t_quotedstring(t):
    r' "(^"|.)*" | \'(^\'|.)*\' '
    t.value = t.value[1:-1]
    return t

def t_control(t):
    r' \%line\s[^\n]*\n '
    seq = t.value.split()
    t.lineno = int(seq[1])
    t.lexer.filename = seq[2]

def t_comment(t):
    r' \#[^\n]* '
    pass

def t_nl(t):
    r' \n+ '
    t.lineno += len( t.value )

def t_whitespace(t):
    r' [ |\t]+ '
    pass

# Error handling rule
def t_error(t):
    print "Illegal character '%s' at line %d in %s." % ( t.value[0], t.lineno , t.lexer.filename)
    t.skip(1)

# Parsing rules

# may be wrong..
precedence = (
    ( 'right', 'Variable', 'Process', 'System', 'Stepper' ),
    ( 'left', 'identifier' )
    )

def createListleft( t ):
    if hasattr(t, 'slice'):
        length = len(t.slice) - 1
    else:
        return [t]
    
    if length == 2:
        aList = t[1]
        aList.append(t[2])
        return aList

    elif t[1] == None:
        return []
    else:
        return [t[1]]

class StepperStmt(object):
    def __init__(self, classname, id, properties):
        self.classname = classname
        self.id = id
        self.properties = properties

class SystemStmt(object):
    def __init__(self, classname, id, properties):
        self.classname = classname
        self.id = id
        self.properties = properties

class EntityStmt(object):  
    def __init__(self, type, classname, id, properties):
        self.type = type
        self.classname = classname
        self.id = id
        self.properties = properties

class PropertyDef(object):
    def __init__(self, name, valuelist):
        self.name = name
        self.valuelist = valuelist

def p_stmts(t):
    '''
    stmts : stmts stmt
          | stmt
    '''
    t[0] = createListleft( t )

def p_stmt(t):
    '''
    stmt : stepper_stmt
         | system_stmt
    '''
    t[0] = t[1]

def p_stepper_stmt(t):
    '''
    stepper_stmt : stepper_decl LBRACE propertylist RBRACE
    '''
    t[0] = StepperStmt(t[1][0], t[1][1], t[3])
    
def p_system_stmt(t):
    '''
    system_stmt : system_decl LBRACE property_entity_list RBRACE
    '''
    t[0] = SystemStmt(t[1][0], t[1][1], t[3])

def p_entity_other_stmt (t):
    '''
    entity_other_stmt : entity_other_decl LBRACE propertylist RBRACE
    '''
    t[0] = EntityStmt(t[1][0], t[1][1], t[1][2], t[3])

# object declarations

def p_info(t):
    '''
    info : quotedstrings
         | quotedstring
    '''
    t[0] = t[1]

def p_stepper_decl(t):
    '''
    stepper_decl : Stepper name LPAREN name RPAREN 
                 | Stepper name LPAREN name RPAREN info
    '''
    t[0] = t[2], t[4]

def p_system_decl(t):
    '''
    system_decl : System name LPAREN systempath RPAREN 
                | System name LPAREN systempath RPAREN info

    '''
    t[0] = t[2], t[4]
   
def p_variable_or_process(t):
    '''
    variable_or_process : Variable
                        | Process
    '''
    t[0] = t[1]

def p_entity_other_decl(t):
    '''
    entity_other_decl : variable_or_process name LPAREN name RPAREN
                      | variable_or_process name LPAREN name RPAREN info
    '''
    t[0] = t[1], t[2], t[4]

# property

def p_propertylist(t):
    '''
    propertylist : propertylist property
                 | property
                 | empty
    '''
    t[0] = createListleft( t )

def p_property(t):
    '''
    property : name valuelist SEMI
    '''
    t[0] = PropertyDef(t[1], t[2])

# property or entity ( for System statement )

def p_property_entity_list(t):
    '''
    property_entity_list : property_entity_list property_entity
                         | property_entity
                         | empty
    '''
    t[0] = createListleft( t )

def p_property_entity(t):
    '''
    property_entity : property
                    | entity_other_stmt
    '''
    t[0] = t[1]

# value
def p_value(t):
    '''
    value : quotedstring
          | number
          | string
          | LBRACKET valuelist RBRACKET
          | quotedstrings
    '''
    if t[1] == '[':
        t[0] = t[2]
    else:
        t[0] = t[1]

def p_valuelist(t):
    '''
    valuelist : valuelist value
              | value
    '''
    t[0] = createListleft( t )
    

def p_string(t):
    '''
    string : name
           | fullid
           | systempath
    '''
    t[0] = t[1]

def p_name(t):
    '''
    name : identifier
         | Variable
         | Process
         | System
         | Stepper
    '''
    t[0] = t[1]

def p_empty(t):
    '''
    empty :
    '''    
    t[0] = None

def p_error(t):
    if t is None:
        print "Syntax error"
    else:
        print "Syntax error at line %d in %s. " % ( t.lineno, t.value )

def initializePLY(outputdir):
    lextabmod = LEXTAB.split('.')
    parsertabmod = PARSERTAB.split('.')

    lextabOutputDir = os.path.join( outputdir, *lextabmod[:-1] )
    try:
        os.makedirs( lextabOutputDir )
    except:
        pass
    lex.lex( lextab=lextabmod[-1], optimize=1, outputdir=lextabOutputDir )

    parsertabOutputDir = os.path.join( outputdir, *parsertabmod[:-1] )
    try:
        os.makedirs( parsertabOutputDir )
    except:
        pass
    yacc.yacc( tabmodule=parsertabmod[-1], outputdir=parsertabOutputDir )

def convertEm2Eml( anEmFileObject, debug=0 ):
    # initialize eml object
    anEml = Eml()
    patchEm2Eml( anEml, anEmFileObject, debug=debug)
    return anEml

def patchEm2Eml( anEml, anEmFileObject, debug=0 ):
    # Build the lexer
    aLexer = lex.lex(lextab=LEXTAB)
    aLexer.filename = 'undefined'
    # Parsing
    aParser = yacc.yacc(optimize=0, tabmodule=PARSERTAB)
    aParser.anEml = anEml
    anAst = aParser.parse( anEmFileObject.read(), lexer=aLexer, debug=debug )

    for aNode in anAst:
        if isinstance( aNode, StepperStmt ):
            anEml.createStepper( aNode.classname, aNode.id )
            for aProperty in aNode.properties:
                assert isinstance( aProperty, PropertyDef )
                anEml.setStepperProperty( aNode.id, aProperty.name, aProperty.valuelist )
        elif isinstance( aNode, SystemStmt ):
            anId = convertSystemID2SystemFullID( aNode.id )
            anEml.createEntity( aNode.classname, anId )
            for aPropertyOrEntity in aNode.properties:
                if isinstance( aPropertyOrEntity, EntityStmt ):
                    anEntityId = '%s:%s:%s' % ( aPropertyOrEntity.type, aNode.id, aPropertyOrEntity.id )
                    anEml.createEntity( aPropertyOrEntity.classname, anEntityId )
                    for aProperty in aPropertyOrEntity.properties:
                        assert isinstance( aProperty, PropertyDef )
                        anEml.setEntityProperty( anEntityId, aProperty.name, aProperty.valuelist )
                                
                elif isinstance( aPropertyOrEntity, PropertyDef ):
                    anEml.setEntityProperty( anId, aPropertyOrEntity.name, aPropertyOrEntity.valuelist )
        else:
            raise NotImplementedError

    return anEml
