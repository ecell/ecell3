#!/usr/bin/env ecell3

"""
A program for converting .em file to EML.
This program is part of E-Cell Simulation Environment Version 3.
"""

__program__ = 'emparser'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Koichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2003 Keio University'
__license__ = 'GPL'


import sys
import os
import string 
import getopt
import tempfile

import ecell.eml
from ecell.ecssupport import *

import lex
import yacc

# Reserved words
reserved = (
#   'Process', 'Variable', 'Stepper', 'System'
#    '(', ')', '{', '}', '[', ']'
)

# List of token names.
tokens = reserved + (
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
    # Delimeters ( ) [ ] { } ;
    'LPAREN', 'RPAREN',
    'LBRACKET', 'RBRACKET',
    'LBRACE', 'RBRACE',
    'SEMI',
    )

filename = ''
reserved_map = { }
for r in reserved:
    reserved_map['r'] = r

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
        #try:
        #     t.value = int(t.value)    
        #except ValueError:
        #     print "Line %d: Number %s is too large!" % (t.lineno,t.value)
        #     t.value = 0
    #t.value = Token( 'number', t.value )
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

#def t_default(t):
#    r' .+ '
#    raise ValueError, "Unexpected error: unmatched input: %s, line %d." % (t.value, t.lineno)

# Define a rule so we can track line numbers
#def t_newline(t):
#    r'\n+'
#    t.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
#t_ignore  = ' \t'

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
         | ecs
        '''

    t[0] = t[1]
    
def p_stepper_stmt(t):
    '''
    stepper_stmt : stepper_decl LBRACE propertylist RBRACE
    '''
    t[0] = t[1], t[3]
    
def p_system_stmt(t):
    '''
    system_stmt : system_decl LBRACE property_entity_list RBRACE
    '''
    t[0] = t[1], t[3]

def p_entity_other_stmt (t):
    '''
    entity_other_stmt : entity_other_decl LBRACE propertylist RBRACE
        '''
    t[0] = t[1], t[3]

# ecs support
def p_ecs(t):
    '''
    ecs : fullid valuelist SEMI
    '''
    aFullPN = createFullPN( t[1] )
    aPropertyName = aFullPN[3]
    aFullID = createFullIDString( convertFullPNToFullID( aFullPN ) )

    # for update property
    if anEml.getEntityProperty( t[1] ):
        anEml.deleteEntityProperty( aFullID, aPropertyName )
        
    anEml.setEntityProperty(aFullID, aPropertyName, t[2])

    t[0] = t[1], t[2]

# object declarations

def p_object_decl(t):
    '''
    object_decl : name LPAREN name RPAREN 
                | name LPAREN name RPAREN info
    '''
    if len(t.slice) == 6:
        t[0] = t[1], t[3], t[5]
    else:
        t[0] = t[1], t[3]

def p_system_object_decl(t):
    '''
    system_object_decl : name LPAREN systempath RPAREN 
                       | name LPAREN systempath RPAREN info
    '''
    if len(t.slice) == 6:
        t[0] = t[1], t[3], t[5]
    else:
        t[0] = t[1], t[3]

def p_info(t):
    '''
    info : quotedstrings
         | quotedstring
    '''
    t[0] = t[1]

def p_stepper_decl(t):
    '''
    stepper_decl : Stepper object_decl
    '''
    t.type = t[1]
    t.classname = t[2][0]
    t.id = t[2][1]
    anEml.createStepper(t.classname, t.id)
    if len(t[2]) == 3:
        anEml.setStepperInfo( t.id, t[2][2])
    
    t[0] = t[1], t[2]
    
def p_system_decl(t):
    '''
    system_decl : System system_object_decl
    '''
    t.type = t[1]
    t.classname = t[2][0]
    t.path      = t[2][1]
    t.id = ecell.eml.convertSystemID2SystemFullID( t.path )
    anEml.createEntity(t.classname, t.id)
    if len(t[2]) == 3:
        anEml.setEntityInfo( t.id, t[2][2] )
    
    t[0] = t[1], t[2]
    
def p_entity_other_decl (t):
    '''
    entity_other_decl : Variable object_decl
                          | Process object_decl
        '''
    t.type = t[1]
    t.classname = t[2][0]
    t.id        = t.path + ':' + t[2][1]
    t.id = t.type + ':' + t.id
    anEml.createEntity( t.classname, t.id )
    if len(t[2]) == 3:
        anEml.setEntityInfo( t.id, t[2][2] )
    
    t[0] = t[1], t[2]

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
    if type(t[2]) == str:
        t[2] = [t[2]]

    if t.type == 'Stepper':
        anEml.setStepperProperty(t.id, t[1], t[2])
    else:
        anEml.setEntityProperty(t.id, t[1], t[2])
        
    #t[0] = t[1], t[2]

# property or entity ( for System statement )

def p_property_entity_list(t):
    '''
        property_entity_list : property_entity_list property_entity
                             | property_entity
                             | empty
        '''
    t[0] =  createListleft( t )


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
    print "Syntax error at line %d in %s. " % ( t.lineno, t.value )
    yacc.errok()
    
# Constract List
    
def createListleft( t ):

    if hasattr(t, 'slice'):
        length = len(t.slice) - 1
    else:
        return [t]

    
    if length == 2:
        aList = t[1]
            
        aList.append( t[2] )
        return aList

    elif t[1] == None:
        return []

    else:
        return [t[1]]


def initializePLY():
    lextabname = "emlextab"
    yacctabname = "emparsetab"

    lex.lex(lextab=lextabname, optimize=1)
    #lex.lex(lextab=lextabname)
    yacc.yacc(tabmodule=yacctabname)

def convertEm2Eml( anEmFileObject, debug=0 ):

    # initialize eml object
    anEml = ecell.eml.Eml()
    patchEm2Eml( anEml, anEmFileObject, debug=debug)

    return anEml

def patchEm2Eml( anEmlObject, anEmFileObject, debug=0 ):

    # initialize eml object
    global anEml
    anEml = anEmlObject
    
    # Build the lexer
    aLexer = lex.lex(lextab="emlextab")
    aLexer.filename = 'undefined'
        # Tokenizen test..
        #while debug == 1:
            
            # Give the lexer some input for test
        #    lex.input(anEmFileObject.read())

        #   tok = aLexer.token( anEmFileObject.read() )
        #    if not tok: break      # No more input
        #    print tok

    # Parsing
    aParser = yacc.yacc(optimize=1, tabmodule="emparsetab")
    anAst = aParser.parse( anEmFileObject.read(), lexer=aLexer ,debug=debug )
        
    import pprint
    if debug != 0:
        print pprint.pprint(anAst)
        
    if anAst == None:
        sys.exit(1)
    
    return anEml

#
# preprocessing methods
#

import StringIO

import ecell.em
em = ecell.em ; del ecell.em

class ecellHookClass(em.Hook):
    def __init__(self, aPreprocessor, anInterpreter):
        self.theInterpreter = anInterpreter
        self.thePreprocessor = aPreprocessor
    def afterInclude( self ):          # def afterInclude( self, interpreter, keywords ):
        ( file, line ) = self.interpreter.context().identify()
        self.thePreprocessor.lineControl( self.theInterpreter, file, line )

    def beforeIncludeHook( self, name, file, locals ):  
        self.thePreprocessor.lineControl( self.theInterpreter, name, 1 )  

    def afterExpand( self, result ):
        self.thePreprocessor.need_linecontrol = 1

    def afterEvaluate(self, result):
        self.thePreprocessor.need_linecontrol = 1
        return

    def afterSignificate(self):
        self.thePreprocessor.need_linecontrol = 1
        return
                         
    def atParse(self, scanner, locals):
        if not self.thePreprocessor.need_linecontrol:
            return

        ( file, line ) = self.theInterpreter.context().identify()
        self.thePreprocessor.lineControl( self.theInterpreter, file, line )
        self.thePreprocessor.need_linecontrol = 0        



class Preprocessor:

    def __init__( self, file, filename ):
        self.need_linecontrol = 0
        self.file = file
        self.filename = filename
        self.interpreter = None

    def __del__( self ):
        self.shutdown()

    def lineControl( self, interpreter, file, line ):
        interpreter.write( '%%line %d %s\n' % ( line, file ) )

    def needLineControl( self, *args ):
        self.need_linecontrol = 1

    def preprocess( self ):

        #
        # init
        #
        Output = StringIO.StringIO()
        self.interpreter = em.Interpreter( output = Output )
        self.interpreter.flatten()
        self.interpreter.addHook(ecellHookClass(self, self.interpreter))   # pseudo.addHook(ecellHookClass(self, self.interpreter))

        #
        # processing
        #

        # write first line
        self.lineControl( self.interpreter, self.filename, 1 )

        if self.file is not None:
            self.interpreter.wrap( self.interpreter.file,\
                      ( self.file, self.filename ) )

        self.interpreter.flush()

        return Output


    def shutdown( self ):
        
        self.interpreter.shutdown()


if __name__ == '__main__':
    initializePLY()

