#!/usr/bin/env ecell3

"""
A program for converting .em file to EML.
This program is part of E-Cell Simulation Environment Version 3.
"""

__program__ = 'emparser'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Kouichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2003 Keio University'
__license__ = 'GPL'


import sys
import os
import string 
import getopt
import tempfile

import ecell.eml
from ecell.spark import *

import lex
import yacc

class Token:
	def __init__(self, type, attr=None, filename="?" , lineno='???'):
                self.type = type
                self.attr = attr
		self.filename = filename
                self.lineno = lineno
		
	def __cmp__(self, o):
		return cmp(self.type, o)

        def __repr__(self):
                return str(self.type) + ':' + str(self.attr)
	
	def __getitem__(self,i):
		raise IndexError

	def __len__(self):
		return 0

	def info(self):
		return "File " + repr(self.filename) + ", line " + repr(self.lineno)
	def error(self):
		print "Error token", self, "(", self.info(), ")"
		raise SystemExit, 1

class AST:
	def __init__(self, type, kids=[]):
		self.type = type
		self._kids = kids
		
	def __getitem__(self, i):
		return self._kids[i]

	def __len__(self):
		return len(self._kids)

	def __repr__(self):
                return str(self.type) + ':' + str(self._kids)

# Reserved words
reserved = (
#   'Process', 'Variable', 'Stepper', 'System'
#    '(', ')', '{', '}', '[', ']'
)

# List of tlen names.
tokens = reserved + (
	'Stepper',
	'System',
	'Variable',
	'Process',
	'number',
	'name',
	'quotedstring',
	'control',
	'comment',
	'nl',
	'whitespace',
	# Delimeters ( ) [ ] { } ;
	'LPAREN', 'RPAREN',
	'LBRACKET', 'RBRACKET',
	'LBRACE', 'RBRACE',
	'SEMI',
	)

reserved_map = { }
for r in reserved:
	reserved_map['r'] = r

def t_Stepper(t):
	r' Stepper[\s|\t] '
	t.value = string.strip( t.value )
	t.value = Token( t.value, t.value )
	return t

def t_System(t):
	r' System[\s|\t] '
	t.value = string.strip( t.value )
	t.value = Token( t.value, t.value )
	return t

def t_Process(t):
	r' Process[\s|\t] '
	t.value = string.strip( t.value )
	t.value = Token( t.value, t.value )
	return t

def t_Variable(t):
	r' Variable[\s|\t] '
	t.value = string.strip( t.value )
	t.value = Token( t.value, t.value )
	return t

# Delimeters
def t_LPAREN(t):
	r'\('
	t.value = Token( t.value, t.value )
	return t

def t_RPAREN(t):
	r'\)'
	t.value = Token( t.value, t.value )
	return t

def t_LBRACKET(t):
	r'\['
	t.value = Token( t.value, t.value )
	return t

def t_RBRACKET(t):
	r'\]'
	t.value = Token( t.value, t.value )
	return t

def t_LBRACE(t):
	r'\{'
	t.value = Token( t.value, t.value )
	return t

def t_RBRACE(t):
	r'\}'
	t.value = Token( t.value, t.value )
	return t

def t_SEMI(t):
	r';'
	t.value = Token( t.value, t.value )
	return t

def t_number(t):
	r' [+-]?(\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)? '
        #try:
        #     t.value = int(t.value)    
        #except ValueError:
        #     print "Line %d: Number %s is too large!" % (t.lineno,t.value)
        #	 t.value = 0
	t.value = Token( 'number', t.value )
	return t

def t_name(t):
	r'[a-zA-Z_/][\w\:\/.]*'
	t.value = Token( 'name', t.value )
	return t

def t_quotedstring(t):
	r' "(^"|.)*" | \'(^\'|.)*\' '
	t.value = Token( 'quotedstring', t.value[1:-1] )
	return t

def t_control(t):
	r' \%line [^\n]*\n '
	seq = string.split(t.value)
	t.lineno = int( seq[1] )
	t.filename = str( seq[2] )

def t_comment(t):
	r' \# [^\n]* '
	pass

def t_nl(t):
	r' \n '
	t.lineno = t.lineno + 1

def t_whitespace(t):
	r' [ |\t]+ '
	pass

def t_default(t):
	r' .+ '
	raise ValueError, "Unexpected error: unmatched input: %s, line %d." % ('', t.lineno)

# Define a rule so we can track line numbers
#def t_newline(t):
#    r'\n+'
#    t.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
#t_ignore  = ' \t'

# Error handling rule
def t_error(t):
	print "Illegal character '%s' at line %d." % ( t.value[0], t.lineno )
	t.skip(1)

# Parsing rules

precedence = (
	('left', 'stmts', 'stmt' ),
	('right', 'property')
	)

def p_stmts(t):
	'''
        stmts : stmt stmts
              | stmt
        '''
	t[0] = createAst( 'stmts', t)


def p_stmt(t):
	'''
        stmt : stepper_stmt
             | system_stmt
        '''
	t[0] = createAst( 'stmt', t )
    
def p_stepper_stmt(t):
	'''
	stepper_stmt : Stepper object_decl LBRACE propertylist RBRACE
	'''
	t[0] = createAst( 'stepper_stmt', t )
    
def p_system_stmt(t):
	'''
	system_stmt : System object_decl LBRACE property_entity_list RBRACE
	'''
	t[0] = createAst( 'system_stmt', t )

def p_entity_other_stmt (t):
	'''
	entity_other_stmt : Variable object_decl LBRACE propertylist RBRACE
                          | Process object_decl LBRACE propertylist RBRACE
        '''
	t[0] = createAst ( 'entity_other_stmt', t )

# object declarations

def p_object_decl(t):
	'''
	object_decl : name LPAREN name RPAREN
	'''
	t[0] = createAst( 'object_decl', t )

# property

def p_propertylist(t):
    '''
    propertylist : property propertylist
                 | property
                 | empty
    '''
    t[0] = createAst( 'propertylist', t )

def p_property(t):
	'''
	property : name valuelist SEMI
	'''
	t[0] = createAst( 'property', t )

# property or entity ( for System statement )

def p_property_entity_list(t):
	'''
        property_entity_list : property_entity property_entity_list
                             | property_entity
                             | empty
        '''
	t[0] =  createAst( 'property_entity_list', t )


def p_property_entity(t):
	'''
	property_entity : property
	                | entity_other_stmt
        '''
	t[0] = createAst( 'property_entity', t )

# value

def p_value(t):
	'''
	value : LBRACKET valuelist RBRACKET
              | quotedstring
              | name
              | number
        '''
	t[0] =  createAst( 'value', t )
        
def p_valuelist(t):
        '''
        valuelist : value valuelist
                  | value
        '''
	t[0] =  createAst( 'valuelist', t )

def p_empty(t):
	'''
	empty :
	'''    
	t[0] = None

def p_error(t):
	print "Syntax error at line %d. (near '%s')" % ( t.lineno, t.value )


# Constract Ast tree
def createAst( type, t):

	length = len(t.slice) - 1

	# for multi entity
	if length != 1:
		aList = []
		i = 1
		while i <= length:
			aList.append( t[i] )
			i = i + 1
		return AST( type, aList )

	# for empty value
	elif t[1] == None:
		return AST( type, [] )

	else:
		return AST( type, [ t[1] ] )
	

def flatten_nodetree( node ):

	if node is None or len( node ) == 0:
		return []

        aList = list()
	while len( node ) >= 1:
		aList.append( node[0] )
		if len( node ) == 1:
			break
		else:
			node = node[1]

	return aList


def flatten_node( node ):
	
	if len( node ) == 1:	
		return node[0].attr 
	else:
		return flatten_propertytree( node[1] )


def flatten_propertytree( node ):

	return map( flatten_node, flatten_nodetree( node ) )
	


class Interpret(GenericASTTraversal):
	def __init__( self, ast, eml ):
		GenericASTTraversal.__init__( self, ast )
		self.eml = eml

		self.postorder()

	def n_stepper_stmt( self, node ):
		aClassname = node[1][0].attr
		anID = node[1][2].attr

		self.eml.createStepper( aClassname, anID )
		
		aPropertyNodeList = flatten_nodetree( node[3] )
		for i in aPropertyNodeList:

			aPropertyName = i[0].attr
			aValueList = flatten_propertytree( i[1] )

			self.eml.setStepperProperty( anID, aPropertyName,\
						     aValueList )

	def n_system_stmt( self, node ):
		aType = node[0].attr
		aClassname = node[1][0].attr
		anID = node[1][2].attr

		aFullID = convert2FullID( self.eml , aType, anID )
		self.eml.createEntity( aClassname, aFullID )
		
		aPropertyNodeList = flatten_nodetree( node[3] )
		for i in aPropertyNodeList:

                        n = i[0]
                        aPropertyName = n[0].attr

			if aPropertyName in ( 'Variable', 'Process' ):
				self.entity_other( n, anID )
				continue
			
                        aValueList = flatten_propertytree( n[1] )

			self.eml.setEntityProperty( aFullID, aPropertyName,\
						    aValueList )
		

        def entity_other( self, node, path ):
		aType = node[0].attr
		aClassname = node[1][0].attr
		anID = path + ':' + node[1][2].attr

		aFullID = convert2FullID( self.eml , aType, anID )
		self.eml.createEntity( aClassname, aFullID )
		
		aPropertyNodeList = flatten_nodetree( node[3] )
		for i in aPropertyNodeList:

			aPropertyName = i[0].attr
			aValueList = flatten_propertytree( i[1] )

			self.eml.setEntityProperty( aFullID, aPropertyName,\
                                                    aValueList )

            

	def default( self, node ):
		pass

def initializePLY():
	lextabname = "emlextab"
	yacctabname = "emparsetab"

	lex.lex(optimize=1, lextab=lextabname)
	yacc.yacc(tabmodule=yacctabname)

def convertEm2Eml( anEmFileObject, debug=0 ):

	# initialize eml object
	anEml = ecell.eml.Eml()
	
	# Build the lexer
	lex.lex(optimize=1, lextab="emlextab")

        # Tokenizen test..
        #while debug == 1:
			
            # Give the lexer some input for test
        #    lex.input(anEmFileObject.read())

        #    tok = lex.token( anEmFileObject.read() )
        #    if not tok: break      # No more input
        #    print tok

	# Parsing
	aParser = yacc.yacc(optimize=1, tabmodule="emparsetab")
	anAst = aParser.parse( anEmFileObject.read() , debug=debug)

	import pprint
	if debug != 0:
		print 'AST:', pprint.pprint(anAst)

	# Generating
	anInterpreter = Interpret( anAst, anEml )

	return anEml

def convert2FullID( anEml, aType, aSystemID ):

        if aType == 'System':
		#FIXME: convertSystemID2SystemFullID() will be deprecated
		return ecell.eml.convertSystemID2SystemFullID( aSystemID )
	elif aType == 'Variable':
		return 'Variable:' + aSystemID
	elif aType == 'Process':
		return 'Process:' + aSystemID

	# error
	raise ValueError, 'Type Error in conver2FullID() (%s)' % aType
    


#
# preprocessing methods
#

import StringIO

import ecell.em
em = ecell.em ; del ecell.em



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

	def atParseHook( self, interpreter, keywords ):
		if not self.need_linecontrol:
			return

		( file, line ) = interpreter.context().identify()
		self.lineControl( interpreter, file, line )
		self.need_linecontrol = 0

	def beforeIncludeHook( self, interpreter, keywords ):
		self.lineControl( interpreter, keywords['name'], 1 )

	def afterIncludeHook( self, interpreter, keywords ):
		( file, line ) = interpreter.context().identify()
		self.lineControl( interpreter, file, line )

	def preprocess( self ):

		#
		# init
		#
		Output = StringIO.StringIO()
		self.interpreter = em.Interpreter( output = Output )
		pseudo = self.interpreter.pseudo
		pseudo.flatten()

		#
		# set hooks
		#
		pseudo.addHook( 'after_include',     self.afterIncludeHook )
		pseudo.addHook( 'before_include',    self.beforeIncludeHook )
		pseudo.addHook( 'after_expand',      self.needLineControl )
		pseudo.addHook( 'after_evaluate',    self.needLineControl )
		pseudo.addHook( 'after_execute',     self.needLineControl )
		pseudo.addHook( 'after_substitute',  self.needLineControl )
		pseudo.addHook( 'after_significate', self.needLineControl )
		pseudo.addHook( 'at_parse',          self.atParseHook )


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




