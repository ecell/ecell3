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

import lex
import yacc

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

filename = ''
reserved_map = { }
for r in reserved:
	reserved_map['r'] = r

def t_Stepper(t):
	r' Stepper[\s|\t] '
	t.value = string.strip( t.value )
	return t

def t_System(t):
	r' System[\s|\t] '
	t.value = string.strip( t.value )
	return t

def t_Process(t):
	r' Process[\s|\t] '
	t.value = string.strip( t.value )
	return t

def t_Variable(t):
	r' Variable[\s|\t] '
	t.value = string.strip( t.value )
	return t

# Delimeters
def t_LPAREN(t):
	r'\('
	return t

def t_RPAREN(t):
	r'\)'
	return t

def t_LBRACKET(t):
	r'\['
	return t

def t_RBRACKET(t):
	r'\]'
	return t

def t_LBRACE(t):
	r'\{'
	return t

def t_RBRACE(t):
	r'\}'
	return t

def t_SEMI(t):
	r';'
	return t

def t_number(t):
	r' [+-]?(\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)? '
        #try:
        #     t.value = int(t.value)    
        #except ValueError:
        #     print "Line %d: Number %s is too large!" % (t.lineno,t.value)
        #	 t.value = 0
	#t.value = Token( 'number', t.value )
	return t

def t_name(t):
	r'[a-zA-Z_/][\w\:\/.]*'
	return t

def t_quotedstring(t):
	r' "(^"|.)*" | \'(^\'|.)*\' '
	t.value = t.value[1:-1]
	return t

def t_control(t):
	r' \%line [^\n]*\n '
	seq = string.split(t.value)
	t.lineno = int(seq[1])
	t.lexer.filename = seq[2]

def t_comment(t):
	r' \# [^\n]* '
	pass

def t_nl(t):
	r' \n '
	t.lineno = t.lineno + 1

def t_whitespace(t):
	r' [ |\t]+ '
	pass

#def t_default(t):
#	r' .+ '
#	raise ValueError, "Unexpected error: unmatched input: %s, line %d." % (t.value, t.lineno)

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

precedence = (
	('left', 'stmts', 'stmt' ),
	('right', 'property')
	)

def p_stmts(t):
	'''
        stmts : stmt stmts

        '''
	t[0] = createList( 'stmts', t)


def p_stmt(t):
	'''
        stmt : stepper_stmt
             | system_stmt
        '''
	t[0] = createList( 'stmt', t )
    
def p_stepper_stmt(t):
	'''
	stepper_stmt : stepper_object_decl LBRACE propertylist RBRACE
	'''
	t[0] = t[1], t[2], t[4]
    
def p_system_stmt(t):
	'''
	system_stmt : system_object_decl LBRACE property_entity_list RBRACE
	'''
	t[0] = t[1], t[2], t[4]

def p_entity_other_stmt (t):
	'''
	entity_other_stmt : entity_other_object_decl LBRACE propertylist RBRACE
        '''
	t[4] = flatten_propertylist(t[4])
	t[0] = t[1], t[2], t[4]

# object declarations

def p_object_decl(t):
	'''
	object_decl : name LPAREN name RPAREN
	'''
	t[0] = t[1], t[3]
	
def p_stepper_object_decl(t):
	'''
	stepper_object_decl : Stepper object_decl
	'''
	t.type = t[1]
	t.classname = t[2][0]
	t.id = t[2][1]
	anEml.createStepper(t.classname, t.id)
	t[0] = t[1], t[2]
	
def p_system_object_decl(t):
	'''
	system_object_decl : System object_decl
	'''
	t.type = t[1]
	t.classname = t[2][0]
	t.path      = t[2][1]
	t.fullid = convert2FullID(t.type, t.path)
	anEml.createEntity(t.classname, t.fullid)
	
	t[0] = t[1], t[2]
	
def p_entity_other_object_decl (t):
	'''
	entity_other_object_decl : Variable object_decl
                                 | Process object_decl
        '''
	t.type = t[1]
	t.classname = t[2][0]
	t.id        = t.path + ':' + t[2][1]
	t.fullid = convert2FullID(t.type, t.id)
	anEml.createEntity(t.classname, t.fullid)
	
	t[0] = t[1], t[2]

# property

def p_propertylist(t):
    '''
    propertylist : property propertylist
                 | property
                 | empty
    '''

    t[0] = createList( 'propertylist', t )

def p_property(t):
	'''
	property : name value SEMI
	'''
	t[2] = flatten_nodelist(t[2])

	if t.type == 'Stepper':
		anEml.setStepperProperty(t.id, t[1], t[2])
	else:
		anEml.setEntityProperty(t.fullid, t[1], t[2])
		
	t[0] = t[1], t[2]

# property or entity ( for System statement )

def p_property_entity_list(t):
	'''
        property_entity_list : property_entity property_entity_list
                             | property_entity
                             | empty
        '''
	t[0] =  createList( 'property_entity_list', t )


def p_property_entity(t):
	'''
	property_entity : property
	                | entity_other_stmt
        '''
	t[0] = createList( 'property_entity', t )

# value

def p_value(t):
	'''
	value : quotedstring
              | name
              | number
	      | matrixlist
        '''
	t[0] =  createList( 'value', t )
        
def p_valuelist(t):
        '''
        valuelist : value valuelist
                  | value
        '''
	t[0] =  createList( 'valuelist', t )
	
def p_matrix(t):
	'''
	matrix : LBRACKET valuelist RBRACKET
        '''
	t[0] = flatten_nodetree(t[2])
	
def p_matrixlist(t):
        '''
        matrixlist : matrix matrixlist
                   | matrix
        '''
	t[0] =  createList( 'matrixlist', t )

def p_empty(t):
	'''
	empty :
	'''    
	t[0] = None

def p_error(t):
	print "Syntax error at line %d in %s. (near '%s')" % ( t.lineno, t.value )
	yacc.errok()
	
# Constract List
def createList( type, t):

	length = len(t.slice) - 1

	if length != 1:
		aList = []
		i = 1
		while i <= length:
			aList.append( t[i] )
			i = i + 1
		return aList

	elif t[1] == None:
		return []

	else:
		return t[1]

def flatten_propertylist( node ):

	if node is None or len( node ) == 0:
		return []

        aList = list()
	while len( node ) >= 1:

		if len( node ) == 1 and type(node) != str:
			break
		elif type(node[0]) == str:
			aList.append( node )
			break
		
		else:
			aList.append( node[0] )
			node = node[1]
			
	return aList
	
def flatten_nodelist( node ):

	if node is None or len( node ) == 0:
		return []

        aList = list()
	while len( node ) >= 1:
		if len( node ) == 1 and type(node) != str:
			break
		elif type(node[0]) == str:
			aList.append( node )
			break
		
		else:
			aList.append( node[0] )
			node = node[1]

	return aList

def flatten_nodetree( node ):

	if node is None or len( node ) == 0:
		return []

        aList = list()
	while len( node ) >= 1:
		if type(node) == str:
			aList.append( node )
			break
		
		elif len( node ) == 1 :
			break

		else:
			aList.append( node[0] )
			node = node[1]
			
	return aList


def flatten_node( node ):
	
	if len( node ) == 1:	
		#return node[0].attr
		return node[0]
	else:
		return flatten_propertytree( node[1] )


def flatten_propertytree( node ):

	return map( flatten_node, flatten_nodetree( node ) )
	



def initializePLY():
	lextabname = "emlextab"
	yacctabname = "emparsetab"

	lex.lex(lextab=lextabname)
	yacc.yacc(tabmodule=yacctabname)

def convertEm2Eml( anEmFileObject, debug=0 ):

	# initialize eml object
	global anEml
	anEml = ecell.eml.Eml()
	
	# Build the lexer
	aLexer = lex.lex(lextab="emlextab")

        # Tokenizen test..
        #while debug == 1:
			
            # Give the lexer some input for test
        #    lex.input(anEmFileObject.read())

        #   tok = aLexer.token( anEmFileObject.read() )
        #    if not tok: break      # No more input
        #    print tok

	# Parsing
	aParser = yacc.yacc(optimize=1, tabmodule="emparsetab")
	anAst = aParser.parse( anEmFileObject.read() ,lexer=aLexer ,debug=debug)
	
		
	import pprint
	if debug != 0:
		print pprint.pprint(anAst)
		
	if anAst == None:
		sys.exit(0)
	
	return anEml


            
		
def convert2FullID( aType, aSystemID ):

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


if __name__ == '__main__':
	initializePLY()

