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
    'createLexer'
    )

import os
import ply.lex as lex

lextabname = "emlextab"

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

# Delimeters
t_LPAREN   = r'\('
t_RPAREN   = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LBRACE   = r'\{'
t_RBRACE   = r'\}'
t_SEMI     = r';'

def t_Stepper( t ):
    r' Stepper[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_System( t ):
    r' System[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_Process( t ):
    r' Process[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_Variable( t ):
    r' Variable[\s|\t] '
    t.value = t.value[:-1]
    return t

def t_number( t ):
    r' [+-]?(\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)? '
        #try:
        #     t.value = int(t.value)    
        #except ValueError:
        #     print "Line %d: Number %s is too large!" % (t.lineno,t.value)
        #     t.value = 0
    #t.value = Token( 'number', t.value )
    return t

def t_fullid( t ):
    r'[a-zA-Z]*:[\w/\.]*:\w*'
    return t

def t_identifier( t ):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    return t

def t_systempath( t ):
    r'[a-zA-Z_/\.]+[\w/\.]*'
    return t

def t_quotedstrings( t ):
    r' """[^"]*""" | \'\'\'[^\']*\'\'\' '
    t.value = t.value[3:-3]
    return t

def t_quotedstring( t ):
    r' "(^"|.)*" | \'(^\'|.)*\' '
    t.value = t.value[1:-1]
    return t

def t_control( t ):
    r' \%line\s[^\n]*\n '
    seq = t.value.split()
    t.lineno = int(seq[1])
    t.lexer.filename = seq[2]

def t_comment( t ):
    r' \#[^\n]* '
    pass

def t_nl( t ):
    r'\n+'
    t.lineno += len( t.value )

def t_whitespace( t ):
    r'[ |\t]+'

def t_error( t ):
    """Error handling rule"""
    print "Illegal character '%s' at line %d in %s." % ( t.value[0], t.lineno , t.lexer.filename)
    t.skip(1)

def createLexer():
    cwd = os.getcwd()
    outputdir = os.path.abspath( os.path.dirname( __file__ )  )
    os.chdir( outputdir )
    l = lex.lex( lextab = lextabname, optimize = 1 )
    os.chdir( cwd )
    return l
