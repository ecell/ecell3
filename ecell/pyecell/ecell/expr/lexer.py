#!/usr/bin/env python
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
# -----------------------------------------------------------------------------
#  expressionparser.py
#
#  An expression parser for SBML Exporter
#  This program is part of E-Cell Simulation Environment Version 3.
#
#  Author : Tatsuya Ishida
# -----------------------------------------------------------------------------

__program__ = 'expressionparser'
__author__ = 'Tatsuya Ishida'
__copyright__ = 'Copyright (C) 2002-2004 Keio University'
__license__ = 'GPL'

__all__ = (
    'createLexer',
    'tokens'
    )


import os
import string
import types

import ply.lex as lex

lextabname = "expressionlextab"

tokens = (
    'DOT','COMMA','NAME','NUMBER',
    'PLUS','MINUS','TIMES','DIVIDE','POWER',
    'LPAREN','RPAREN',
    )


# Tokens

t_PLUS    = r'\+'
t_MINUS   = r'\-'
t_TIMES   = r'\*'
t_DIVIDE  = r'\/'
t_POWER   = r'\^'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_NAME    = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_COMMA   = r','
t_DOT     = r'.'


def t_NUMBER(t):
    r' (\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)? '
        
    return t

# Ignored characters
t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lineno += t.value.count("\n")
    
def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.skip(1)
    
def createLexer():
    cwd = os.getcwd()
    outputdir = os.path.abspath( os.path.dirname( __file__ )  )
    os.chdir( outputdir )
    l = lex.lex( lextab = lextabname, optimize = 1 )
    os.chdir( cwd )
    return l

