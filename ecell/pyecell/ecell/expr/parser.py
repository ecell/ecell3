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
    'createParser',
    'AddOpNode',
    'BinaryOpNode',
    'DerefOpNode',
    'DivOpNode',
    'FunctionCallNode',
    'IdentifierNode',
    'ListNode',
    'MethodNode',
    'MulOpNode',
    'NegOpNode',
    'Node',
    'PowOpNode',
    'ScalarNode',
    'SubOpNode',
    'UnaryOpNode',
    'VarRefNode'
    )

import os
import string
import types

import ply.yacc as yacc
import ecell.expr.lexer as lex

yacctabname = "expressionparsetab"

tokens = lex.tokens

# Parsing rules

precedence = (
    ('left','DOT'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    ('left','POWER'),
    ('right','UMINUS'),
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

class BinaryOpNode( Node ):
    pass

class AddOpNode( BinaryOpNode ):
    pass

class SubOpNode( BinaryOpNode ):
    pass

class MulOpNode( BinaryOpNode ):
    pass

class DivOpNode( BinaryOpNode ):
    pass

class PowOpNode( BinaryOpNode ):
    pass

class DerefOpNode( BinaryOpNode ):
    pass

class UnaryOpNode( Node ):
    pass

class NegOpNode( UnaryOpNode ):
    pass

class ListNode( Node ):
    pass

class VarRefNode( Node ):
    pass

class IdentifierNode( Node ):
    pass

class FunctionCallNode( Node ):
    pass

class MethodNode( BinaryOpNode ):
    pass 

class ScalarNode( Node ):
    pass

def p_statement_expr( p ):
    'statement : expression'
    p[ 0 ] = p[ 1 ]

def p_expression_bin_op( p ):
    '''expression : add_op
                  | sub_op
                  | mul_op
                  | div_op
                  | pow_op
    '''
    p[ 0 ] = p[ 1 ]

def p_expression_add_op( p ):
    'add_op : expression PLUS expression'
    p[ 0 ] = AddOpNode( p[ 1 ], p[ 3 ] )

def p_expression_sub_op( p ):
    'sub_op : expression MINUS expression'
    p[ 0 ] = SubOpNode( p[ 1 ], p[ 3 ] )

def p_expression_mul_op( p ):
    'mul_op : expression TIMES expression'
    p[ 0 ] = MulOpNode( p[ 1 ], p[ 3 ] )

def p_expression_div_op( p ):
    'div_op : expression DIVIDE expression'
    p[ 0 ] = DivOpNode( p[ 1 ], p[ 3 ] )

def p_expression_pow_op( p ):
    'pow_op : expression POWER expression'
    p[ 0 ] = PowOpNode( p[ 1 ], p[ 3 ] )

def p_expression_uminus( p ):
    'expression : MINUS expression %prec UMINUS'
    p[ 0 ] = NegOpNode( p[ 2 ] )

def p_expression_factor( p ):
    '''expression : expr_component'''
    p[ 0 ] = p[ 1 ]

def p_expression_group( p ):
    'group : LPAREN expression RPAREN'
    p[ 0 ] = p[ 2 ]

def p_expression_component( p ):
    '''expr_component : group
                      | method
                      | var_deref
                      | func
                      | scalar
    '''
    p[ 0 ] = p[ 1 ]

def p_expression_scalar( p ):
    '''scalar : number
              | var_ref
    '''
    p[ 0 ] = p[ 1 ]

def p_scalar_number( p ):
    'number : NUMBER'
    p[ 0 ] = ScalarNode( p[ 1 ] )

def p_scalar_var_ref( p ):
    'var_ref : NAME'
    p[ 0 ] = VarRefNode( p[ 1 ] )

def p_expression_arguments( p ):
    '''arguments : arguments COMMA expression
                 | expression
                 | empty'''
    if len( p ) >= 3:
        p[ 1 ].addChild( p[ 3 ] )
        p[ 0 ] = p[ 1 ]
    else:
        p[ 0 ] = ListNode()
        if p[ 1 ] != None:
            p[ 0 ].addChild( p[ 1 ] )

def p_expression_method( p ):
    'method : expr_component DOT func'
    p[ 0 ] = MethodNode( p[ 1 ], p[ 3 ] )

def p_expression_func( p ):
    'func : NAME LPAREN arguments RPAREN'
    p[ 0 ] = FunctionCallNode( IdentifierNode( p[ 1 ] ), p[ 3 ] )

def p_expression_var_deref( p ):
    'var_deref : expr_component DOT NAME'
    p[ 0 ] = DerefOpNode( p[ 1 ], IdentifierNode( p[ 3 ] ) )

def p_empty( p ):
    '''
    empty :
    '''
    p[ 0 ] = None

def p_error(t):
    print "Syntax error at '%s'" % t.value

def createParser():
    outputdir = os.path.abspath( os.path.dirname( __file__ )  )
    y = yacc.yacc( tabmodule = yacctabname, outputdir = outputdir, debug = 0 )
    return y

