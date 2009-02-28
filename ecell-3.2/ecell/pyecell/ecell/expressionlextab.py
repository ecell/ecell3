#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
# expressionlextab.py.  This file automatically created by PLY. Don't edit.
_lexre = '(?P<t_NUMBER> (\\d+(\\.\\d*)?|\\d*\\.\\d+)([eE][+-]?\\d+)? )|(?P<t_newline>\\n+)|(?P<t_NAME>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<t_PLUS>\\+)|(?P<t_POWER>\\^)|(?P<t_DIVIDE>\\/)|(?P<t_LPAREN>\\()|(?P<t_TIMES>\\*)|(?P<t_MINUS>\\-)|(?P<t_RPAREN>\\))|(?P<t_COLON>,)|(?P<t_COMMA>.)'
_lextab = [
  None,
  ('t_NUMBER','NUMBER'),
  None,
  None,
  None,
  ('t_newline','newline'),
  (None,'NAME'),
  (None,'PLUS'),
  (None,'POWER'),
  (None,'DIVIDE'),
  (None,'LPAREN'),
  (None,'TIMES'),
  (None,'MINUS'),
  (None,'RPAREN'),
  (None,'COLON'),
  (None,'COMMA'),
]
_lextokens = {'POWERLPAREN': None, 'RPAREN': None, 'NAME': None, 'NUMBER': None, 'TIMES': None, 'PLUS': None, 'COLON': None, 'COMMA': None, 'MINUS': None, 'DIVIDE': None}
_lexignore = ' \t'
_lexerrorf = 't_error'
