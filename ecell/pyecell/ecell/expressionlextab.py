# expressionlextab.py.  This file automatically created by PLY. Don't edit.
_lexre = '(?P<t_NUMBER> (\\d+(\\.\\d*)?|\\d*\\.\\d+)([eE][+-]?\\d+)? )|(?P<t_newline>\\n+)|(?P<t_NAME>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<t_PLUS>\\+)|(?P<t_DIVIDE>\\/)|(?P<t_LPAREN>\\()|(?P<t_TIMES>\\*)|(?P<t_MINUS>\\-)|(?P<t_RPAREN>\\))|(?P<t_COLON>,)|(?P<t_COMMA>.)'
_lextab = [
  None,
  ('t_NUMBER','NUMBER'),
  None,
  None,
  None,
  ('t_newline','newline'),
  (None,'NAME'),
  (None,'PLUS'),
  (None,'DIVIDE'),
  (None,'LPAREN'),
  (None,'TIMES'),
  (None,'MINUS'),
  (None,'RPAREN'),
  (None,'COLON'),
  (None,'COMMA'),
]
_lextokens = {'RPAREN': None, 'NAME': None, 'NUMBER': None, 'TIMES': None, 'PLUS': None, 'LPAREN': None, 'COLON': None, 'COMMA': None, 'MINUS': None, 'DIVIDE': None}
_lexignore = ' \t'
_lexerrorf = 't_error'
