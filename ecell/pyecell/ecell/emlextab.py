# emlextab.py.  This file automatically created by PLY. Don't edit.
_lexre = '(?P<t_Stepper> Stepper[\\s|\\t] )|(?P<t_System> System[\\s|\\t] )|(?P<t_Process> Process[\\s|\\t] )|(?P<t_Variable> Variable[\\s|\\t] )|(?P<t_LPAREN>\\()|(?P<t_RPAREN>\\))|(?P<t_LBRACKET>\\[)|(?P<t_RBRACKET>\\])|(?P<t_LBRACE>\\{)|(?P<t_RBRACE>\\})|(?P<t_SEMI>;)|(?P<t_number> [+-]?(\\d+(\\.\\d*)?|\\d*\\.\\d+)([eE][+-]?\\d+)? )|(?P<t_name>[a-zA-Z_/][\\w\\:\\/.]*)|(?P<t_quotedstrings> """[^"]*""" | \\\'\\\'\\\'[^\\\']*\\\'\\\'\\\' )|(?P<t_quotedstring> "(^"|.)*" | \\\'(^\\\'|.)*\\\' )|(?P<t_control> \\%line [^\\n]*\\n )|(?P<t_comment> \\# [^\\n]* )|(?P<t_nl> \\n )|(?P<t_whitespace> [ |\\t]+ )'
_lextab = [
  None,
  ('t_Stepper','Stepper'),
  ('t_System','System'),
  ('t_Process','Process'),
  ('t_Variable','Variable'),
  ('t_LPAREN','LPAREN'),
  ('t_RPAREN','RPAREN'),
  ('t_LBRACKET','LBRACKET'),
  ('t_RBRACKET','RBRACKET'),
  ('t_LBRACE','LBRACE'),
  ('t_RBRACE','RBRACE'),
  ('t_SEMI','SEMI'),
  ('t_number','number'),
  None,
  None,
  None,
  ('t_name','name'),
  ('t_quotedstrings','quotedstrings'),
  ('t_quotedstring','quotedstring'),
  None,
  None,
  ('t_control','control'),
  ('t_comment','comment'),
  ('t_nl','nl'),
  ('t_whitespace','whitespace'),
]
_lextokens = {'control': None, 'comment': None, 'LBRACE': None, 'nl': None, 'name': None, 'SEMI': None, 'Process': None, 'RPAREN': None, 'Stepper': None, 'System': None, 'RBRACE': None, 'quotedstring': None, 'LBRACKET': None, 'LPAREN': None, 'quotedstrings': None, 'Variable': None, 'number': None, 'RBRACKET': None, 'whitespace': None}
_lexignore = None
_lexerrorf = 't_error'
