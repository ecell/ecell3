# emlextab.py.  This file automatically created by PLY. Don't edit.
_lexre = '(?P<t_Stepper> Stepper[\\s|\\t] )|(?P<t_System> System[\\s|\\t] )|(?P<t_Process> Process[\\s|\\t] )|(?P<t_Variable> Variable[\\s|\\t] )|(?P<t_number> [+-]?(\\d+(\\.\\d*)?|\\d*\\.\\d+)([eE][+-]?\\d+)? )|(?P<t_fullid>[a-zA-Z]*:[\\w/\\.]*:\\w*)|(?P<t_identifier>[a-zA-Z_][a-zA-Z0-9_]*)|(?P<t_systempath>[a-zA-Z_/\\.]+[\\w/\\.]*)|(?P<t_quotedstrings> """[^"]*""" | \\\'\\\'\\\'[^\\\']*\\\'\\\'\\\' )|(?P<t_quotedstring> "(^"|.)*" | \\\'(^\\\'|.)*\\\' )|(?P<t_control> \\%line\\s[^\\n]*\\n )|(?P<t_comment> \\#[^\\n]* )|(?P<t_nl> \\n+ )|(?P<t_whitespace> [ |\\t]+ )|(?P<t_RBRACE>\\})|(?P<t_LBRACE>\\{)|(?P<t_RBRACKET>\\])|(?P<t_RPAREN>\\))|(?P<t_LBRACKET>\\[)|(?P<t_LPAREN>\\()|(?P<t_SEMI>;)'
_lextab = [
  None,
  ('t_Stepper','Stepper'),
  ('t_System','System'),
  ('t_Process','Process'),
  ('t_Variable','Variable'),
  ('t_number','number'),
  None,
  None,
  None,
  ('t_fullid','fullid'),
  ('t_identifier','identifier'),
  ('t_systempath','systempath'),
  ('t_quotedstrings','quotedstrings'),
  ('t_quotedstring','quotedstring'),
  None,
  None,
  ('t_control','control'),
  ('t_comment','comment'),
  ('t_nl','nl'),
  ('t_whitespace','whitespace'),
  (None,'RBRACE'),
  (None,'LBRACE'),
  (None,'RBRACKET'),
  (None,'RPAREN'),
  (None,'LBRACKET'),
  (None,'LPAREN'),
  (None,'SEMI'),
]
_lextokens = {'LBRACE': None, 'systempath': None, 'fullid': None, 'SEMI': None, 'Process': None, 'Stepper': None, 'System': None, 'RBRACE': None, 'quotedstring': None, 'LBRACKET': None, 'LPAREN': None, 'quotedstrings': None, 'Variable': None, 'number': None, 'RBRACKET': None, 'identifier': None, 'RPAREN': None}
_lexignore = None
_lexerrorf = 't_error'
