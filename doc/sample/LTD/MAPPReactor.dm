CLASSNAME = 'MAPPReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = []

PROTECTED_AUX = '''
  Reactant S0;
  Reactant S1;
  Reactant P0;
'''
defineMethod('initialize','''
  S0 = getReactant( "S0" );
  S1 = getReactant( "S1" );
  P0 = getReactant( "P0" );
''')

defineMethod('react',
'''
  const Real s0( S0.getSubstance()->getQuantity() );
  const Real s1( S1.getSubstance()->getQuantity() );
  
  const Real p( s0 + s1 );
  
  P0.getSubstance()->setQuantity(p);
''')

