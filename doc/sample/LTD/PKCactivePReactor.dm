CLASSNAME = 'PKCactivePReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = []

PROTECTED_AUX = '''
  Reactant S0;
  Reactant S1;
  Reactant S2;
  Reactant S3;
  Reactant P0;
'''

defineMethod('initialize','''
  S0 = getReactant( "S0" );
  S1 = getReactant( "S1" );
  S2 = getReactant( "S2" );
  S3 = getReactant( "S3" );
  P0 = getReactant( "P0" );
''')

defineMethod('react',
'''
  const Real s0( S0.getSubstance()->getQuantity() );
  const Real s1( S1.getSubstance()->getQuantity() );
  const Real s2( S2.getSubstance()->getQuantity() );
  const Real s3( S3.getSubstance()->getQuantity() );

  const Real p( s0 + s1 + s2 + s3 );

  P0.getSubstance()->setQuantity(p);
''')
