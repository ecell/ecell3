
CLASSNAME = 'DegradeClb5Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
  Reactant C0;
'''

defineMethod( 'initialize', '''
  S0 = getReactant( "S0" );
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  const Real S( S0.getSubstance()->getConcentration() );
  const Real E1( C0.getSubstance()->getConcentration() );
  Real v( k1 + k2 * E1 );
  Real V( v * S );

  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

