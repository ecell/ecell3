
CLASSNAME = 'DegradeClb2Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','k3',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
  Reactant C0;
  Reactant C1;
  Reactant C2;
'''

defineMethod( 'initialize', '''
 S0 = getReactant( "S0" );
 C0 = getReactant( "C0" );
 C1 = getReactant( "C1" );
 C2 = getReactant( "C2" );
''' )

defineMethod( 'react', '''
  const Real S( S0.getSubstance()->getConcentration() );
  const Real E1( C0.getSubstance()->getConcentration() );
  const Real E2( C1.getSubstance()->getConcentration() );
  const Real E3( C2.getSubstance()->getConcentration() );
  Real v( k1 * (E1 - E2) + k2 * E2 + k3 * E3);
  Real V( v * S );
  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

