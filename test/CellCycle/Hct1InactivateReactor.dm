
CLASSNAME = 'Hct1InactivateReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','e3',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
  Reactant C0;
  Reactant C1;
  Reactant C2;
  Reactant C3;
'''

defineMethod( 'initialize', '''
  S0 = getReactant( "S0" );
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
  C2 = getReactant( "C2" );
  C3 = getReactant( "C3" );
''' )

defineMethod( 'react', '''
  const Real S( S0.getSubstance()->getConcentration() );
  const Real E1( C0.getSubstance()->getConcentration() );
  const Real E2( C1.getSubstance()->getConcentration() );
  const Real E3( C2.getSubstance()->getConcentration() );
  const Real E4( C3.getSubstance()->getConcentration() );
  Real v( E1 + e1 * E2 + e2 * E3 + e3 * E4 );
  v *= k2;
  v += k1;
  Real V( v * S ); 
  V /= J + S;
  V += getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

