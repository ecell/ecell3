
CLASSNAME = 'Sic1ComplexDegradeReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','e3',0.0),('Real','e4',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
  Reactant C0;
  Reactant C1;
  Reactant C2;
  Reactant C3;
  Reactant C4;
  Reactant C5;
'''

defineMethod( 'initialize', '''
  S0 = getReactant( "S0" );
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
  C2 = getReactant( "C2" );
  C3 = getReactant( "C3" );
  C4 = getReactant( "C4" );
  C5 = getReactant( "C5" );
''' )

defineMethod( 'react', '''
  const Real S( S0.getSubstance()->getConcentration() );
  const Real E1( C0.getSubstance()->getConcentration() ); 
  const Real E2( C1.getSubstance()->getConcentration() );
  const Real E3( C2.getSubstance()->getConcentration() );
  const Real E4( C3.getSubstance()->getConcentration() );
  const Real E5( C4.getSubstance()->getConcentration() );
  const Real E6( C5.getSubstance()->getConcentration() );

  Real v( e1 * E1 + e2 * E2 + E3 + e3 * E4 + e4 * E5 );
  v *= k1;
  v *= S;
  Real V( v );
  V /= ( J + E6 );
  V *= S0.getSubstance()->getConcentration();
  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

