
CLASSNAME = 'Sic1ComplexDegradeProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','e3',0.0),('Real','e4',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
  VariableReference C3;
  VariableReference C4;
  VariableReference C5;
'''

defineMethod( 'initialize', '''
  S0 = getVariableReference( "S0" );
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  C2 = getVariableReference( "C2" );
  C3 = getVariableReference( "C3" );
  C4 = getVariableReference( "C4" );
  C5 = getVariableReference( "C5" );
''' )

defineMethod( 'process', '''
  const Real S( S0.getVariable()->getConcentration() );
  const Real E1( C0.getVariable()->getConcentration() ); 
  const Real E2( C1.getVariable()->getConcentration() );
  const Real E3( C2.getVariable()->getConcentration() );
  const Real E4( C3.getVariable()->getConcentration() );
  const Real E5( C4.getVariable()->getConcentration() );
  const Real E6( C5.getVariable()->getConcentration() );

  Real v( e1 * E1 + e2 * E2 + E3 + e3 * E4 + e4 * E5 );
  v *= k1;
  v *= S;
  Real V( v );
  V /= ( J + E6 );
  V *= S0.getVariable()->getConcentration();
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

