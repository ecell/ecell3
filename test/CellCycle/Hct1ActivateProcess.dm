
CLASSNAME = 'Hct1ActivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  VariableReference P0;
  VariableReference C0;
  VariableReference C1;
'''

defineMethod( 'initialize', '''
  P0 = getVariableReference( "P0" );
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
''' )

defineMethod( 'process', '''
  const Real P( P0.getVariable()->getConcentration() ); 
  const Real E1( C0.getVariable()->getConcentration() ); 
  const Real E2( C1.getVariable()->getConcentration() ); 
  Real V( k1 + k2 * E1 );
  V *= E2 - P;
  V /= J + E2 - P;
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

