
CLASSNAME = 'BiUniProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
  VariableReference C1;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
''' )

defineMethod( 'process', '''
  const Real E1( C0.getVariable()->getConcentration() ); 
  const Real E2( C1.getVariable()->getConcentration() ); 
  Real V( k * E1 * E2 );

  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )


