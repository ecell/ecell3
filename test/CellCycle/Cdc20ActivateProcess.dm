
CLASSNAME = 'Cdc20ActivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX = '''
  VariableReference P0;
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  P0 = getVariableReference( "P0" );
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  const Real P( P0.getVariable()->getConcentration());
  const Real E1( C0.getVariable()->getConcentration());
  Real V( k * (E1 - P) );

  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )


