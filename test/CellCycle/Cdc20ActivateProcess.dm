
CLASSNAME = 'Cdc20ActivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX = '''
  Connection P0;
  Connection C0;
'''

defineMethod( 'initialize', '''
  P0 = getConnection( "P0" );
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  const Real P( P0.getVariable()->getConcentration());
  const Real E1( C0.getVariable()->getConcentration());
  Real V( k * (E1 - P) );

  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )


