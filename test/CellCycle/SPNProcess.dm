
CLASSNAME = 'SPNProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  const Real E1( C0.getVariable()->getConcentration() );
  Real V( k * E1 );
  V /= J + E1;
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

