CLASSNAME = 'Bck2PProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','Bck2',0.0)]

PROTECTED_AUX = '''
  Connection C0;
  Connection P0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  P0 = getConnection( "P0" );
''' )

defineMethod( 'process', '''
  const Real E1( C0.getVariable()->getConcentration() ); // const?
  Real V( Bck2 * E1);
  V *= N_A * getSuperSystem()->getVolume();
  P0.getVariable()->setValue( V );
''' )
