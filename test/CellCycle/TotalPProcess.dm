CLASSNAME = 'TotalPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = []

PROTECTED_AUX = '''
  Connection C0;
  Connection C1;
  Connection P0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
  P0 = getConnection( "P0" );
''' )

defineMethod( 'react', '''
  Real V( C0.getVariable()->getConcentration() );

  //  V -= C0.getVariable()->getConcentration();
  V -= C1.getVariable()->getConcentration();
  
  V *= N_A * getSuperSystem()->getVolume();
  P0.getVariable()->setValue( V );
''' )
