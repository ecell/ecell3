CLASSNAME = 'BUDProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0),('Real','e',0.0),]

PROTECTED_AUX = '''
  Connection C0;
  Connection C1;
  Connection C2;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
  C2 = getConnection( "C2" );
''' )

defineMethod( 'process', '''
  Real E1 = C0.getVariable()->getConcentration();
  Real E2 = C1.getVariable()->getConcentration();
  Real E3 = C2.getVariable()->getConcentration();
  Real V = E1 + E2 + e*E3;
  V *= k;
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux(V);
''' )
