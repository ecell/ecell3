CLASSNAME = 'BUDProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0),('Real','e',0.0),]

PROTECTED_AUX = '''
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  C2 = getVariableReference( "C2" );
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
