CLASSNAME = 'TotalPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = []

PROTECTED_AUX = '''
  VariableReference C0;
  VariableReference C1;
  VariableReference P0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  P0 = getVariableReference( "P0" );
''' )

defineMethod( 'process', '''
  Real V( C0.getVariable()->getConcentration() );

  //  V -= C0.getVariable()->getConcentration();
  V -= C1.getVariable()->getConcentration();
  
  V *= N_A * getSuperSystem()->getVolume();
  P0.getVariable()->setValue( V );
''' )
