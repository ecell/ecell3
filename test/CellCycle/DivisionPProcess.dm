CLASSNAME = 'DivisionPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','m',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
  VariableReference P1;
  VariableReference P2;
  VariableReference P0;
  Real prev; // declare member variable
  Real counter; // seconds (when step inverval is 0.001)
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
  P1 = getVariableReference( "P1" );
  P2 = getVariableReference( "P2" );
  P0 = getVariableReference( "P0" );
  prev = 0; // initialize member variable
  counter = 0; // for debugging purposes!!!
''' )

defineMethod( 'process', '''
  // Real prev( 0 ); // removed static 

  counter += 0.001;
  Real D( 1.026/m - 32 );
  Real f( exp(-m * D) );
  const Real E( C0.getVariable()->getConcentration() ); // Clb2

  //std::cout << "prev:" << prev;
  std::cout << ", E:" << E;

  if(prev > 0.3 && E <= 0.3)
  {
    std::cout << "BANG!!\a";
    std::cout << "counter:" << counter;
    std::cout << "prev:" << prev;
    std::cout << ", E:" << E;
    std::cout << ", Clb2:" << P0.getVariable()->getValue() << "\n";
    P1.getVariable()->setValue( 0 ); // BUD
    P2.getVariable()->setValue( 0 ); // SPN
    Real Pro1( P0.getVariable()->getConcentration() ); // P1 -> Pro1
    Pro1 *= N_A * getSuperSystem()->getVolume(); // P1 -> Pro1
    P0.getVariable()->setValue( Pro1 * f ); // mass

  }


  prev = E;
''' )
