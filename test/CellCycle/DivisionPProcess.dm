CLASSNAME = 'DivisionPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','m',0.0)]

PROTECTED_AUX = '''
  Connection C0;
  Connection P1;
  Connection P2;
  Connection P0;
  Real prev;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  P1 = getConnection( "P1" );
  P2 = getConnection( "P2" );
  P0 = getConnection( "P0" );
  prev = 0;
''' )

defineMethod( 'process', '''
  Real prev( 0 ); // removed static 
  int counter = 0; // for debugging purposes!!!
  Real D( 1.026/m - 32 );
  Real f( exp(-m * D) );

  const Real E( C0.getVariable()->getConcentration() ); // put Clb2t instead of Clb2 for C0

  if(prev > 0.3 && E <= 0.3)
  {
    std::cout << "BANG!!\a";
    std::cout << "prev:" << prev;
    std::cout << ", E:" << E;
    std::cout << ", Clb2:" << P0.getVariable()->getValue() << "\n";
    P1.getVariable()->setValue( 0 );
    P2.getVariable()->setValue( 0 );
    Real Pro1( P0.getVariable()->getConcentration() ); // P1 -> Pro1
    Pro1 *= N_A * getSuperSystem()->getVolume(); // P1 -> Pro1
    P0.getVariable()->setValue( Pro1 * f ); // P1 -> Pro1

  }


  prev = E;
''' )
