CLASSNAME = 'DivisionPReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','m',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
  Reactant P1;
  Reactant P2;
  Reactant P0;
  Real prev;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  P1 = getReactant( "P1" );
  P2 = getReactant( "P2" );
  P0 = getReactant( "P0" );
  prev = 0;
''' )

defineMethod( 'react', '''
  //  static Real prev( 0 ); better to remove static if this reactor is used only once in the model
  int counter = 0; // for debugging purposes!!!
  Real D( 1.026/m - 32 );
  Real f( exp(-m * D) );

  const Real E( C0.getSubstance()->getConcentration() ); // put Clb2t instead of Clb2 for C0

  if(prev > 0.3 && E <= 0.3)
  {
    std::cout << "BANG!!\a";
    std::cout << "prev:" << prev;
    std::cout << ", E:" << E;
    std::cout << ", Clb2:" << P0.getSubstance()->getQuantity() << "\n";
    P1.getSubstance()->setQuantity( 0 );
    P2.getSubstance()->setQuantity( 0 );
    Real Pro1( P0.getSubstance()->getConcentration() ); // P1 -> Pro1
    Pro1 *= N_A * getSuperSystem()->getVolume(); // P1 -> Pro1
    P0.getSubstance()->setQuantity( Pro1 * f ); // P1 -> Pro1

  }


  prev = E;
''' )
