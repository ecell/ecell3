CLASSNAME = 'MichaelisUniUniReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','KmS',0.0), ('Real','KcF',0.0)]

REACTANT_SLOTS = [('S0','Concentration'),('C0','Quantity')]

DIFFERENTIATE_METHOD =\
'''
  Real velocity( KcF );

  //velocity *= getCatalyst(0)->getSubstance().getQuantity();
  //  Real S = getSubstrate(0)->getSubstance().getConcentration();


  velocity *= C0_Quantity->getReal();
  const Real S( S0_Concentration->getReal() );
  velocity *= S;
  velocity /= ( KmS + S );

  process( velocity );
'''
