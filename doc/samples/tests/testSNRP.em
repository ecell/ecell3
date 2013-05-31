# A test model for Spatiocyte 
# written by Satya Arjunan <satya.arjunan(a)gmail.com>

Stepper SpatiocyteStepper(SS)
{
  VoxelRadius 4.4e-9;
}

Stepper ODEStepper(DE)
{
}

System System( / )
{
  StepperID       SS; 
  Variable Variable(GEOMETRY)
    {
      Value 3;     
    } 
  Variable Variable(LENGTHX)
    {
      Value 4.5e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 1e-6;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 0; 
    } 
  Variable Variable(A)
    {
      Value 0;         # molecule number
      Name "HD";
    } 
  Variable Variable(B)
    {
      Value 0;         # molecule number
      Name "HD";
    } 
  Variable Variable(C)
    {
      Value 0;         # molecule number
      Name "HD";
    } 
  Variable Variable(D)
    {
      Value 0;         # molecule number
      Name "HD";
    } 
}

System System(/Surface)
{
  StepperID SS;

  Variable Variable(DIMENSION)
    {
      Value 2;         # { 3: Volume
                       #   2: Surface }
    }
  Variable Variable(VACANT)
    {
      Value 0;
    }

  Variable Variable(A)
    {
      Value 2000;         # molecule number
      Name "HD";
    }
  Process MassActionProcess( ES_to_E_P )
    {
      StepperID DE;
      VariableReferenceList [_ Variable:/Surface:A -1]
                            [_ Variable:/:A 1];
      k 2.3e-3;
    }  

  Variable Variable(B)
    {
      Value 2000;         # molecule number
      Name "HD";
    }
  Process SpatiocyteNextReactionProcess(sph)
    {
      VariableReferenceList   [_ Variable:/Surface:B -1]
                              [_ Variable:/:B 1];
      k 2.3e-3;
    }  

  Variable Variable(Ca)
    {
      Value 1000;         # molecule number
    }
  Variable Variable(Cb)
    {
      Value 1000;         # molecule number
    }
  Process MoleculePopulateProcess(populate3)
    {
      VariableReferenceList  [_ Variable:/Surface:Ca]
                             [_ Variable:/Surface:Cb];
    }
  Process SpatiocyteNextReactionProcess(sp)
    {
      VariableReferenceList   [_ Variable:/Surface:Ca -1]
                              [_ Variable:/:C 1];
      k 2.3e-3;
    }  
  Process SpatiocyteNextReactionProcess(sp2)
    {
      VariableReferenceList   [_ Variable:/Surface:Cb -1]
                              [_ Variable:/:C 1];
      k 2.3e-3;
    }  
  Variable Variable(E)
    {
      Value 1000;         # molecule number
    }
  Variable Variable(F)
    {
      Value 1000;         # molecule number
    }
  Process MoleculePopulateProcess(populate5)
    {
      VariableReferenceList  [_ Variable:/Surface:E]
                             [_ Variable:/Surface:F];  
    }
  Process DiffusionProcess(diffpe)
    {
      VariableReferenceList   [_ Variable:/Surface:F];
      D 0.02e-12; 
    }
  Process DiffusionProcess(diffpg)
    {
      VariableReferenceList   [_ Variable:/Surface:E];
      D 0.02e-12; 
    }
  Process SpatiocyteNextReactionProcess(pg2)
    {
      VariableReferenceList   [_ Variable:/Surface:E -1]
                              [_ Variable:/:D 1];
      k 2.3e-3;          # s^{-1}
    }  
  Process SpatiocyteNextReactionProcess(pe2)
    {
      VariableReferenceList   [_ Variable:/Surface:F -1]
                              [_ Variable:/:D 1];
      k 2.3e-3;          # s^{-1}
    }  
  }
