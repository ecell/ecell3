Stepper SpatiocyteStepper(SS) { VoxelRadius 3.3e-9; SearchVacant 0;}   # m

System System(/)
{
  StepperID       SS; 
  Variable Variable(GEOMETRY)
    {
      Value 0;
    } 
  Variable Variable(LENGTHX)
    {
      Value 1.8e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 1.8e-6;      # m
    } 
  Variable Variable(LENGTHZ)
    {
      Value 1.8e-6;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 0; 
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
  Variable Variable(MinD)
    {
      Value 400;         # molecule number
    }
  Process MoleculePopulateProcess(clss)
    {
      VariableReferenceList   [_ Variable:/Surface:MinD];
    }  
  Process PeriodicBoundaryDiffusionProcess( pro4 )
    {
      VariableReferenceList [ _ Variable:/Surface:MinD ];
      D 10e-12;
    }
  Process IteratingLogProcess(logDiffusion)
    {
      VariableReferenceList [ _ Variable:/Surface:MinD ];
      Iterations 1000;
      Diffusion 1;
      LogDuration 1e-3;
      LogInterval 1e-5;
    }
}

