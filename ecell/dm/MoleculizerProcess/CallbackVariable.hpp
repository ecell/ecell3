#include "RegisterClass.hpp"

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>

USE_LIBECS;

namespace libecs
{


  LIBECS_DM_CLASS( CallbackVariable, Variable )
  {
  public:
    LIBECS_DM_OBJECT( CallbackVariable, Variable)
      {
        INHERIT_PROPERTIES( Variable );
      }
    
  public:
    CallbackVariable()
      :
      ptrRegisterClass( NULL ),
      neverBeenOverZeroBefore( true ),
      Variable()
      {
      }
     
    void registerCallback(RegisterClass* aRegisterClass)
    {
      ptrRegisterClass = aRegisterClass;
    }

    SET_METHOD( Real, Value)
      {
        if (value > 0 && neverBeenOverZeroBefore)
          {
            neverBeenOverZeroBefore = false;
            if(ptrRegisterClass)
              {
                ptrRegisterClass->addString( getFullID().getID() );
              }
          }

        if (!isFixed() )
          {
            loadValue( value );
          }
      }

  private:
    RegisterClass* ptrRegisterClass;
    bool neverBeenOverZeroBefore;
    Integer theValues;
    Integer DataChangeNdx;
      
  };


  
}
