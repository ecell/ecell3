#ifndef __PROPERTYSLOTMAKER_HPP
#define __PROPERTYSLOTMAKER_HPP

#include "libecs.hpp"

#include "PropertySlot.hpp"
#include "PropertyInterface.hpp"



namespace libecs
{


  /** \addtogroup property
   *@{
   */


  class PropertySlotMaker
  {
    
  public:
    
    template<class T, typename SlotType>
    static PropertySlotPtr 
    createPropertySlot( StringCref aName,
			T& anObject,
			Type2Type<SlotType>,
			typename ConcretePropertySlot<T,SlotType>::SetMethodPtr
			aSetMethodPtr,
			typename ConcretePropertySlot<T,SlotType>::GetMethodPtr
			aGetMethodPtr )
    {
      if( aSetMethodPtr == NULLPTR )
	{
	  aSetMethodPtr = &PropertyInterface::nullSet;
	}
      
      if( aGetMethodPtr == NULLPTR )
	{
	  aGetMethodPtr = &PropertyInterface::nullGet<SlotType>;
	}
      
      return new ConcretePropertySlot<T,SlotType>( aName,
						   anObject,
						   aSetMethodPtr,
						   aGetMethodPtr );
    
    }

  };




  /*@}*/

}

#endif /* __PROPERTYSLOTMAKER_HPP */
