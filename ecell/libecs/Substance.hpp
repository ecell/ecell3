//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER





#ifndef ___SUBSTANCE_H___
#define ___SUBSTANCE_H___
#include <cmath>
#include <errno.h>
#include <iostream.h>
#include "Defs.h"
//#include "ecell/TimeManager.h"
//#include "ecell/MessageWindow.h"
#include "Entity.h"


class RootSystem;
class Reactor;
class Integrator;
class Accumulator;


// exceptions

class IntegrationErr {};
class LTZ : public IntegrationErr {};


/*!
  Substance class is used to represent molecular species.
*/
class Substance : public Entity
{
//friend class RootSystem;

friend class SSystem;
friend class ECSMainWindow;
friend class Stepper;
friend class Integrator;
friend class Accumulator;

public:

  //! A class name string of system default accumulator
  const static string SYSTEM_DEFAULT_ACCUMULATOR_NAME;

  void setQuantity(const Message& message);
  void setAccumulator(const Message& message);
  const Message getQuantity(const string& keyword);
  const Message getAccumulator(const string& keyword);

private:

  static string USER_DEFAULT_ACCUMULATOR_NAME;


  Accumulator* _accumulator;
  Integrator* _integrator;

  Float _quantity;
  Float _fraction;
  Float _velocity;

  Quantity _bias;

  bool _fixed;

  Concentration _concentration;

  Concentration calculateConcentration();
  
  void mySetQuantity(Float q) {_quantity = q; _concentration = -1;}

protected:


  void setAccumulator(const string& classname);
  void setAccumulator(Accumulator* a);
  void setIntegrator(Integrator* i) {_integrator = i;}

  void makeSlots();

public:

  Substance();

  virtual ~Substance();

  static Substance* instance() {return new Substance;}

  const string fqpn() const;

  /*!
    Add bias value in current step.
    Usually used by Interfaces.
    \param b bias value.
    \return current bias value in current step.
   */
  Quantity bias(Quantity b)                {return _bias += b;}

  /*!
    \return the number of molecules.
   */
  Float quantity() const               {return _quantity;}

  /*!
    Fixes or unfixes this Substance.
    \param f Boolean value. true -> fix, false -> unfix.
    \return Current value.
   */
  bool fix(bool f)                         {return _fixed = f;}
  /*!
    \return true if the Substance is fixed or false if not.
   */
  bool fix() const                         {return _fixed;}

  /*!
    Whether concentration of this substance can be calculated or not.
    It must have a supersystem which have volume other than zero to
    calculate concentration.
    \return true -> if concentration can be obtained. false -> if not.
   */
  bool haveConcentration();

  /*!
    Returns a concentration value if it have.
    An error occurs when concentration values is requested from
    a substance which have no concentration value.
    \return Concentration in M (mol/L).
   */
  inline Concentration concentration() ;

  /*!
    Initializes this substance. 
    Called at startup.
   */
  void initialize();

  /*!
    Clear phase.
    Add bias to current value and clear the bias by zero.
    Then call clear() of integrator.
   */
  void clear();
  /*!
    This is called one or several times in react phase.
    Time of call is determined by type of integrator.
   */
  void turn();

  /*!
    Transit phase.
    Perform integration by a result calculated by integrator.
   */
  void transit();

  /*!
    \return current velocity value in (number of molecules)/(step)
   */
  Float velocity() const
    {return _velocity;}

  /*!
    \param v velocity in number of molecules to add.
    \return current velocity value in (number of molecules)/(step)
   */
  Float velocity(Float v);

  /*!
    Returns activity value of a Substance object.
    The activity is defined as current velocity.

    \return activity value of Substance in Float.
   */
  virtual Float activity();

  /*!
    Set a quantity with no check of fix() and anything.
    This updates the accumulator immediately.
    Intended to be used externally to the cell model.
    (e.g. loading cell state.) Use setQuantity() for usual purposes.

    \sa setQuantity
   */
  void loadQuantity(Float q);

  /*!
    Simply set a quantity of this Substance. Checks fix(). 
    This updates the accumulator immediately.
    Use this method within the cell model.

    \sa fix
    */
  void setQuantity(Float q)    
    { if(!fix()) loadQuantity(q);}


  /*!
    Get a quantity via save() method of the Accumulator.
    The way of getting the quantity depends on the Accumulator.
   */
  Float saveQuantity();

  //! set a class name string of user default accumulator
  static void setUserDefaultAccumulatorName(const string& name) 
    {USER_DEFAULT_ACCUMULATOR_NAME = name;}

  //! a class name string of user default accumulator
  static const string& userDefaultAccumulatorName() 
    {return USER_DEFAULT_ACCUMULATOR_NAME;}


};

/*
inline void Substance::react()
{
//  _integrator->react();
}
*/

inline Concentration Substance::concentration()
{
  return (_concentration >= 0) ? 
    _concentration : calculateConcentration(); 
}

typedef Substance* SubstancePtr;

#endif /* ___SUBSTANCE_H___ */





