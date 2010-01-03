//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public // License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#define BOOST_TEST_MODULE "VirtualMachine"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/type_traits.hpp>

#include "scripting/VirtualMachine.hpp"
#include "scripting/Assembler.hpp"

#include <iostream>

using namespace libecs;

BOOST_AUTO_TEST_CASE(testBasic)
{
    for ( Real i = 0; i < 100; i += 0.8 ) {
        std::auto_ptr<scripting::Code> code(new scripting::Code());
        scripting::VirtualMachine vm;
        scripting::Assembler a( code.get() );
        a.appendInstruction(
            scripting::Instruction< scripting::PUSH_REAL >( i ) );
        a.appendInstruction( scripting::Instruction< scripting::RET >() );
        BOOST_CHECK_EQUAL( i, vm.execute( *code ) );
    }

    for ( Real i = 0; i < 100; i += 0.8 ) {
        std::auto_ptr<scripting::Code> code(new scripting::Code());
        scripting::VirtualMachine vm;
        scripting::Assembler a( code.get() );
        a.appendInstruction(
            scripting::Instruction< scripting::PUSH_REAL >( i ) );
        a.appendInstruction(
            scripting::Instruction< scripting::PUSH_REAL >( i ) );
        a.appendInstruction( scripting::Instruction< scripting::ADD >() );
        a.appendInstruction( scripting::Instruction< scripting::RET >() );
        BOOST_CHECK_CLOSE_FRACTION( i * 2, vm.execute( *code ), 50 );
    }

    for ( Real i = 0; i < 100; i += 0.8 ) {
        std::auto_ptr<scripting::Code> code(new scripting::Code());
        scripting::VirtualMachine vm;
        scripting::Assembler a( code.get() );
        a.appendInstruction(
            scripting::Instruction< scripting::PUSH_REAL >( i ) );
        a.appendInstruction(
            scripting::Instruction< scripting::PUSH_REAL >( i ) );
        a.appendInstruction( scripting::Instruction< scripting::ADD >() );
        a.appendInstruction(
            scripting::Instruction< scripting::PUSH_REAL >( i ) );
        a.appendInstruction( scripting::Instruction< scripting::ADD >() );
        a.appendInstruction( scripting::Instruction< scripting::RET >() );
        BOOST_CHECK_CLOSE_FRACTION( i * 3, vm.execute( *code ), 50 );
    }
}

