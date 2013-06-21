# Build the examples...

# Loop over all the examples, specifying the source and library.
add_custom_target (examples ALL)
foreach (EXAMPLE ${EXAMPLES})

  add_executable (${EXAMPLE} ${EXAMPLE}.cpp)
  add_dependencies (examples ${EXAMPLE})

  target_link_libraries (${EXAMPLE} Random)

endforeach ()

# Set the run time path for shared libraries for non-Windows machines.
if (NOT WIN32)
  set_target_properties (${TOOLS} PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
endif ()

# RandomSave uses boost serialization if it's available
if (Boost_SERIALIZATION_FOUND)
  include_directories (${Boost_INCLUDE_DIRS})
  set_target_properties (RandomSave PROPERTIES
    COMPILE_DEFINITIONS HAVE_BOOST_SERIALIZATION=1)
  target_link_libraries (RandomSave ${Boost_LIBRARIES})
endif ()

# RandomThread uses OpenMP if it's available
if (OPENMP_FOUND)
  set_target_properties (RandomThread PROPERTIES
    COMPILE_FLAGS ${OpenMP_CXX_FLAGS} COMPILE_DEFINITIONS HAVE_OPENMP=1)
  if (NOT WIN32)
    set_target_properties (RandomThread
      PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  endif ()
endif ()

# RandomLambda uses C++11 lambda if it's available
include (CheckCXXCompilerFlag)
check_cxx_compiler_flag (-std=gnu++0x GNUPP11)
if (GNUPP11)
  set_target_properties (RandomLambda PROPERTIES
    COMPILE_FLAGS -std=gnu++0x)
endif ()

if (HAVE_MPFR)
  add_executable (MPFRExample MPFRExample.cpp)
  add_dependencies (examples MPFRExample)
  target_link_libraries (MPFRExample ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
endif ()

if (WIN32)
  # Only install RandomExample and only on Windows systems
  install (TARGETS RandomExample DESTINATION bin)
endif ()

# Copy the shared library on Windows systems to this directory (tools) so
# that the tests can be run.
if (WIN32 AND RANDOM_SHARED_LIB)
  get_target_property (RANDOM_LIBRARY_FILE Random
    "LOCATION_${CMAKE_CFG_INTDIR}")
  add_custom_command (TARGET examples POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy ${RANDOM_LIBRARY_FILE} ${CMAKE_CFG_INTDIR}/)
endif ()

# Turn on testing
enable_testing ()

# Here are the tests.
add_test (RandomExample0 RandomExample)
add_test (RandomSave0 RandomSave)
add_test (RandomCoverage0 RandomCoverage)
