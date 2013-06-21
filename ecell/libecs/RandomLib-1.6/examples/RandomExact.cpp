/**
 * \file RandomExact.cpp
 * \brief Using %RandomLib to generate exact random results
 *
 * Compile/link with, e.g.,\n
 * g++ -I../include -O2 -funroll-loops
 *   -o RandomExact RandomExact.cpp ../src/Random.cpp\n
 * ./RandomExact
 *
 * See \ref otherdist, for more information.
 *
 * Copyright (c) Charles Karney (2006-2012) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <utility>
#include <RandomLib/Random.hpp>
#include <RandomLib/RandomNumber.hpp>
#include <RandomLib/ExactNormal.hpp>
#include <RandomLib/ExactExponential.hpp>
#include <RandomLib/InversePiProb.hpp>
#include <RandomLib/InverseEProb.hpp>

int main(int, char**) {
  // Create r with a random seed
  RandomLib::Random r; r.Reseed();
  std::cout << "Using " << r.Name() << "\n"
            << "with seed " << r.SeedString() << "\n\n";
  {
    std::cout
      << "Sampling exactly from the normal distribution.  First number is\n"
      << "in binary with ... indicating an infinite sequence of random\n"
      << "bits.  Second number gives the corresponding interval.  Third\n"
      << "number is the result of filling in the missing bits and rounding\n"
      << "exactly to the nearest representable double.\n";
    const int bits = 1;
    RandomLib::ExactNormal<bits> ndist;
    long long num = 20000000ll;
    long long bitcount = 0;
    int numprint = 16;
    for (long long i = 0; i < num; ++i) {
      long long k = r.Count();
      RandomLib::RandomNumber<bits> x = ndist(r); // Sample
      bitcount += r.Count() - k;
      if (i < numprint) {
        std::pair<double, double> z = x.Range();
        std::cout << x << " = "   // Print in binary with ellipsis
                  << "(" << z.first << "," << z.second << ")"; // Print range
        double v = x.Value<double>(r); // Round exactly to nearest double
        std::cout << " = " << v << "\n";
      } else if (i == numprint)
        std::cout << std::flush;
    }
    std::cout
      << "Number of bits needed to obtain the binary representation averaged\n"
      << "over " << num << " samples = " << bitcount/double(num) << "\n\n";
  }
  {
    std::cout
      << "Sampling exactly from exp(-x).  First number is in binary with\n"
      << "... indicating an infinite sequence of random bits.  Second\n"
      << "number gives the corresponding interval.  Third number is the\n"
      << "result of filling in the missing bits and rounding exactly to\n"
      << "the nearest representable double.\n";
    const int bits = 1;
    RandomLib::ExactExponential<bits> edist;
    long long num = 100000000ll;
    long long bitcount = 0;
    int numprint = 16;
    for (long long i = 0; i < num; ++i) {
      long long k = r.Count();
      RandomLib::RandomNumber<bits> x = edist(r); // Sample
      bitcount += r.Count() - k;
      if (i < numprint) {
        std::pair<double, double> z = x.Range();
        std::cout << x << " = "   // Print in binary with ellipsis
                  << "(" << z.first << "," << z.second << ")"; // Print range
        double v = x.Value<double>(r); // Round exactly to nearest double
        std::cout << " = " << v << "\n";
      } else if (i == numprint)
        std::cout << std::flush;
    }
    std::cout
      << "Number of bits needed to obtain the binary representation averaged\n"
      << "over " << num << " samples = " << bitcount/double(num) << "\n\n";
  }
  {
    std::cout
      << "Random bits with 1 occurring with probability 1/pi exactly:\n";
    long long num = 100000000ll;
    int numprint = 72;
    RandomLib::InversePiProb pp;
    long long nbits = 0;
    long long k = r.Count();
    for (long long i = 0; i < num; ++i) {
      bool b = pp(r);
      nbits += int(b);
      if (i < numprint) std::cout << int(b);
      else if (i == numprint) std::cout << "..." << std::flush;
    }
    std::cout << "\n";
    std::cout << "Frequency of 1 averaged over " << num << " samples = 1/"
              << double(num)/nbits << "\n"
              << "bits/sample = " << (r.Count() - k)/double(num) << "\n\n";
  }
  {
    std::cout
      << "Random bits with 1 occurring with probability 1/e exactly:\n";
    long long num = 1000000000ll;
    int numprint = 72;
    RandomLib::InverseEProb ep;
    long long nbits = 0;
    long long k = r.Count();
    for (long long i = 0; i < num; ++i) {
      bool b = ep(r);
      nbits += int(b);
      if (i < numprint) std::cout << int(b);
      else if (i == numprint) std::cout << "..." << std::flush;
    }
    std::cout << "\n";
    std::cout << "Frequency of 1 averaged over " << num << " samples = 1/"
              << double(num)/nbits << "\n"
              << "bits/sample = " << (r.Count() - k)/double(num) << "\n";
  }
  return 0;
}
