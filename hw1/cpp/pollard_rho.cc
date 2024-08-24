#include "pollard_rho.h"

#include <cstdlib>
#include <ctime>

#include "euclid.h"

namespace hw1 {

PollardRho::PollardRho(const ::std::vector<uint64_t> &poly) : poly_(poly, 1) {
  // Seed the RNG.
  srand(time(NULL));
}

uint64_t PollardRho::Factor(uint64_t number) {
  // Set the correct modulus.
  poly_.SetModulus(number);

  // Start at a random point.
  uint64_t node1 = ((uint64_t)rand() << 32) | rand();
  uint64_t node2 = node1;

  uint64_t gcd = 1;
  while (gcd == 1) {
    node1 = poly_.Evaluate(node1);
    node2 = poly_.Evaluate(poly_.Evaluate(node2));

    uint64_t diff;
    if (node1 > node2) {
      diff = node1 - node2;
    } else {
      diff = node2 - node1;
    }
    gcd = numerical::Gcd(diff, number);
  }

  return gcd;
}

}  // namespace hw1
