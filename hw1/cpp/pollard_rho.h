#ifndef HW1_POLLARD_RHO_H_
#define HW1_POLLARD_RHO_H_

#include <cstdint>

#include <vector>

#include "poly.h"

namespace hw1 {

// Implements Pollard's rho algorithm for factorization.
class PollardRho {
 public:
  // Args:
  //  poly: The polynomial coefficients to use.
  explicit PollardRho(const ::std::vector<uint64_t> &poly);

  // Factors an integer.
  // Args:
  //  number: The number to factor.
  // Returns:
  //  A non-trivial factor of the input, or zero upon failure.
  uint64_t Factor(uint64_t number);

 private:
  // Internal polynomial to use.
  numerical::Poly poly_;
};

}  // namespace hw1

#endif  // HW1_RHO_H_
