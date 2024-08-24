#include "euclid.h"

namespace hw1 {
namespace numerical {

uint64_t Gcd(uint64_t dividend, uint64_t divisor) {
  uint64_t remainder = dividend % divisor;

  while (remainder != 0) {
    dividend = divisor;
    divisor = remainder;
    remainder = dividend % divisor;
  }

  return divisor;
}

}  // namespace numerical
}  // namespace hw1
