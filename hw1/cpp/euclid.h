#ifndef HW1_NUMERICAL_EUCLID_H_
#define HW1_NUMERICAL_EUCLID_H_

#include <cstdint>


namespace hw1::numerical {

// Uses Euclid's algorithm to determine the GCD of two numbers.
// Args:
//  dividend: The first number.
//  divisor: The second number.
// Returns: The GCD of the two numbers.
uint64_t Gcd(uint64_t dividend, uint64_t divisor);

} // namespace hw1::numerical


#endif  // HW1_NUMERICAL_EUCLID_H_
