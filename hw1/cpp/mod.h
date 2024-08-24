#ifndef HW1_NUMERICAL_MOD_H_
#define HW1_NUMERICAL_MOD_H_

#include <cstdint>

namespace hw1::numerical {

// Mod operator that's safe for use with negative numbers, unlike the C++
// default.
// Args:
//  dividend: The number to divide.
//  divisor: The number to divide by.
// Returns:
//  The remainder of dividend / divisor, behaving as we expect it to if dividend
//  is negative. (Ex. Mod(-3, 5) is 2.)
int64_t Mod(int64_t dividend, int64_t divisor);

// Performs modular exponentiation.
// Args:
//  base: The base.
//  exp: The exponent.
//  mod: The modulus.
// Returns:
//  base ^ exp % mod
uint64_t Power(uint64_t base, uint64_t exp, uint64_t mod);

} // namespace hw1::numerical


#endif  // HW1_NUMERICAL_MOD_H_
