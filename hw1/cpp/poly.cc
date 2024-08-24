#include "poly.h"

#include "mod.h"


namespace hw1::numerical {

Poly::Poly(const ::std::vector<uint64_t> &coefficients, uint64_t modulus)
    : coefficients_(coefficients), modulus_(modulus) {}

uint64_t Poly::Evaluate(uint64_t value) {
  uint64_t result = 0;

  // Compute all the powers.
  for (uint32_t i = 0; i < coefficients_.size(); ++i) {
    const uint64_t power = coefficients_.size() - i - 1;
    result += Power(value, power, modulus_);
  }

  return result;
}

void Poly::SetModulus(uint64_t modulus) {
  modulus_ = modulus;
}

} // namespace hw1::numerical

