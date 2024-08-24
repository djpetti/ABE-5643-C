#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

#include "pollard_rho.h"

// Prints usage information and exits the program.
// Args:
//  name: The name of the program.
void PrintUsageAndExit(const char *name) {
  printf("Usage: %s number\n", name);
  exit(1);
}

int main(int argc, const char **argv) {
  if (argc != 2) {
    // Incorrect number of arguments.
    PrintUsageAndExit(argv[0]);
  }

  const uint32_t number = atoi(argv[1]);

  // The polynomial coefficients to use for factoring.
  ::std::vector<uint64_t> poly;
  poly.push_back(1);
  poly.push_back(0);
  poly.push_back(1);

  // Factor the number.
  ::hw1::PollardRho pollard(poly);
  const uint64_t factor = pollard.Factor(number);

  if (!factor) {
    printf("Could not find a factor of %u.\n", number);
  }
  printf("Found a factor: %lu.\n", factor);

  return 0;
}
