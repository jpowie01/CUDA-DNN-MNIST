#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>

/* Debugging levels
 * ----------------
 *
 *  Level 0 - Disabled
 *  Level 1 - Only debug prints
 *  Level 2 - Debug prints + matrices
 */
#define DEBUG 0

#if defined(DEBUG) && DEBUG >= 1
 #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...)
#endif

float randomFloat(float a, float b);
int randomInt(int a, int b);

#endif /* !UTILS_HPP */
