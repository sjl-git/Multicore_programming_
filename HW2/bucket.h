#ifndef _BUCKET_H_
#define _BUCKET_H_
#include <unistd.h>

struct Bucket { //16B in its size
  uint32_t key;  // 2B key
  uint16_t valid; // 0=empty 1=occupied
  uint16_t meta; // Currently Unused. Reserved for 1B misc metadata (timestamp, contained lock, memo, tag, or anything)
  uint64_t value;
};

#endif
