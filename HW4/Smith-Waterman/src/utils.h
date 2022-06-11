#ifndef UTILS_H_
#define UTILS_H_


#define MIN(x, y) (((x)>(y))?(y):(x))
#define MAX(x, y) (((x)<(y))?(y):(x))
#define MAX3(x, y, z) (((x)>(y))?(((x)>(z))?(x):(z)):(((y)>(z))?(y):(z)))
#define MIN3(x, y, z) (((x)<(y))?(((x)<(z))?(x):(z)):(((y)<(z))?(y):(z)))

#endif
