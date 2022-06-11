#ifndef _HASH_TABLE_H_
#define _HASH_TABLE_H_

#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

// basic hash table structure
class hash_table {
  public:

    virtual uint32_t hash(uint32_t key)  =0;
    virtual uint32_t hash_next(uint32_t key, uint32_t prev_index) = 0;
    virtual bool read(uint32_t key, uint64_t* value_buffer) = 0;
    virtual bool insert(uint32_t key, uint64_t value) = 0;
    virtual int num_items() = 0;
    //virtual bool remove(uint32_t key) = 0; //JL: We do not consider remove function .
};

#endif
