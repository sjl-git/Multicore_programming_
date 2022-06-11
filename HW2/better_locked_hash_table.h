#ifndef _BETTER_LOCKED_HASH_TABLE_H_
#define _BETTER_LOCKED_HASH_TABLE_H_


#include <iostream>
#include <mutex>
#include <thread>
#include "hash_table.h"
#include "bucket.h"

class better_locked_probing_hash_table : public hash_table {

  private:
    Bucket* table;
    const int TABLE_SIZE; //we do not consider resizing. Thus the table has to be larger than the max num items.

    /* TODO: put your own code here  (if you need something)*/
    /****************/
    std::mutex* mutexes;
    int numMutex = 10000;
    /****************/
    /* TODO: put your own code here */

    public:

    better_locked_probing_hash_table(int table_size):TABLE_SIZE(table_size){
      this->table = new Bucket[TABLE_SIZE]();
      for(int i=0;i<TABLE_SIZE;i++) {
        this->table[i].valid=0; //means empty
      }
      this->mutexes = (std::mutex*) malloc(sizeof(std::mutex)*numMutex);
    }

    virtual uint32_t hash(uint32_t x)  
    {
      //https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
      return (x % TABLE_SIZE); 
    }

    virtual uint32_t hash_next(uint32_t key, uint32_t prev_index)
    {
      //linear probing. no special secondary hashfunction
      return ((prev_index + 1)% TABLE_SIZE); 
    }

    //the buffer has to be allocated by the caller
    bool read(uint32_t key, uint64_t* value_buffer){
      uint64_t index = this->hash(key);
      for (int i=0; i<TABLE_SIZE; i++) {
        if (table[index].valid) {
          if (table[index].key == key) {
            *value_buffer = table[index].value;
            return true;
          } else {
            index = this->hash_next(key, index);
          }
        } else {
          break;
        }
      }

      return false;
    }


    bool insert(uint32_t key, uint64_t value) {
      uint64_t index = this -> hash(key);
      for (int i=0; i<TABLE_SIZE; i++) {
        std::lock_guard<std::mutex> lock(mutexes[index%numMutex]);
        if (!table[index].valid || (table[index].key == key)) {
          table[index].valid = true;
          table[index].key = key;
          table[index].value = value;
          return true;
        } else {
          index = this->hash_next(key, index);
        }
      }
      return false;
    }

    int num_items() {
      int count=0;
      for(int i=0;i<TABLE_SIZE;i++) {
        if(table[i].valid==true) count++;
      }
      return count;
    }
};

#endif
