#include "hash_table.h"
#include "better_locked_hash_table.h"
#include "locked_hash_table.h"

#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "error.h"



//PRNG: Pseudo Random Number Generator
//rand() is not thread-safe. So we use a simple near-random number instead
//the same as the hashfunctions, but no modular
//https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
uint32_t PRNG(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}


uint64_t gen_value(uint32_t key) {
  uint64_t val = key*5; //it is just an arbitrary recipe 
  val |= (val << 32); //make sure 64b works
  return val;
}

//sanity check
bool check_value(uint32_t key, uint64_t val) {
  uint64_t new_val = gen_value(key);
  return new_val == val;
}

void init_worker(hash_table* HT, int tid, int num_threads, int init_items) {
  for (int i = tid; i < init_items; i += num_threads) {
    uint32_t key = i;
    uint64_t value = gen_value(key);
    //std::cout<<"insert "<<key<<std::endl;
    bool res = HT->insert(key, value);
    assert(res == true); //should have succeeded.
  }
}

void test_worker(hash_table* HT, int tid, int num_threads, 
                  int num_init_items, int num_new_items,
                  int additional_reads_per_op) {
  uint64_t val_buffer=0;
  int seed = tid*tid*7 + tid*3;

  for (int i = tid; i < num_new_items; i += num_threads) { //static assignment
    uint32_t key = i + num_init_items;
    uint64_t value = gen_value(key);

    bool res = HT->insert(key, value);
    //std::cout<<tid<<":insert"<<key<<std::endl;
    assert(res == true); //should have succeeded.
    bool insert_chk = HT->read(key, &val_buffer);
    for (int j = 0; j < additional_reads_per_op; j++) {
      int key = PRNG( seed*100000 + j) % num_init_items;
      //std::cout<<tid<<":check "<<key<<std::endl;
      bool res = HT->read(key, &val_buffer);
      //all keys are within num_init, so they should exist
      if(!res || !check_value(key, val_buffer)) {
        std::cout<<"FAIL during test!: key: "<<std::hex<<key<<std::dec<<" exist?:"<<res<<" valbuffer:"<<std::hex<<val_buffer<<std::dec<<std::endl;
      }
    }
  }


}


//run this in a sequential thread for sanity check
void sequential_sanity_check(hash_table *HT, int num_init_items, int num_new_items) {
  uint64_t val_buffer=0;
  int end_items = HT->num_items();
  if(end_items != (num_init_items + num_new_items)) {
    std::cout<<"num items is wrong! :"<<end_items<<std::endl;
  }
  //init items
  for(int i=0; i<num_init_items; i++) {
    uint32_t key = i;
    bool res = HT->read(key, &val_buffer);
      if(!res || !check_value(key, val_buffer)) {
        //std::cout<<"FAIL during post-test sanity check for init items!: key: "<<key<<" exist?:"<<res<<" valbuffer:"<<val_buffer<<std::endl;
        std::cout<<"FAIL during post_test check!: key: "<<std::hex<<key<<std::dec<<" exist?:"<<res<<" valbuffer:"<<std::hex<<val_buffer<<std::dec<<std::endl;
      }
  }
  //new items
  for(int i=0; i<num_new_items; i++) {
    uint32_t key = i + num_init_items;
    bool res = HT->read(key, &val_buffer);
      if(!res || !check_value(key, val_buffer)) {
        //std::cout<<"FAIL during post-test sanity check for new items!: key: "<<key<<" exist?:"<<res<<" valbuffer:"<<val_buffer<<std::endl;
        std::cout<<"FAIL during post_test check(new)!: key: "<<std::hex<<key<<std::dec<<" exist?:"<<res<<" valbuffer:"<<std::hex<<val_buffer<<std::dec<<std::endl;
      }
  }
}

int main(int argc, char** argv) {
  ////////////////////////////////////////////////////////////////////////
  // parameters, arguments
  ////////////////////////////////////////////////////////////////////////
  install_backtrace_handler();
  if (argc != 7) {
    std::cout << "Usage: ./a.out TABLE_SIZE N_init_items N_new_items additional_reads_per_op N_threads use_custom"
              << std::endl;
  }
  int table_size= atoi(argv[1]);
  int num_init_items = atoi(argv[2]);
  int num_new_items= atoi(argv[3]);
  int additional_reads_per_op = atoi(argv[4]);
  int num_threads = atoi(argv[5]);
  int use_custom= atoi(argv[6]);
  std::cout<<"TABLE_SIZE "<<table_size<<" init: "<<num_init_items<<" new: "<<num_new_items<<" NT: "<<num_threads<<" additional_reads: "<<additional_reads_per_op<<" use_custom: "<<use_custom<<std::endl;
  ////////////////////////////////////////////////////////////////////////
  // initilaize hash table
  ////////////////////////////////////////////////////////////////////////
  hash_table* HT;
  switch(use_custom) {
    case 0:
    std::cout<<"baseline HT "<<use_custom<<std::endl;
    HT = new locked_probing_hash_table(table_size);
    break;
    case 1:
    std::cout<<"better HT "<<use_custom<<std::endl;
    HT = new better_locked_probing_hash_table(table_size);
    break;
  }
  std::cout << "start filling" << std::endl;
  std::chrono::duration<double> diff;
  auto fill_start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; t++) {
    threads.push_back(std::thread(init_worker, HT, t, num_threads, num_init_items));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto fill_end = std::chrono::steady_clock::now();
  diff = fill_end - fill_start;
  std::cout << "init hash table took " << diff.count() << " sec" << std::endl;
  ////////////////////////////////////////////////////////////////////////
  // test some queries
  ////////////////////////////////////////////////////////////////////////
  threads.clear();
  std::cout << "start test" << std::endl;
  auto test_start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_threads; t++) {
    threads.push_back(
        std::thread(test_worker, HT, t, num_threads, num_init_items, num_new_items, additional_reads_per_op));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto test_end = std::chrono::steady_clock::now();
  diff = test_end - test_start;
  std::cout << "test " << additional_reads_per_op * num_new_items<< " ops took " << diff.count()
            << " sec" << std::endl;
  ////////////////////////////////////////////////////////////////////////
  // sanity check 
  ////////////////////////////////////////////////////////////////////////
  auto chk_start = std::chrono::steady_clock::now();
  sequential_sanity_check(HT, num_init_items, num_new_items);
  auto chk_end = std::chrono::steady_clock::now();
  diff = chk_end - chk_start;
  //std::cout << "sanity check PASSED: "<<diff.count()<<" sec" << std::endl;
  std::cout << "sanity check PASSED: "<<std::endl;
  return 0;
}
