# HT Test
run: HTtest
    #std::"Usage: ./a.out TABLE_SIZE N_init_items N_new_items additional_reads_per_op N_threads use_baseline"
	./HTtest 10000000 4000000 4000000 9 16 1

HTtest: main.cc 
	g++ -std=c++11 -g -pthread  $< -o $@ -lboost_system  -lboost_thread

clean:
	rm -f HTtest *.err *.log *.out

# Condor
remote_base: HTtest
	condor_submit HW2_base.cmd

remote_better: HTtest
	condor_submit HW2_better.cmd

report:
	cat base.out
	cat better.out

queue:
	condor_q

status:
	condor_status

remove:
	condor_rm

# Development
format:
	clang-format -i -style=Google *.cc *.h
