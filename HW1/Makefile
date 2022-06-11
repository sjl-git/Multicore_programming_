# 1D filter 
run: filter 
	./filter 1073741824

filter: filter.cc 
	g++ -std=c++11 -O3 -g -pthread  $< -o $@ -lboost_system  -lboost_thread  -march=znver2

clean:
	rm *.txt filter

# Condor
remote: filter
	condor_submit hw1.cmd

queue:
	condor_q

status:
	condor_status

remove:
	condor_rm
