 #########################
 # Environment variable
 #########################
 build=build
 src=src

exec:
	g++ -O3 -g $(build)/main.o $(build)/similarity_algorithm_sequential.o $(build)/smith_waterman_sequential.o $(src)/similarity_algorithm_parallel.cc $(src)/smith_waterman_parallel.cc -o sw -fopenmp -march=znver2
  
run:
	./sw 950 950

clean:
	rm sw *_prl *_seq *.err *.log *.out

# Condor

remote:
	condor_submit HW4_smith_waterman.cmd

report:
	cat smith_waterman.out

queue:
	condor_q

statue:
	condor_status

remove:
	condor_rm
