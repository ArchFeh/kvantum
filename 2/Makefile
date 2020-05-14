all:
	mpicxx main.cpp -o prog
submit:
	for n in 20 24 28 30; do \
    		for k in 1 3 $$n; do \
    			for i in 1 2 4 8 160; do \
			mpisubmit.pl -w 00:15  -p $$i prog -- $$n $$k; \
    			done \
    		done \
    	done
