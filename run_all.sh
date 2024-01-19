#!/bin/bash

run_line() {
    echo "--------------------------------------------------------------------------------"
    echo python $*; echo
    python $*
}

#run_line any_test_run_transformers.py
#run_line any_test_run_transformers.py --modname mvitv2 --run-name mvitv2_test
#run_line any_test_run_transformers.py --use-dc2 --data-dir "./tests/deepdisc/test_data/dc2/"
#run_line any_test_run_transformers.py --use-dc2 --data-dir "./tests/deepdisc/test_data/dc2/" --modname mvitv2 --run-name mvitv2_test
#run_line any_test_run_transformers.py --use-dc2 --use-redshift --data-dir "./tests/deepdisc/test_data/dc2/"
#run_line any_test_run_transformers.py --use-dc2 --use-redshift --data-dir "./tests/deepdisc/test_data/dc2/" --modname mvitv2  --run-name mvitv2_test

#run_line any_test_eval_model.py
#run_line any_test_eval_model.py --run-name mvitv2_test
#run_line any_test_eval_model.py --use-dc2 --testfile "./tests/deepdisc/test_data/dc2/single_test.json"



# Not working:
# run_line any_test_eval_model.py --use-dc2 --testfile "./tests/deepdisc/test_data/dc2/single_test.json" --run-name mvitv2_test
# RuntimeError: The size of tensor a (6) must match the size of tensor b (3) at non-singleton dimension 0