#!/bin/bash

outfile="./run_all.log"

run_start() {
    echo "Run:" $(date -u) > $outfile
    echo >> $outfile
}

run_line() {
    echo python $* "..."

    echo "python" $* "..." >> $outfile
    echo >> $outfile
    python $* >> $outfile
    echo >> $outfile
}

run_end() {
    echo "Done."
}

run_start

### test_run_transformers combinations
run_line any_test_run_transformers.py
run_line any_test_run_transformers.py --modname mvitv2 --run-name mvitv2_test
run_line any_test_run_transformers.py --use-dc2 --data-dir "./tests/deepdisc/test_data/dc2/"
run_line any_test_run_transformers.py --use-dc2 --data-dir "./tests/deepdisc/test_data/dc2/" --modname mvitv2 --run-name mvitv2_test
run_line any_test_run_transformers.py --use-dc2 --use-redshift --data-dir "./tests/deepdisc/test_data/dc2/"
run_line any_test_run_transformers.py --use-dc2 --use-redshift --data-dir "./tests/deepdisc/test_data/dc2/" --modname mvitv2  --run-name mvitv2_test


### test_eval_model combinations
run_line any_test_eval_model.py
run_line any_test_eval_model.py --run-name mvitv2_test
run_line any_test_eval_model.py --use-dc2 --testfile "./tests/deepdisc/test_data/dc2/single_test.json"


### Not working:

# The redshift version here could use some looking at. I inferred it from the
# corresoponding test_eval_DC2_redshift, but had to add:
#   - cfg.model.roi_heads.zloss_factor = 1.0 (arbitrarily choosing 1.0 here)
#   - adding the 3rd expected value from get_matched_z_pdfs.
# The existence of new_get_matched_z_pdfs makes me think the original script
# use a revisit, so there may be some outdated things I've copied over.
# Currently getting:
#   File ".../python3.9/site-packages/detectron2/structures/instances.py", line 66, in __getattr__
#     raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
#   AttributeError: Cannot find field 'pred_redshift_pdf' in the given Instances!

#run_line any_test_eval_model.py --use-dc2 --use-redshift --testfile "./tests/deepdisc/test_data/dc2/single_test.json"


# (RuntimeError: The size of tensor a (6) must match the size of tensor b (3) at non-singleton dimension 0)

#run_line any_test_eval_model.py --use-dc2 --testfile "./tests/deepdisc/test_data/dc2/single_test.json" --run-name mvitv2_test

run_end