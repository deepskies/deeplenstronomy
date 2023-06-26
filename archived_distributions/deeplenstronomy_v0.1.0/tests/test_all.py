# Main tester

import os
os.system('rm -rf TestResults') # fresh start


kwargs_sets = {0: {}, # default arguments
               1: {'save_to_disk': True},
               2: {'save_to_disk': True, 'image_file_format': 'h5'},
               3: {'save_to_disk': True, 'skip_image_generation': True},
               4: {'store_in_memory': False},
               5: {'store_sample': True},
               6: {'skip_image_generation': True, 'survey': 'des'},
               7: {'solve_lens_equation': True},
               8: {'return_planes': True}
}

# Run all tests by writing last test to a file
try:
    f = open('status.txt', 'r')
    previous_test = int(f.read().strip())
    f.close()
except FileNotFoundError:
    previous_test = -1
    
next_test = previous_test + 1
next_test = next_test if next_test in kwargs_sets.keys() else 0

# Update the status file
f = open('status.txt', 'w+')
f.write(str(next_test))
f.close()

# Run tests
os.system("pytest test_existance_and_attributes.py -v --capture=tee-sys")
os.system("pytest test_expected_outputs.py -v --capture=tee-sys")
os.system("pytest test_expected_behaviors_fixed.py -v --capture=tee-sys")
os.system("pytest test_expected_behaviors_configurations.py -v --capture=tee-sys")




