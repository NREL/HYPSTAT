'''
Authors: Joe Brauch, Yijin Li, Steven Percy

This file contains a demonstration of how a simple model could be run using the Case Study included here.
'''

from HYPSTAT import HYPSTAT

model = HYPSTAT(yaml_file_path='Case_Study/Case_Study_Scenario.yaml')
model.two_step_solve(solver='glpk')
model.write_outputs('Case_Study/Outputs/demo_run')
print('Done!')