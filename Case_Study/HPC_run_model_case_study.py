from model_object import HYPSTAT
years = [2030, 2040, 2050]
for year in years:
    print('### Running model for year {y} ###'.format(y=year))
    print()
    test = HYPSTAT(yaml_file_path='Case_Study_Scenario_HPC.yaml',year=year)
    test.two_step_solve(solver='gurobi')
    test.write_outputs('Outputs/Case_Study')
    print('Done!')
