"""Minimal jobshop example."""
import sys
import collections
from ortools.sat.python import cp_model
import torch
import numpy as np
import json
import pickle
import pandas as pd
def ortool(data=None,n_machine=5,n_job=5, time_limit = 10.0, result_print = False,data_no = None, mul_coef = 1000):
    """Open Shop Scheduling Problem with Arrival Time"""
    # Sample Data.
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2),(3,6),(4,8)],  # Job0
        [(0, 2), (1, 1), (2, 4),(3,3),(2,5)],  # Job1
        [(0,5),(1, 4), (2, 3),(3,3),(4,6)],
        [(0,5),(1, 4), (2, 3),(3,3),(4,6)],
        [(0,5),(1, 4), (2, 3),(3,3),(4,6)],  # Job2
    ]
    jobs_arival = [0 #Job0
                  ,1 #Job1
                  ,2,3,4] #Job2

    #Convert torch data into OR-tool interpretable format
    if data is not None:
        jobs_data = [ [] for _ in range(n_job)]
        jobs_arival = []
        #job_info = data[:-n_job,0].view(n_machine,n_job).permute(1,0)
        #ariv_info = data[-n_job:,0]
        ariv_info = data[0]
        job_info = data[1]
        for job_id, job in enumerate(job_info):
            jobs_arival.append( ariv_info[job_id].item())
            for machine_id, machine in enumerate(job):
                if machine.item() == 0: continue
                jobs_data[job_id].append((machine_id, machine.item()))

    print(ariv_info)
    print(job_info)
    print(jobs_data)
    print(jobs_arival)
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)
    pure_time = [ sum([task[1] for task in job] ) for job in jobs_data]


    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):

        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)



    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])
    for job_id, job in enumerate(jobs_data):
        model.AddNoOverlap([all_tasks[job_id, task_id].interval for task_id, _ in enumerate(job)])

    # Arival Time Constraints
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job)):
            model.Add(all_tasks[job_id, task_id].start >= jobs_arival[job_id])




    # Sum of Completion Time Objective
    job_completion_time = [ model.NewIntVar(0, horizon, 'com_time_{}'.format(job_id) ) for job_id, _ in enumerate(jobs_data)]
    # Is there any other clever way to get job completion time?
    for job_id, job in enumerate(jobs_data):
        model.AddMaxEquality(job_completion_time[job_id], [all_tasks[job_id, task_id].end for task_id, _ in enumerate(job) ] )

    # All of these objective functions are identical
    #model.Minimize(cp_model.LinearExpr.Sum(job_completion_time) - np.array(jobs_arival).sum() )
    #model.Minimize(cp_model.LinearExpr.Sum(job_completion_time))
    model.Minimize(cp_model.LinearExpr.Sum(job_completion_time) - np.array(jobs_arival).sum() - horizon)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    result = {}
    result['data_no'] = data_no
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE :
        #print(status == cp_model.OPTIMAL)
        #print('Solution:')
        pi = []

        result['is_optimal'] = (status == cp_model.OPTIMAL)
        result['is_feasible'] = (status == cp_model.FEASIBLE)
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            result['machine_{}'.format(machine)] = {}
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                pi.append(assigned_task.job + n_job * machine)
                name = 'job_%i_task_%i' % (assigned_task.job,
                                           assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                result['machine_{}'.format(machine)][name] = (start,duration)
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp


            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line


        # Finally print the solution found.
        if result_print :
            print(f'Optimal Schedule Length: {solver.ObjectiveValue()/n_job}')
            print(output)
        result['solution'] = solver.ObjectiveValue()/n_job
        result['sequence'] = pi
    else:
        print('No solution found.')

    # Statistics.
    if result_print :
        print('\nStatistics')
        print('  - conflicts: %i' % solver.NumConflicts())
        print('  - branches : %i' % solver.NumBranches())
        print('  - wall time: %f s' % solver.WallTime())
    result['wall_time'] = solver.WallTime()

    print(f'Optimal Schedule Length: {solver.ObjectiveValue()/n_job}')
    #return
    return result




if __name__ == '__main__':
    # with open('/data/project/hughnk/rl_thesis/scheduling_rl/data/osspa/osspa300_1520_seed1234.pkl', 'rb') as f:
    #     x49, mask = pickle.load(f)
    with open(sys.argv[1],'rb') as f:
        data = pickle.load(f)
#    with open('/data/project/hughnk/rl_thesis/scheduling_rl/data/osspa/osspa49_test_seed1234.pkl','rb') as g:
#        x49, mask = pickle.load(g)

#    with open('/data/project/hughnk/rl_thesis/scheduling_rl/data/osspa/osspa100_test_seed1234.pkl','rb') as g:
#        x100, mask = pickle.load(g)

#    with open('/data/project/hughnk/rl_thesis/scheduling_rl/data/osspa/osspa300_drop5_seed4353.pkl','rb') as g:
#        x300, mask = pickle.load(g)

    #print(data)
    final_result = []
    time_limit_list = [5,10]
    job_n = [len(data[0][0])]
    machine_n = [len(data[0][1])]
    dataset = [data]
#    print(len(x300))

    coef = 1
    result = pd.DataFrame(columns=['data_no','time_limit','cost','size'])
    for jn, mn, x in zip(job_n, machine_n, dataset):
        for time_limit in time_limit_list:
            for idx, data in enumerate(x):
                res = ortool(data,mn,jn,time_limit=time_limit,data_no=idx, mul_coef = coef)
                print(res)
                row = {'data_no': idx, 'time_limit': time_limit, 'cost': res['solution']/coef,'size': jn*mn}
                result = result.append(row, ignore_index= True)
                if idx % 10 == 0:
                    print('data {} cost {} on time limit {}'.format(idx,res['solution']/coef,time_limit ))
            print('time limit {} finished'.format(time_limit))
    result.to_csv('./ortool_result/result_300.csv')
