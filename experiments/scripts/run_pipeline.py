import subprocess
import sys
import os
import clingo
from clingo.script import enable_python

enable_python()
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

def run_clingo_incremental(domain_file, problem_file, framework_file, output_file, run_with_stats, max_time=1500):
    ctl = clingo.Control([
     "--configuration=crafty",
     "--restarts=L,100",
     "--save-progress=10"
 ])
    ctl.load(domain_file)
    ctl.load(problem_file)
    ctl.load(framework_file)
    ctl.ground([("base", [])])
    for t in range(0, max_time + 1):
        ctl.ground([("step", [clingo.Number(t)])])
        ctl.ground([("check", [clingo.Number(t)])])
        ctl.assign_external(clingo.Function("query", [clingo.Number(t)]), True)
        result = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                result = [str(atom) for atom in model.symbols(shown=True)]
            solve_result = handle.get()
            if run_with_stats:
                stats = ctl.statistics
                atoms = stats['problem']['lp']['atoms']
                rules = stats['problem']['lp']['rules']
                totalTime = stats['summary']['times']['total']
                cpuTime = stats['summary']['times']['cpu']
                print(f"atoms: {atoms}, rules: {rules}, totalTime: {totalTime}, cpuTime: {cpuTime}")
        if solve_result.satisfiable:
            model_str = "\n".join(sorted(result))
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(model_str)
            return t, True, model_str
        ctl.assign_external(clingo.Function("query", [clingo.Number(t)]), False)
        print(f"t={t}: UNSAT, versuche t={t+1}...")
    return max_time, False, None

def run_workflow(domain_file, problem_file, framework_file,
                 domain_output, problem_output, primitives_output,
                 clingo_output, tasklist_output):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hddl_to_lp_script = os.path.join(script_dir, "hddl_to_lp.py")
    parse_result_script = os.path.join(script_dir, "parseResult.py")
    print("Übersetzung")
    subprocess.run(
        ["python3", hddl_to_lp_script, domain_file, problem_file,
         domain_output, problem_output, primitives_output],
        check=True
    )
    print("Clingo")
    time_bound, success, model_str = run_clingo_incremental(
        domain_file=domain_output, problem_file=problem_output,
        framework_file=framework_file, output_file=clingo_output,
        run_with_stats=False, max_time=1500
    )
    print('Primitive Tasks extrahieren')
    subprocess.run(
        ["python3", parse_result_script, primitives_output,
         clingo_output, tasklist_output],
        check=True
    )


if __name__ == "__main__":

    with open('slurm_info.txt', 'w') as f:
        f.write(f"SLURM_JOB_ID={SLURM_JOB_ID}\n")
    run_workflow(
        domain_file=sys.argv[1], problem_file=sys.argv[2],
        framework_file=sys.argv[3], domain_output=sys.argv[4],
        problem_output=sys.argv[5], primitives_output=sys.argv[6],
        clingo_output=sys.argv[7], tasklist_output=sys.argv[8]
    )
