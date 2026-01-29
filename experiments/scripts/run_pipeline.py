import subprocess
import sys
import os
import clingo
from clingo.script import enable_python

enable_python()

# SLURM Job-ID erfassen (für sacct-Abfrage nach dem Experiment)
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

# Example usage:
# python3 run_pipeline.py domain.hddl problem.hddl framework.lp domain_output.lp problem_output.lp primitives.txt clingo_output.txt orderedtasklist.txt


def run_clingo_incremental(domain_file, problem_file, framework_file, output_file, run_with_stats, max_time=1500):
    """Run Clingo with incremental solving using the Python API.

    Args:
        domain_file: Translated ASP domain file
        problem_file: Translated ASP problem file
        framework_file: Framework file with #program directives
        output_file: Output file for result
        run_with_stats: Output statistics for each timestep
        max_time: Maximum time steps

    Returns:
        Tuple (time_bound, success, model_str)
    """
    ctl = clingo.Control([
     "--configuration=crafty",   # Besser für strukturierte/crafted Probleme
     "--restarts=L,100",         # Luby-Restarts für schwierige Probleme
     "--save-progress=10"        # Cache Wahrheitswerte bei Backjumps
 ])

    # Load all files
    ctl.load(domain_file)
    ctl.load(problem_file)
    ctl.load(framework_file)

    # 1. Ground base (initial state, types, objects, time(0))
    ctl.ground([("base", [])])

    for t in range(0, max_time + 1):
        # 2. Ground step for current time step
        ctl.ground([("step", [clingo.Number(t)])])

        # 3. Ground check block
        ctl.ground([("check", [clingo.Number(t)])])

        # 4. Activate external query(t)
        ctl.assign_external(clingo.Function("query", [clingo.Number(t)]), True)

        # 5. Solve
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
            # Solution found
            model_str = "\n".join(sorted(result))
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(model_str)
            return t, True, model_str

        # 6. Deactivate external for next iteration
        ctl.assign_external(clingo.Function("query", [clingo.Number(t)]), False)

        print(f"t={t}: UNSAT, versuche t={t+1}...")

    return max_time, False, None


def run_workflow(domain_file, problem_file, framework_file,
                 domain_output, problem_output, primitives_output,
                 clingo_output, tasklist_output):
    """
    Run the complete HTN planning workflow.

    Args:
        domain_file: Input HDDL domain file
        problem_file: Input HDDL problem file
        framework_file: Input ASP framework file
        domain_output: Output ASP domain file
        problem_output: Output ASP problem file
        primitives_output: Output primitives list file
        clingo_output: Output file for clingo results
        tasklist_output: Output file for ordered task list
    """

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hddl_to_lp_script = os.path.join(script_dir, "hddl_to_lp.py")
    parse_result_script = os.path.join(script_dir, "parseResult.py")

    print("--- 1. Starte Übersetzung (hddl_to_lp) ---")
    try:
        subprocess.run(
            ["py", hddl_to_lp_script, domain_file, problem_file,
             domain_output, problem_output, primitives_output],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei hddl_to_lp: {e}")
        return False
    except FileNotFoundError:
        print(f"Fehler: Script '{hddl_to_lp_script}' nicht gefunden.")
        return False

    print("\n--- 2. Starte Clingo (Inkrementelles Solving) ---")

    # Check if required files exist
    required_files = [domain_output, problem_output, framework_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Fehler: Die Datei '{f}' wurde nicht gefunden.")
            return False

    try:
        time_bound, success, model_str = run_clingo_incremental(
            domain_file=domain_output,
            problem_file=problem_output,
            framework_file=framework_file,
            output_file=clingo_output,
            run_with_stats=False,
            max_time=1500
        )

        if not success:
            print(f"Clingo: Keine Lösung gefunden (max_time={time_bound})")
        else:
            print(f"Clingo fertig. Plan gefunden bei t={time_bound}. Ergebnis in '{clingo_output}' gespeichert.")
    except Exception as e:
        print(f"Fehler bei Clingo: {e}")
        return False

    print("\n--- 3. Verarbeite Ergebnisse (parseResult) ---")
    try:
        subprocess.run(
            ["py", parse_result_script, primitives_output,
             clingo_output, tasklist_output],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei parseResult: {e}")
        return False
    except FileNotFoundError:
        print(f"Fehler: Script '{parse_result_script}' nicht gefunden.")
        return False

    print("\n--- Workflow abgeschlossen ---")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: run_pipeline.py <domain> <problem> <framework> <domain_output> <problem_output> <primitives_output> <clingo_output> <tasklist_output>")
        print("\nArguments:")
        print("  <domain>             - Input HDDL domain file")
        print("  <problem>            - Input HDDL problem file")
        print("  <framework>          - Input ASP framework file")
        print("  <domain_output>      - Output ASP domain file")
        print("  <problem_output>     - Output ASP problem file")
        print("  <primitives_output>  - Output primitives list file")
        print("  <clingo_output>      - Output file for clingo results")
        print("  <tasklist_output>    - Output file for ordered task list")
        print("\nExample:")
        print("  python3 run_pipeline.py domain.hddl problem.hddl framework.lp \\")
        print("                          domain_out.lp problem_out.lp primitives.txt \\")
        print("                          clingo_out.txt tasklist.txt")
        sys.exit(1)

    with open('slurm_info.txt', 'w') as f:
        f.write(f"SLURM_JOB_ID={SLURM_JOB_ID}\n")

    success = run_workflow(
        domain_file=sys.argv[1],
        problem_file=sys.argv[2],
        framework_file=sys.argv[3],
        domain_output=sys.argv[4],
        problem_output=sys.argv[5],
        primitives_output=sys.argv[6],
        clingo_output=sys.argv[7],
        tasklist_output=sys.argv[8]
    )

    sys.exit(0 if success else 1)
