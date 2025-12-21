import subprocess
import sys
import os
import tempfile

# Example usage:
# python3 run_pipeline.py domain.hddl problem.hddl framework.lp domain_output.lp problem_output.lp primitives.txt clingo_output.txt orderedtasklist.txt


def create_temp_framework(time_bound):
    """Create temporary framework file with given time bound.

    Args:
        time_bound: Maximum time step (creates time(0..time_bound))

    Returns:
        Path to the temporary framework file
    """
    content = f"""time(0..{time_bound}).

in_state(A, T+1) :-
    time(T),
    in_state(A, T),
    not out_state(A, T+1).

0 {{ taskTBA(A, T) : action(A) }} 1 :- time(T).

:- not plan_found.

#show taskTBA/2.
"""
    # Create temp file that persists until explicitly deleted
    fd, temp_path = tempfile.mkstemp(suffix='.lp', prefix=f'framework_t{time_bound}_')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return temp_path


def run_clingo_iterative(clingo_inputs, output_file, max_time=300, start_time=1, step=1):
    """Run Clingo with iterative deepening on time bound.

    Args:
        clingo_inputs: List of input files (domain, problem) - framework will be replaced
        output_file: Path to write clingo output
        max_time: Maximum time bound to try
        start_time: Starting time bound
        step: Increment for each iteration

    Returns:
        Tuple of (time_bound, success, returncode)
    """
    # Filter out any existing framework file
    inputs_without_framework = [f for f in clingo_inputs if 'framework' not in f.lower()]

    temp_framework = None

    try:
        for time_bound in range(start_time, max_time + 1, step):
            # Clean up previous temp file
            if temp_framework and os.path.exists(temp_framework):
                os.remove(temp_framework)

            # Create new framework with current time bound
            temp_framework = create_temp_framework(time_bound)

            # Run clingo
            current_inputs = inputs_without_framework + [temp_framework]
            result = subprocess.run(
                ["clingo"] + current_inputs,
                capture_output=True,
                text=True
            )

            # Clingo Return Codes:
            # 10 = SAT (satisfiable)
            # 20 = UNSAT (unsatisfiable)
            # 30 = SAT + optimum proven

            if result.returncode in [10, 30]:
                # Solution found!
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                print(f"Lösung gefunden mit time_bound={time_bound}!")
                return time_bound, True, result.returncode

            if result.returncode == 20:
                # UNSAT - try next time bound
                print(f"time_bound={time_bound}: UNSAT, versuche {time_bound + step}...")
            else:
                # Other error
                print(f"time_bound={time_bound}: Clingo returncode={result.returncode}")
                if result.stderr:
                    print(f"  Fehler: {result.stderr}")

        # Max time reached without solution
        print(f"Keine Lösung gefunden bis time_bound={max_time}")
        return max_time, False, 20

    finally:
        # Clean up temp file
        if temp_framework and os.path.exists(temp_framework):
            os.remove(temp_framework)


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
            ["python3", hddl_to_lp_script, domain_file, problem_file,
             domain_output, problem_output, primitives_output],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei hddl_to_lp: {e}")
        return False
    except FileNotFoundError:
        print(f"Fehler: Script '{hddl_to_lp_script}' nicht gefunden.")
        return False

    print("\n--- 2. Starte Clingo (Iterative Deepening) ---")

    # Check if required files exist
    clingo_inputs = [problem_output, domain_output]
    for f in clingo_inputs:
        if not os.path.exists(f):
            print(f"Fehler: Die Datei '{f}' wurde nicht gefunden.")
            return False

    try:
        time_bound, success, returncode = run_clingo_iterative(
            clingo_inputs=clingo_inputs,
            output_file=clingo_output,
            max_time=300,
            start_time=5,
            step=1
        )

        if not success:
            print(f"Clingo: Keine Lösung gefunden (max_time={time_bound})")
        else:
            print(f"Clingo fertig. Optimale Planlänge: {time_bound}. Ergebnis in '{clingo_output}' gespeichert.")
    except FileNotFoundError:
        print("Fehler: 'clingo' wurde nicht gefunden. Ist es installiert und im PATH?")
        return False

    print("\n--- 3. Verarbeite Ergebnisse (parseResult) ---")
    try:
        subprocess.run(
            ["python3", parse_result_script, primitives_output,
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
