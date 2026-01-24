#!/usr/bin/env python3
"""
HTN Planning Experiment for bwUniCluster

This script sets up and executes a Fast Downward Lab experiment for running
HTN planning benchmarks through the HDDL → ASP → Clingo → Result parsing pipeline.

The experiment runs on the bwUniCluster SLURM system with the following flow:
1. HDDL domain and problem files are translated to ASP
2. Clingo solves the ASP encoding
3. Results are parsed into an ordered task list

Each run executes the complete pipeline via run_pipeline.py with:
- 1 CPU core per run
- 8GB RAM per run
- 30 minutes time limit
"""

import sys
from pathlib import Path
from lab.experiment import Experiment
from lab.reports import Attribute, Report

from bwuni_environment import BWUniEnvironment

# ============================================================================
# Path Configuration
# ============================================================================

EXPERIMENT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = EXPERIMENT_DIR / "benchmarks"
FRAMEWORK_DIR = EXPERIMENT_DIR / "framework"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"

# ============================================================================
# Environment Configuration
# ============================================================================

env = BWUniEnvironment(
    email="glenn.zundel@stud.uni-heidelberg.de",
    memory_per_cpu="8100M",
)

# ============================================================================
# Experiment Setup
# ============================================================================

# Create the experiment with environment
exp = Experiment(environment=env)

# Define benchmark suite: (domain_name, problem_file)
SUITE = [
    # Hiking (10 problems)
    ("hiking", "problem01.hddl"),
    ("hiking", "problem02.hddl"),
    ("hiking", "problem03.hddl"),
    ("hiking", "problem04.hddl"),
    ("hiking", "problem05.hddl"),
    ("hiking", "problem06.hddl"),
    ("hiking", "problem07.hddl"),
    ("hiking", "problem08.hddl"),
    ("hiking", "problem09.hddl"),
    ("hiking", "problem10.hddl"),
    # Depots (10 problems)
    ("depots", "problem01.hddl"),
    ("depots", "problem02.hddl"),
    ("depots", "problem03.hddl"),
    ("depots", "problem04.hddl"),
    ("depots", "problem05.hddl"),
    ("depots", "problem06.hddl"),
    ("depots", "problem07.hddl"),
    ("depots", "problem08.hddl"),
    ("depots", "problem09.hddl"),
    ("depots", "problem10.hddl"),
    # Satellite (10 problems)
    ("satellite", "problem01.hddl"),
    ("satellite", "problem02.hddl"),
    ("satellite", "problem03.hddl"),
    ("satellite", "problem04.hddl"),
    ("satellite", "problem05.hddl"),
    ("satellite", "problem06.hddl"),
    ("satellite", "problem07.hddl"),
    ("satellite", "problem08.hddl"),
    ("satellite", "problem09.hddl"),
    ("satellite", "problem10.hddl"),
    # Barman-BDI (10 problems)
    ("barman-bdi", "problem01.hddl"),
    ("barman-bdi", "problem02.hddl"),
    ("barman-bdi", "problem03.hddl"),
    ("barman-bdi", "problem04.hddl"),
    ("barman-bdi", "problem05.hddl"),
    ("barman-bdi", "problem06.hddl"),
    ("barman-bdi", "problem07.hddl"),
    ("barman-bdi", "problem08.hddl"),
    ("barman-bdi", "problem09.hddl"),
    ("barman-bdi", "problem10.hddl"),
    # AssemblyHierarchical (10 problems)
    ("assemblyhierarchical", "problem01.hddl"),
    ("assemblyhierarchical", "problem02.hddl"),
    ("assemblyhierarchical", "problem03.hddl"),
    ("assemblyhierarchical", "problem04.hddl"),
    ("assemblyhierarchical", "problem05.hddl"),
    ("assemblyhierarchical", "problem06.hddl"),
    ("assemblyhierarchical", "problem07.hddl"),
    ("assemblyhierarchical", "problem08.hddl"),
    ("assemblyhierarchical", "problem09.hddl"),
    ("assemblyhierarchical", "problem10.hddl"),
    # Blocksworld-GTOHP (10 problems)
    ("blocksworld-gtohp", "problem01.hddl"),
    ("blocksworld-gtohp", "problem02.hddl"),
    ("blocksworld-gtohp", "problem03.hddl"),
    ("blocksworld-gtohp", "problem04.hddl"),
    ("blocksworld-gtohp", "problem05.hddl"),
    ("blocksworld-gtohp", "problem06.hddl"),
    ("blocksworld-gtohp", "problem07.hddl"),
    ("blocksworld-gtohp", "problem08.hddl"),
    ("blocksworld-gtohp", "problem09.hddl"),
    ("blocksworld-gtohp", "problem10.hddl"),
    # Rover-GTOHP (10 problems)
    ("rover-gtohp", "problem01.hddl"),
    ("rover-gtohp", "problem02.hddl"),
    ("rover-gtohp", "problem03.hddl"),
    ("rover-gtohp", "problem04.hddl"),
    ("rover-gtohp", "problem05.hddl"),
    ("rover-gtohp", "problem06.hddl"),
    ("rover-gtohp", "problem07.hddl"),
    ("rover-gtohp", "problem08.hddl"),
    ("rover-gtohp", "problem09.hddl"),
    ("rover-gtohp", "problem10.hddl"),
    # Factories-simple (10 problems)
    ("factories-simple", "problem01.hddl"),
    ("factories-simple", "problem02.hddl"),
    ("factories-simple", "problem03.hddl"),
    ("factories-simple", "problem04.hddl"),
    ("factories-simple", "problem05.hddl"),
    ("factories-simple", "problem06.hddl"),
    ("factories-simple", "problem07.hddl"),
    ("factories-simple", "problem08.hddl"),
    ("factories-simple", "problem09.hddl"),
    ("factories-simple", "problem10.hddl"),
    # Towers (10 problems)
    ("towers", "problem01.hddl"),
    ("towers", "problem02.hddl"),
    ("towers", "problem03.hddl"),
    ("towers", "problem04.hddl"),
    ("towers", "problem05.hddl"),
    ("towers", "problem06.hddl"),
    ("towers", "problem07.hddl"),
    ("towers", "problem08.hddl"),
    ("towers", "problem09.hddl"),
    ("towers", "problem10.hddl"),
    # Transport (10 problems)
    ("transport", "problem01.hddl"),
    ("transport", "problem02.hddl"),
    ("transport", "problem03.hddl"),
    ("transport", "problem04.hddl"),
    ("transport", "problem05.hddl"),
    ("transport", "problem06.hddl"),
    ("transport", "problem07.hddl"),
    ("transport", "problem08.hddl"),
    ("transport", "problem09.hddl"),
    ("transport", "problem10.hddl"),
]

# ============================================================================
# Run Configuration
# ============================================================================

for domain, problem in SUITE:
    # Generate run identifier
    problem_name = problem.replace(".hddl", "")
    run_id = f"{domain}-{problem_name}"

    # Create run
    run = exp.add_run()
    run.set_property("id", [domain, problem_name])
    run.set_property("domain", domain)
    run.set_property("problem", problem_name)

    # Add benchmark resources
    run.add_resource(
        "domain",
        BENCHMARKS_DIR / domain / "domain.hddl",
        "domain.hddl"
    )
    run.add_resource(
        "problem",
        BENCHMARKS_DIR / domain / problem,
        "problem.hddl"
    )

    # Add framework resource
    run.add_resource(
        "framework",
        FRAMEWORK_DIR / "framework.lp",
        "framework.lp"
    )

    # Add script resources
    run.add_resource(
        "run_pipeline",
        SCRIPTS_DIR / "run_pipeline.py",
        "run_pipeline.py"
    )
    run.add_resource(
        "hddl_to_lp",
        SCRIPTS_DIR / "hddl_to_lp.py",
        "hddl_to_lp.py"
    )
    run.add_resource(
        "parseResult",
        SCRIPTS_DIR / "parseResult.py",
        "parseResult.py"
    )

    # Add pipeline command
    # The run_pipeline.py script orchestrates:
    # 1. hddl_to_lp.py: HDDL → ASP translation
    # 2. clingo: ASP solving
    # 3. parseResult.py: Result extraction
    run.add_command(
        "run-pipeline",
        [
            sys.executable, "{run_pipeline}",
            "domain.hddl",           # arg1: domain input
            "problem.hddl",          # arg2: problem input
            "framework.lp",          # arg3: framework input
            "domain_output.lp",      # arg4: domain ASP output
            "problem_output.lp",     # arg5: problem ASP output
            "primitives.txt",        # arg6: primitives list
            "clingo_output.txt",     # arg7: clingo raw output
            "orderedtasklist.txt"    # arg8: final task list
        ],
        time_limit=1800,              # 30 minutes (in seconds)
        memory_limit=8100             # 8GB RAM
    )

# ============================================================================
# Reporting Configuration
# ============================================================================

# Define attributes to track and report
attributes = [
    Attribute("domain", absolute=False),
    Attribute("problem", absolute=False),
    Attribute("time", absolute=False, min_wins=True),
    Attribute("memory", absolute=False, min_wins=True),
]

# Create HTML report
report = Report(attributes=attributes, format="html")
exp.add_report(report, outfile="results.html")

# ============================================================================
# Execution Steps
# ============================================================================

# Add execution steps for SLURM workflow:
# 1. build  - Create run directories and copy resources
# 2. start  - Submit SLURM jobs
# 3. fetch  - Fetch results after jobs complete (optional, for remote clusters)
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)

# Parse command line arguments and execute requested steps
# Usage:
#   python3 experiment.py build       # Build the experiment
#   python3 experiment.py start       # Submit SLURM jobs
#   python3 experiment.py results.html # Generate report (after jobs finish)
#   python3 experiment.py --all       # Run all steps sequentially
exp.run_steps()
