#!/usr/bin/env python3
import sys
from pathlib import Path
from lab.experiment import Experiment
from lab.reports import Attribute, Report
from lab.environments import LocalEnvironment

exp_dir = Path(__file__).resolve().parent
benchmarks = exp_dir / "benchmarks"
framework = exp_dir / "framework"
scripts = exp_dir / "scripts"

env = LocalEnvironment(processes=2)
exp = Experiment(environment=env)

SUITE = [('assemblyhierarchical', 'problem01.hddl'),
        ('assemblyhierarchical', 'problem02.hddl'),
        ('assemblyhierarchical', 'problem03.hddl'),
         ('barman-bdi', 'problem01.hddl'),
        ('barman-bdi', 'problem02.hddl'),
        ('depots', 'problem01.hddl'),
        ('depots', 'problem02.hddl'),
        ('depots', 'problem03.hddl'),
        ('depots', 'problem04.hddl'),
        ('depots', 'problem05.hddl'),
        ('depots', 'problem06.hddl'),
        ('factories-simple', 'problem01.hddl'),
        ('factories-simple', 'problem02.hddl'),
        ('hiking', 'problem01.hddl'),
        ('robot', 'problem01.hddl'),
        ('robot', 'problem02.hddl'),
        ('robot', 'problem03.hddl'),
        ('robot', 'problem04.hddl'),
        ('robot', 'problem05.hddl'),
        ('rover-gtohp', 'problem01.hddl'),
        ('rover-gtohp', 'problem02.hddl'),
        ('rover-gtohp', 'problem03.hddl'),
        ('rover-gtohp', 'problem04.hddl'),
        ('rover-gtohp', 'problem05.hddl'),
        ('satellite', 'problem01.hddl'),
        ('satellite', 'problem02.hddl'),
        ('satellite', 'problem03.hddl'),
        ('satellite', 'problem04.hddl'),
        ('towers', 'problem01.hddl'),
        ('towers', 'problem02.hddl'),
        ('towers', 'problem03.hddl'),
        ('towers', 'problem04.hddl'),
        ('towers', 'problem05.hddl'),
        ('towers', 'problem06.hddl'),
         ]

for domain, problem in SUITE:
    prob_name = problem.replace(".hddl", "")
    run = exp.add_run()
    run.set_property("id", [domain, prob_name])
    run.set_property("domain", domain)
    run.set_property('problem', prob_name)
    run.add_resource("domain", benchmarks / domain / "domain.hddl", "domain.hddl")
    run.add_resource("problem", benchmarks / domain / problem, "problem.hddl")
    run.add_resource("framework", framework / "framework.lp", "framework.lp")
    run.add_resource("run_pipeline", scripts / "run_pipeline.py", "run_pipeline.py")
    run.add_resource("hddl_to_lp", scripts / "hddl_to_lp.py", "hddl_to_lp.py")
    run.add_resource("parseResult", scripts / "parseResult.py", "parseResult.py")
    run.add_command(
        "run-pipeline",
        [sys.executable, "{run_pipeline}", "domain.hddl", "problem.hddl", "framework.lp",
         "domain_output.lp", "problem_output.lp", "primitives.txt", "clingo_output.txt", "orderedtasklist.txt"],
        time_limit=1800, memory_limit=8100
    )

attributes = [
    Attribute("domain", absolute=False),
    Attribute("problem", absolute=False),
    Attribute("time", absolute=False, min_wins=True),
    Attribute('memory', absolute=False, min_wins=True),
]
report = Report(attributes=attributes, format="html")
exp.add_report(report, outfile="results.html")
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.run_steps()
