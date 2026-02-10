from lab.environments import BaselSlurmEnvironment

class BWUniEnvironment(BaselSlurmEnvironment):
    def __init__(self, email="glenn.zundel@stud.uni-heidelberg.de",
                 partition="cpu_il", **kwargs):
        extra_options = (
            f"#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --mail-user={email}\n"
            "#SBATCH --time=02:00:00\n"
            "#SBATCH --cpus-per-task=1"
        )
        super().__init__(
            email=email, partition=partition, extra_options=extra_options,
            time_limit_per_task="2:00:00", cpus_per_task=1, **kwargs
        )
    def _get_job_header(self, step, is_last):
        header = super()._get_job_header(step, is_last)
        lines = header.split('\n')
        lines = [l for l in lines if not l.startswith('#SBATCH --qos=')]
        return '\n'.join(lines)
