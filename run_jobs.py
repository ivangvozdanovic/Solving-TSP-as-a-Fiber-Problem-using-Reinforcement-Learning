from run_q_learning import run_q_learning_instance


# Increase the job registry if you want more jobs to run parallel.
job_registry = {
    "tsp_sub1": run_q_learning_instance,
    "tsp_sub2": run_q_learning_instance,
    "tsp_sub3": run_q_learning_instance,
}



def run_Q_wrapper(task_config):
    
    job_name = task_config["job"]
    job_func = job_registry.get(job_name)
    
    if job_func is None:
        raise ValueError(f"Unknown job type: {job_name}")
        
    job_func(**task_config["params"])