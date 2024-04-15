# PP HW 4 Report Template
> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## 1. Overview
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c)
1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    - `ucp_init`
    - `ucp_worker_create`
    - `ucp_ep_create`
2. What is the specific significance of the division of UCP Objects in the program? What important information do they carry?
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`
3. Based on the description in HW4, where do you think the following information is loaded/created?
    - `UCX_TLS`
    - TLS selected by UCX

## 2. Implementation

> Describe how you implemented the two special features of HW4.
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
2. How do the functions in these files call each other? Why is it designed this way?
3. Observe when Line 1 and 2 are printed during the call of which UCP API?
4. Does it match your expectations for questions **1-3**? Why?
5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.

## 3. Optimize System 
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```
-------------------------------------------------------------------
/opt/modulefiles/openmpi/4.1.5:

module-whatis   {Sets up environment for OpenMPI located in /opt/openmpi}
conflict        mpi
module          load ucx
setenv          OPENMPI_HOME /opt/openmpi
prepend-path    PATH /opt/openmpi/bin
prepend-path    LD_LIBRARY_PATH /opt/openmpi/lib
prepend-path    CPATH /opt/openmpi/include
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

> Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
```bash
module load openmpi/4.1.5
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/standard/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/standard/osu_bw
```

### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```


## 4. Experience & Conclusion
1. What have you learned from this homework?
2. Feedback (optional)
