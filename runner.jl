# Script meant for editing and running as job on cluster

include("md/main.jl")

# @time resume_md("md/many_h2o/10h2o_0.1_temp.xyz", 1000; v_scale=1.04)
@time resume_md("md/many_h2o/10h2o_free_temp.xyz", 20000; Δt=10.0)

