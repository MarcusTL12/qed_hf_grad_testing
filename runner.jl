# Script meant for editing and running as job on cluster

include("md/main.jl")

# @time resume_md("md/many_h2o/10h2o_0.1_temp.xyz", 100; v_scale=1.1)
@time resume_md("md/many_h2o/10h2o_free_temp.xyz", 100; v_scale=1.142)

