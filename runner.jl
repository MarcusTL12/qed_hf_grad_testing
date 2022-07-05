# Script meant for editing and running as job on cluster

include("md/main.jl")

@time resume_md("md/many_h2o/10h2o_0.1_freeze.xyz", 10000)
# @time resume_md("md/many_h2o/10h2o_free_temp.xyz", 20000; Δt=10.0)

