# Script meant for editing and running as job on cluster

include("md/main.jl")

# @time resume_md("md/many_h2o/10h2o_free.xyz", 100, 40)

@time resume_md("md/many_h2o/10h2o_0.1.xyz", 100, 44)

