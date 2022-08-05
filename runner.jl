# Script meant for editing and running as job on cluster

# include("numgrad/main.jl")
include("md/main.jl")

curfile = "md/many_h2o/30h2o_free.xyz"
# curfile = "md/many_h2o/30h2o_0.1.xyz"
# curfile = "md/many_h2o/50h2o_free.xyz"
# curfile = "md/many_h2o/20h2o_free.xyz"

# @time resume_md(curfile, 10000)
@time keep_temp(curfile, 275, 1000, 600)
