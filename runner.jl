# Script meant for editing and running as job on cluster

# include("numgrad/main.jl")
include("md/main.jl")

curfile = "md/many_h2o/20h2o_free.xyz"

# resume_md(curfile, 50)

# let avg = get_avg_last_T(curfile, 20)

#     while !isfile("stop")
#         if avg < 275
#             println("raising temp from $avg")
#             resume_md(curfile, 50; v_scale=min(√(275 / avg), 1.2))
#         else
#             println("temp is fine")
#             resume_md(curfile, 50)
#         end
#         avg = get_avg_last_T(curfile, 20)
#     end

# end

@time resume_md(curfile, 10000)
# @time resume_md("md/many_h2o/10h2o_free_temp.xyz", 20000; Δt=10.0)

# rf = make_runner_func("grad", "OHH"^10, "aug-cc-pvtz", 44)
# ef = make_ef(rf)

# r = [
#     -4.65854 -1.71304 -0.21052
#     -5.00927 -2.11280 -1.05037
#     -5.48298 -1.27641 0.08067
#     -4.66893 2.72495 0.87660
#     -5.16661 3.08174 1.63405
#     -4.03486 2.13204 1.36125
#     -0.90837 1.39445 -0.00979
#     -0.02519 1.08438 -0.27878
#     -1.47599 0.77968 -0.54641
#     -2.80619 1.24244 2.03033
#     -1.97336 1.31883 1.50869
#     -2.81797 0.26090 2.15654
#     -2.66329 -0.08357 -1.36545
#     -3.35384 0.51629 -1.72589
#     -3.23927 -0.78429 -0.98622
#     -6.05485 0.39667 -3.39252
#     -6.30833 1.04540 -4.07227
#     -5.52756 0.98372 -2.79254
#     -4.57759 1.97984 -1.79155
#     -3.84928 2.64639 -1.87424
#     -4.82963 2.20321 -0.85953
#     -2.53760 3.52428 -0.90336
#     -1.80116 2.96013 -0.57696
#     -3.08106 3.59469 -0.09144
#     -2.90122 -1.47660 1.88684
#     -2.62792 -2.38542 2.09673
#     -3.50716 -1.63557 1.12441
#     -5.96727 -2.13060 -2.47644
#     -6.31428 -2.63249 -3.23415
#     -6.03425 -1.20895 -2.83768
# ]' |> copy

# g1 = @time find_grad4(ef, r, 1e-3, 1, 1)
# @show g1
