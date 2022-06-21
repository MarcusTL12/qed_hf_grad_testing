using PyCall
using OhMyREPL
using LinearAlgebra

pushfirst!(pyimport("sys")."path", "./geomeTRIC")

engine = pyimport("engine")

include("../common.jl")


function test_engine()
    atoms = split_atoms("OHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        0.00074 -0.00014 0.00003
        0.03041 0.10161 0.97727
        -0.27764 -0.92380 -0.08655
        -0.00484 0.00131 2.75024
        0.78062 -0.00539 3.32392
        -0.68819 0.38765 3.32456
    ]' |> copy

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 8)

    egf = make_e_and_grad_func(rf)

    qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

    qed_hf_engine.calc_new(r, nothing)
end

function test_engine2()
    atoms = split_atoms("OHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        0.296136   0.637227   0.138607
        0.115088   0.264813   0.993599
       -0.483927   0.464701  -0.36692
       -0.132064  -0.63688    2.80747
        0.707587  -0.847976   3.19146
       -0.66172   -0.320645   3.52525
    ]

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    egf = make_e_and_grad_func(rf)

    qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

    m = engine.run_opt(qed_hf_engine)

    m.xyzs[end]
end

function test_h2o()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = Float64[
        0 0 0
        1 0 0
        0 1 0
    ]

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    egf = make_e_and_grad_func(rf)

    qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

    m = engine.run_opt(qed_hf_engine)

    m.xyzs[end]
end
