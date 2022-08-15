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
        0.296136 0.637227 0.138607
        0.115088 0.264813 0.993599
        -0.483927 0.464701 -0.36692
        -0.132064 -0.63688 2.80747
        0.707587 -0.847976 3.19146
        -0.66172 -0.320645 3.52525
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

function test_thalidomide()
    atoms = split_atoms("OOOOCCCCCCCCCCCCCNNHHHHHHHHHH")
    basis = "cc-pvdz"
    r = Float64[
        1.2126677058733322 -0.085455654371993 -2.194836295049347
        -0.6770967934695011 2.2328833557241263 0.4988214156857565
        -5.028732011643822 1.2578183454163758 0.1100295341889958
        -0.1390687248676852 -0.3873451637724787 2.1139375749982037
        1.2799471285190458 -0.20205156487442427 -1.0168642810028887
        -1.136088381432121 0.16049518931259257 -0.5535902582156154
        -2.201286493903016 -0.9018927040693103 -0.3378660048681535
        -3.5506568682346216 -0.36393564925610117 -0.7926729342068737
        -3.9106140262315763 0.9259328821845592 -0.11336089312612818
        -1.5029710716360574 1.4739180675305736 0.11569650077267837
        2.487300443760497 -0.36221164863183125 -0.1620084548931173
        3.820724474715334 -0.40154527471459966 -0.5088149179372081
        4.743068889194752 -0.5016759256230403 0.5264821416911692
        4.329714253489133 -0.5624170865671299 1.8545475488722611
        2.980776657522796 -0.5337755401468147 2.1894758023311534
        2.0772994506913576 -0.43294554799771995 1.1536012972218148
        0.5922398079953282 -0.34869009668610257 1.1821325896631276
        0.19402045998681697 -0.22608388806038948 -0.14722023277710734
        -2.8458121999832806 1.7310395734194557 0.24510484594485746
        -1.0594939707247069 0.36647946554026894 -1.6193086921892719
        -1.9398415998557668 -1.7811988493909305 -0.9097411104556726
        -2.2400278887933824 -1.172846450151339 0.7143433201175563
        -3.539187637773618 -0.17438621775752514 -1.8682739338838383
        -4.362810092787828 -1.052872185521205 -0.6031870087524228
        -3.0825333778317843 2.5788814513954788 0.7135783876641664
        4.131434031173338 -0.34497025072689064 -1.540257761730527
        5.798631801011611 -0.5269666893526539 0.30005909786343443
        5.071899033380387 -0.6317857789290491 2.6358515059495344
        2.6507508746310187 -0.5788803723759923 3.215420477561139
    ]

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.2

    rf = make_runner_func("grad.2", freq, pol, coup, atoms, basis, 40)

    egf = make_e_and_grad_func(rf)

    qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

    m = engine.run_opt(qed_hf_engine)

    m
end
