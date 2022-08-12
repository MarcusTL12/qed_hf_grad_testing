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
        1.222051702977709 -0.047976935285102135 -2.176702298518663
        -0.761716069473667 2.318190562165025 0.2688610474325302
        -5.072114658499238 1.1695348731162998 -0.06150589766231997
        -0.1265718828837004 -0.26782655832103797 2.1396549893562944
        1.2913592718548452 -0.15366985992816215 -0.9978078719613436
        -1.1401339621406408 0.1133114784252714 -0.5518639933168988
        -2.1660745545731133 -0.9628310899799294 -0.2118838056858447
        -3.5383227024631174 -0.5299261060552017 -0.7147944682224556
        -3.9427534155620347 0.8325450085904837 -0.21192397282135994
        -1.551395336867879 1.4826841732519151 -0.025609993451771236
        2.501297313905752 -0.2863417068497698 -0.13918865263730257
        3.83582294903732 -0.33181455746633065 -0.48615119119245365
        4.757039209450443 -0.4380333436868053 0.5510870155889122
        4.3423605449966365 -0.49559712572118836 1.8802797769389081
        2.9928401747616986 -0.4512829135534189 2.2159383529904106
        2.0901992450813203 -0.3460309528803493 1.1779834339651574
        0.603220071886174 -0.26287781107041364 1.2049453717640406
        0.20486089543729144 -0.19016913037814087 -0.12728966129642744
        -2.905901705776993 1.710017560080849 0.05338436937533853
        -1.0834861267803242 0.21717632383902905 -1.6357095769344292
        -1.8712955792290549 -1.8979092316121875 -0.681751285767459
        -2.189702963776529 -1.1226037230408945 0.8649373728496724
        -3.5459353342946676 -0.47672918151738103 -1.807531265954358
        -4.322626008698774 -1.2226158154186721 -0.42435952458902715
        -3.175781541224535 2.6150276711077085 0.38660331584004376
        4.147049246603189 -0.28148763846686164 -1.5189570969727801
        5.81374468182685 -0.473287709597674 0.32543109428541395
        5.084847183563263 -0.573973768163315 2.6621776202997394
        2.6613732334794578 -0.49198669916449717 3.242826051743382
       
    ]

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.05

    rf = make_runner_func("grad.2", freq, pol, coup, atoms, basis, 40)

    egf = make_e_and_grad_func(rf)

    qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

    m = engine.run_opt(qed_hf_engine)

    m.xyzs[end]
end
