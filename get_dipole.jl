
const dip_reg = r"Dipole moment in \[Debye\].+?-+\n(.+?) +-{2}"s

function get_dipole(name)
    m = match(dip_reg, read("$name.out", String))
    parse.(Float64, split(m.captures[1])[4:4:end])
end

function make_dip_inp_func(atoms, basis)
    function make_inp(r)
        r = reshape(r, 3, length(r) ÷ 3)
        io = IOBuffer()

        print(
            io,
            """
system
    name: Dipole
    charge: 0
end system

do
    ground state
end do

memory
    available: 1024
end memory

method
    hf
end method

hf mean value
   dipole
end hf mean value

solver scf
    restart
    gradient threshold: 1d-10
end solver scf

geometry
basis: $basis
"""
        )

        for (i, a) in enumerate(atoms)
            println(io, "    ", a, "    ", r[1, i], ' ', r[2, i], ' ', r[3, i])
        end

        println(io, "end geometry")

        String(take!(io))
    end
end

function run_inp_clean(name, omp)
    run(`$(homedir())/eT_clean/build/eT_launch.py $(name).inp --omp $(omp)`)
    nothing
end

function find_dipole(atoms, basis, r, omp)
    name = "dipole"
    inp_func = make_dip_inp_func(atoms, basis)
    inp = inp_func(r)
    open("$name.inp", "w") do io
        print(io, inp)
    end
    run_inp_clean(name, omp)
    get_dipole(name)
end
