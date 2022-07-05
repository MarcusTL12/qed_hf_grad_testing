using OhMyREPL
using PyCall

pyscf = pyimport("pyscf")

pyimport("pyscf.lib").num_threads(parse(Int, read("omp.txt", String)))

function make_mol(atoms, r, basis)
    pyscf.M(atom=[(a, (rs...,))
                      for (a, rs) in zip(split_atoms(atoms), eachcol(r))],
        basis=basis)
end

function make_hf_func(atoms, basis)
    function hf_func(r)
        rhf = pyscf.scf.RHF(make_mol(atoms, r, basis))
        rhf.conv_tol = 1e-16
        rhf.max_cycle = 1000
        rhf.run()
    end
end

function make_ef(hf_func)
    function ef(r)
        @time hf_func(r).e_tot
    end
end

const atom_reg = r"[A-Z][a-z]?"

function split_atoms(atoms)
    [m.match for m in eachmatch(atom_reg, atoms)]
end

function find_grad2(ef, r, h, q, atm)
    dr = zeros(size(r))
    dr[q, atm] = h
    (ef(r + dr) - ef(r - dr)) / 2h / 1.8897261245650618
end

function find_grad4(ef, r, h, q, atm)
    dr = zeros(size(r))
    dr[q, atm] = h
    (-ef(r + 2dr) + 8ef(r + dr) - 8ef(r - dr) + ef(r - 2dr)) / 12h / 1.8897261245650618
end

function find_grad(ef, r, h; method=find_grad2)
    grad = zeros(size(r))
    for i in 1:size(r, 1), j in 1:size(r, 2)
        grad[i, j] = method(ef, r, h, i, j)
    end
    grad
end
