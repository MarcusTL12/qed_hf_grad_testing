import numpy as np
import geometric
import geometric.molecule
import tempfile


class qed_hf_engine(geometric.engine.Engine):
    def __init__(self, egf, atoms, r):
        self.egf = egf
        molecule = geometric.molecule.Molecule()
        molecule.elem = atoms
        molecule.xyzs = [r]
        super(qed_hf_engine, self).__init__(molecule)

    def calc_new(self, coords, _):
        e, g = self.egf(coords.T)
        return {'energy': e, 'gradient': g.T.ravel()}


def run_opt(eng):
    tmpf = tempfile.mktemp()
    print(eng.egf)
    return geometric.optimize.run_optimizer(
        customengine=eng, check=1, input=tmpf, logIni='./log.ini', constraints="./constraints.txt"
    )
