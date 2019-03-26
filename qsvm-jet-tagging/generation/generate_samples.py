import argparse
import operator
import os
import time
from functools import lru_cache, reduce
from itertools import combinations

import numpy as np
import pandas as pd
import progressbar
import pyjet
from numpythia import PDG_ID, STATUS, Pythia


def CalcDeltaR(j1, j2):
    eta1 = j1.eta
    phi1 = j1.phi
    eta2 = j2.eta
    phi2 = j2.phi

    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5

    return dR

# energy correlators
# https://arxiv.org/pdf/1411.0665.pdf


def CalcEECorr(jet, n=1, beta=1.0):

    assert n == 2 or n == 3, f'n must be in [2, 3] but is {n}'

    jet_particles = jet.constituents()

    if len(jet_particles) < n:
        return -1

    currentSum = 0

    if n == 2:
        for p1, p2 in combinations(jet_particles, 2):
            # get the terms of the triplet at hand
            pt1 = p1.pt
            pt2 = p2.pt
            dr12 = CalcDeltaR(p1, p2)

            # calculate the partial contribution
            thisterm = pt1*pt2 * (dr12)**beta

            # sum it up
            currentSum += thisterm

        eec = currentSum/(jet.pt)**2

    elif n == 3:
        dr = {(p1, p2): CalcDeltaR(p1, p2)
              for p1, p2 in combinations(jet_particles, 2)}
        for p1, p2, p3 in combinations(jet_particles, 3):
            # get the terms of the triplet at hand
            dr12 = dr[(p1, p2)]
            dr13 = dr[(p1, p3)]
            dr23 = dr[(p2, p3)]

            # calculate the partial contribution
            thisterm = p1.pt*p2.pt*p3.pt * (dr12*dr13*dr23)**beta

            # sum it up
            currentSum += thisterm

        eec = currentSum/(jet.pt)**3
    return eec


def generate_samples(gen_type='qcd', n=10, debug=False, recalculate=False):
    # If the samples already exist in a file, simply return them.
    filename = os.path.join('samples', f'{gen_type}_{n}.pkl')
    if not recalculate and os.path.exists(filename):
        print(f'Loading samples from file {filename}')
        return pd.read_pickle(filename)

    start = time.time()
    print(f'Generating {gen_type} with {n} events.')
    assert gen_type == 'qcd' or gen_type == 'higgs', f'gen_type must be one of [qcd, higgs] but was {gen_type}.'

    pythia_config = os.path.join('pythia_config', f'{gen_type}.cmnd')
    pythia = Pythia(config=pythia_config, random_state=1)

    final_jets = []
    for event in progressbar.progressbar(pythia(events=n), max_value=n):

        # Run jet finding.
        jets = pyjet.cluster(event.all(STATUS == 1), R=1,
                             p=-1, ep=True).inclusive_jets()

        found_jet = None

        if gen_type == 'higgs':
            seen_higgs = False
            # If we are generating Higgs, we need to save where they went.
            for particle in event.all(PDG_ID == 25, return_hepmc=True):
                parents = particle.parents()
                if parents.size > 0 and parents[0]['pdgid'] == 5100039:
                    if not seen_higgs or particle.pt > higgs.pt:
                        higgs = particle
                        seen_higgs = True

            if not seen_higgs:
                continue

            for jet in jets[:4]:
                dR = CalcDeltaR(jet, higgs)
                if dR < 1.0:
                    if debug:
                        print('Found Higgs Jet')
                        found_jet = jet
        else:
            found_jet = jets[0]

        if found_jet is None:
            continue

        if debug:
            print(found_jet)

        # Calculating the ECF functions for two subjettiness
        ee2 = CalcEECorr(found_jet, n=2, beta=1.0)
        ee3 = CalcEECorr(found_jet, n=3, beta=1.0)
        d2 = ee3/ee2**3

        final_jets.append([found_jet.pt, found_jet.eta,
                           found_jet.phi, found_jet.mass, ee2, ee3, d2])

    # Save the final jets to a file.
    final_jets = pd.DataFrame(data=final_jets, columns=[
                              'pt', 'eta', 'phi', 'mass', 'ee2', 'ee3', 'd2'])
    final_jets.to_pickle(filename)

    print("==========================")
    print("DONE WITH GENERATION - returning")
    print(f'Total time = {time.time() - start}')
    print("==========================")

    return final_jets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate samples (jet kinematics from qcd or higgs events).')
    parser.add_argument('gen_type', choices=[
                        'qcd', 'higgs'], help='The type of events to generate.')
    parser.add_argument(
        'n', type=int, help='The number of events to generate.')
    parser.add_argument('--recalculate', action='store_true',
                        help='If set, ignore cached files and regenerate.')
    args = parser.parse_args()
    generate_samples(args.gen_type, args.n, recalculate=args.recalculate)
