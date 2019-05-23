"""Sample Generation

These routines handle generating, storing, and retrieving sample data. The data
is higgs/QCD jet kinematics, and is generated using pythia and fastjet.
"""

import argparse
import glob
import math
import operator
import os
import re
import time
from functools import lru_cache, reduce
from itertools import combinations

import numpy as np
import pandas as pd
import progressbar
import pyjet
from common import utils
from numpythia import PDG_ID, STATUS, Pythia


def CalcDeltaRArray(p, a):
    dEta = p['eta'] - \
        a['eta'].repeat(p.shape[0]).reshape(a.shape[0], p.shape[0])
    dPhi = np.abs(p['phi'] - a['phi'].repeat(p.shape[0]
                                             ).reshape(a.shape[0], p.shape[0]))
    mask = dPhi > np.pi
    dPhi[mask] *= -1
    dPhi[mask] += 2 * np.pi
    return (dPhi**2 + dEta**2)**0.5


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


def angle(jet, particles):
    ptot2 = (jet.px**2 + jet.py**2 + jet.pz**2) * \
        (particles['px']**2 + particles['py']**2 + particles['pz']**2)
    arg = (jet.px * particles['px'] + jet.py *
           particles['py'] + jet.pz * particles['pz']) / ptot2**(1/2)
    arg[np.isnan(arg)] = 1.0
    arg[arg > 1.0] = 1.0
    arg[arg < -1.0] = -1.0
    return np.arccos(arg)


def calc_angularity(jet):
    jet_particles = jet.constituents_array(ep=True)

    if jet_particles.shape[0] == 0:
        return -1
    if jet.mass < 1.e-20:
        return -1

    theta = angle(jet, jet_particles)
    e_theta = jet_particles['E'] * np.sin(theta)**-2 * (1 - np.cos(theta))**3

    return np.sum(e_theta) / jet.mass


@lru_cache(maxsize=1)
def t0(jet):
    return sum(p.pt * CalcDeltaR(p, jet) for p in jet.constituents())


def tn(jet, n):
    assert n >= 0
    if n == 0:
        return t0(jet)
    particles = jet.constituents_array()
    if len(particles) < n:
        return -1
    subjets = pyjet.cluster(particles, R=1.0, p=1).exclusive_jets(n)
    subjets_array = [subjet.constituents_array() for subjet in subjets]
    wta_axes = [a[np.argmax(a['pT'])] for a in subjets_array]
    wta_axes = np.array(wta_axes, dtype=subjets_array[0].dtype)
    return np.sum(particles['pT']*CalcDeltaRArray(particles, wta_axes).min(axis=0)) / t0(jet)


def calc_KtDeltaR(jet):
    particles = jet.constituents_array()
    if particles.shape[0] < 2:
        return 0.0

    subjets = pyjet.cluster(particles, R=0.4, p=1).exclusive_jets(2)

    return CalcDeltaR(subjets[0], subjets[1])


def get_pt_range(pt_cut):
    if pt_cut == 'low':
        return (250, 500)
    elif pt_cut == 'high':
        return (1000, 1200)


def get_filename(gen_type, n, pt_cut):
    project_dir = utils.get_project_path()
    if pt_cut is not None:
        pt_min, pt_max = get_pt_range(pt_cut)
        filename = os.path.join(project_dir, 'samples',
                                f'{gen_type}_{n}_pt_{pt_min}_{pt_max}.pkl')
    else:
        filename = os.path.join(project_dir, 'samples', f'{gen_type}_{n}.pkl')
    return filename


def get_pythia_config(gen_type, pt_cut):
    code_dir = utils.get_source_path()
    if pt_cut is not None:
        pythia_config = os.path.join(
            code_dir, 'pythia_config', f'{gen_type}_{pt_cut}.cmnd')
    else:
        pythia_config = os.path.join(
            code_dir, 'pythia_config', f'{gen_type}.cmnd')
    return pythia_config


def generate_samples(gen_type='qcd', n=10, pt_cut=None, excess_factor=None, debug=False, recalculate=False):
    assert pt_cut is None or pt_cut in (
        'low', 'high'), f'invalid pt_cut={pt_cut}'
    pythia_config = get_pythia_config(gen_type, pt_cut)
    filename = get_filename(gen_type, n, pt_cut)
    if excess_factor is None and pt_cut is not None:
        excess_factor = 1.5
    else:
        excess_factor = 1
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not recalculate and os.path.exists(filename):
        if debug:
            print(f'Loading samples from file {filename}')
        return pd.read_pickle(filename)

    start = time.time()
    print(f'Generating {gen_type} with {excess_factor*n} events.')
    assert gen_type == 'qcd' or gen_type == 'higgs', f'gen_type must be one of [qcd, higgs] but was {gen_type}.'

    pythia = Pythia(config=pythia_config, random_state=1)

    final_jets = []
    for event in progressbar.progressbar(pythia(events=excess_factor*n), max_value=excess_factor*n):

        # Run jet finding.
        jets = pyjet.cluster(event.all((STATUS == 1) & (STATUS == 1)), R=1.0,
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

        angularity = calc_angularity(found_jet)

        t1 = tn(found_jet, n=1)
        t2 = tn(found_jet, n=2)
        t3 = tn(found_jet, n=3)
        t21 = t2 / t1 if t1 > 0.0 else 0.0
        t32 = t3 / t2 if t2 > 0.0 else 0.0

        KtDeltaR = calc_KtDeltaR(found_jet)

        final_jets.append([found_jet.pt, found_jet.eta,
                           found_jet.phi, found_jet.mass, ee2, ee3, d2, angularity, t1, t2, t3, t21, t32, KtDeltaR])

    # Save the final jets to a file.
    final_jets = pd.DataFrame(data=final_jets, columns=[
                              'pt', 'eta', 'phi', 'mass', 'ee2', 'ee3', 'd2', 'angularity', 't1', 't2', 't3', 't21', 't32', 'KtDeltaR'])
    if pt_cut is not None:
        final_jets = final_jets[(final_jets['pt'] > pt_min)
                                & (final_jets['pt'] < pt_max)]
        if final_jets.shape[0] < n:
            raise RuntimeError(
                f'excess factor was insufficient, less than {n} events ({final_jets.shape[0]}) were generated satisfying specified pt cut.')
        final_jets = final_jets[:n]
        final_jets.reset_index(inplace=True, drop=True)
    final_jets.to_pickle(filename)

    print("==========================")
    print("DONE WITH GENERATION - returning")
    print(f'Total time = {time.time() - start}')
    print("==========================")

    return final_jets


def load_samples(gen_type='qcd', n=1000, pt_cut=None):
    # Find the largest sample file
    project_dir = utils.get_project_path()
    if pt_cut is None:
        pattern = re.compile(os.path.join(
            project_dir, 'samples', f'{gen_type}_([0-9]+)'))
    else:
        pt_min, pt_max = get_pt_range(pt_cut)
        pattern = re.compile(f'{gen_type}_([0-9]+)_pt_{pt_min}_{pt_max}')
    sample_paths = [os.path.splitext(os.path.split(p)[1])[0] for p in glob.glob(
        os.path.join(project_dir, 'samples', '*.pkl'))]
    max_size = -1
    for p in sample_paths:
        m = pattern.search(p)
        if m:
            max_size = max(max_size, int(m.group(1)))
    if max_size >= n:
        samples = generate_samples(gen_type, max_size, pt_cut)
        return samples.iloc[:n]
    return generate_samples(gen_type, n, pt_cut)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate samples (jet kinematics from qcd or higgs events).')
    parser.add_argument('gen_type', choices=[
                        'qcd', 'higgs'], help='The type of events to generate.')
    parser.add_argument(
        'n', type=int, help='The number of events to generate.')
    parser.add_argument('--pt_cut', choices=['low', 'high'],
                        help='If set, apply pt cut on data (low = [250, 500] GeV, high = [1000, 1200] GeV).')
    parser.add_argument('--recalculate', action='store_true',
                        help='If set, ignore cached files and regenerate.')
    parser.add_argument('--debug', action='store_true',
                        help='If set, print debug info.')
    args = parser.parse_args()
    jets = generate_samples(args.gen_type, args.n,
                            recalculate=args.recalculate, debug=args.debug, pt_cut=args.pt_cut)
