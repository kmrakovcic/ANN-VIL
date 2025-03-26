from . import survival_probability, intrinsic_gamma, measured_gamma
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import concurrent.futures
import os
import time
from functools import partial
import math


def one_opacity_node(E, z, Eqg, i, j):
    opacity = survival_probability.opacity(E, z, Eqg)
    return opacity, i, j


def create_opacity_grid(E_min, E_max, Eqg_min, Eqg_max, N=(5, 5), z=0.035, parallel=True, verbose=False):
    assert len(N) == 2
    opacities = np.zeros(N) * np.nan
    E = np.logspace(E_min, E_max, N[0])
    Eqg = np.logspace(Eqg_min, Eqg_max, N[1])
    timeout_per_node = 5
    cpu_count = min(os.cpu_count(), N[0] * N[1])
    start_time = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(one_opacity_node, E[i], z, Eqg[j], i, j): i * N[1] + j + 1 for i in range(N[0])
                       for j in range(N[1])}
            completed = 0
            try:
                for future in concurrent.futures.as_completed(futures,
                                                              timeout=N[0] * N[1] * timeout_per_node / cpu_count):
                    episode_id = futures[future]
                    try:
                        opacity, i, j = future.result(timeout=timeout_per_node)
                        opacities[i, j] = opacity
                        completed += 1
                        if completed == N[0] * N[1]:
                            executor.shutdown(wait=False, cancel_futures=True)
                        if verbose:
                            print(f"\rGrid points {completed} / {N[0] * N[1]} calculated, ETA:",
                                  round((time.time() - start_time) * (N[0] * N[1] - completed) / completed, 1),
                                  end="s   ")

                    except concurrent.futures.TimeoutError:
                        if verbose:
                            print(f"\nTimeout occurred in grid point {episode_id}, skipping...")
                        else:
                            pass

                    except Exception as e:
                        if verbose:
                            print(f"\nError in grid point {episode_id}: {e}")
            except concurrent.futures.TimeoutError:
                if verbose:
                    print(f"\rTimeout occurred after {completed} grid points, continuing sequential...")
                executor.shutdown(wait=False, cancel_futures=True)
                for c, [i, j] in enumerate(np.argwhere(np.isnan(opacities))):
                    opacities[i, j] = survival_probability.opacity(E[i], z, Eqg[j])
                    if verbose:
                        print(f"\rGrid points {completed + c + 1} / {N[0] * N[1]} calculated", end="")
    else:
        for i in range(N[0]):
            for j in range(N[1]):
                opacities[i, j] = survival_probability.opacity(E[i], z, Eqg[j])
    return opacities, E, Eqg


def create_one_training_example(spectrum_parameters, lc_parameters, Eqg,
                                spectrum_error=None, lc_error=None, E_min=10 ** 10.55, E_max=10 ** 13.7,
                                z=0.035, photon_num=2000, interpolation_grid=None, t_observation=4 * 28 * 60,
                                verbose=False, i=0):
    photon_count = 0
    tries_count = 0
    intrinsic_time = None
    kappa2 = measured_gamma.distanceContrib(z)
    E_max = min([E_max, measured_gamma.max_energy(Eqg, t_observation, E_min, kappa2)])
    if E_max <= E_min:
        return np.nan, np.nan, i
    if interpolation_grid is not None:
        opacity_interpolator = RegularGridInterpolator((interpolation_grid[0],
                                                        interpolation_grid[1]),
                                                       interpolation_grid[2])
    else:
        opacity_interpolator = None
    if verbose:
        start_time = time.time()
    try:
        while photon_count < photon_num:
            tries_count += 1
            if tries_count > 10*photon_num:
                return np.nan, np.nan, i
            if lc_error is not None:
                A1, mean1, sigma1, A2, mean2, sigma2 = np.abs(np.random.normal(list(lc_parameters), list(lc_error)))
            else:
                A1, mean1, sigma1, A2, mean2, sigma2 = lc_parameters
            if spectrum_error is not None:
                E0, alpha = np.abs(np.random.normal(list(spectrum_parameters), list(spectrum_error)))
            else:
                E0, alpha = spectrum_parameters
            if intrinsic_time is None:
                intrinsic_time = intrinsic_gamma.intrinsic_times(A1, mean1, sigma1, A2, mean2, sigma2, size=1)
                intrinsic_energy = intrinsic_gamma.intrinsic_energy(E_min, E_max, E0, alpha, size=1)
                if opacity_interpolator is None:
                    opacity = survival_probability.opacity(intrinsic_energy[0], z, Eqg)
                else:
                    opacity = opacity_interpolator((intrinsic_energy[0], Eqg))
                survival_prob = [np.exp(-opacity)]
                survived_mask = [False]
            else:
                intrinsic_time = np.concatenate([intrinsic_time,
                                            intrinsic_gamma.intrinsic_times(A1, mean1,
                                                                        sigma1, A2,
                                                                        mean2, sigma2,
                                                                        size=1)])
                intrinsic_energy = np.concatenate([intrinsic_energy,
                                            intrinsic_gamma.intrinsic_energy(E_min, E_max,
                                                                           E0, alpha, size=1)])
                if opacity_interpolator is None:
                    opacity = survival_probability.opacity(intrinsic_energy[-1], z, Eqg)
                else:
                    opacity = opacity_interpolator((intrinsic_energy[-1], Eqg))
                survival_prob = np.concatenate([survival_prob, [np.exp(-opacity)]])
                survived_mask = np.concatenate([survived_mask, [False]])
            if t_observation > 0:
                if (np.random.random() < survival_prob[-1]) and (
                        intrinsic_time[-1] + measured_gamma.timeDelay(intrinsic_energy[-1], Eqg,
                                                                  kappa2) - measured_gamma.timeDelay(E_min, Eqg,
                                                                                                     kappa2) < t_observation):
                    survived_mask[-1] = True
                    photon_count += 1
            else:
                if np.random.random() < survival_prob[-1]:
                    survived_mask[-1] = True
                    photon_count += 1
            if verbose:
                if photon_count != 0:
                    print("\r", photon_count, "/", photon_num, "measured, ETA:",
                        round((time.time() - start_time) * (photon_num - photon_count) / photon_count, 1), "s", end="   ")

        assert survived_mask.sum() == photon_num
        measured_time = intrinsic_time[survived_mask] + measured_gamma.timeDelay(intrinsic_energy[survived_mask], Eqg,
                                                                             kappa2)
        measured_energy = measured_gamma.detectionEnergy(intrinsic_energy[survived_mask], z)
        sort_index = np.argsort(measured_time)
        measured_photons = np.vstack([measured_time - measured_time.min(), measured_energy]).T[sort_index]
        intrinsic_photons = np.vstack([intrinsic_time, intrinsic_energy, survival_prob]).T[sort_index]
        return measured_photons, intrinsic_photons, i
    except Exception as e:
        print(f"Exception in worker: {e}")
        return np.nan, np.nan, i


def create_train_set(examples_num, spectrum_parameters, lc_parameters, Eqg=[10 ** 16, 10 ** 26],
                     spectrum_error=(0., 0.24), lc_error=(6., 185., 301., 11., 220., 283.),
                     E=[10 ** 10.55, 10 ** 13.7],
                     z=0.035, photon_num=2000, t_observation=4 * 28 * 60,
                     interpolation_grid_file=None, verbose=False,
                     parallel=True):
    if interpolation_grid_file is not None:
        opacity_grid, E_grid, Eqg_grid = np.load(interpolation_grid_file).values()
        interpolation_grid = (E_grid, Eqg_grid, opacity_grid)
    else:
        interpolation_grid = None
    sp_par = np.random.uniform(np.repeat(np.array(spectrum_parameters[0])[:, np.newaxis], examples_num, 1).T,
                               np.repeat(np.array(spectrum_parameters[1])[:, np.newaxis], examples_num, 1).T)
    lc_par = np.random.uniform(np.repeat(np.array(lc_parameters[0])[:, np.newaxis], examples_num, 1).T,
                               np.repeat(np.array(lc_parameters[1])[:, np.newaxis], examples_num, 1).T)
    Eqg_par = np.power(10, np.random.uniform(math.log10(Eqg[0]), math.log10(Eqg[1]), examples_num))
    timeout_per_node = 0.03 * photon_num
    cpu_count = min(max(1, os.cpu_count() - 1), examples_num)
    X_train = np.zeros((examples_num, photon_num, 2)) * np.nan
    Y_train = np.concatenate([sp_par, lc_par, Eqg_par[:, np.newaxis]], axis=-1)
    start_time = time.time()
    one_example_partial = partial(create_one_training_example, photon_num=photon_num,
                                  lc_error=lc_error,
                                  spectrum_error=spectrum_error,
                                  interpolation_grid=interpolation_grid,
                                  z=z, E_min=E[0], E_max=E[1],
                                  verbose=False, t_observation=t_observation)
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(one_example_partial, sp_par[i], lc_par[i], Eqg_par[i], i=i): i for i in
                       range(examples_num)}
            completed = 0
            i = 0
            try:
                for future in concurrent.futures.as_completed(futures,
                                                              timeout=examples_num * timeout_per_node / cpu_count):
                    episode_id = futures[future]
                    try:
                        measured_photons, intrinsic_photons, i = future.result(timeout=timeout_per_node)
                        X_train[i, :, :] = measured_photons
                        completed += 1
                        if completed == examples_num:
                            executor.shutdown(wait=False, cancel_futures=True)
                        if verbose:
                            print(f"\rExamples {completed} / {examples_num} generated, ETA:",
                                  round((time.time() - start_time) * (examples_num - completed) / completed, 1),
                                  end="s   ")
                    except concurrent.futures.TimeoutError:
                        if verbose:
                            print(f"\nTimeout occurred in example {episode_id}, skipping...")
                        else:
                            pass

                    except Exception as e:
                        if verbose:
                            print(f"\nError in example {episode_id}: {e}")
            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                if verbose:
                    print(f"\rTimeout occurred after {completed} examples, continuing sequential...")
                for c, i in enumerate(np.unique(np.argwhere(np.isnan(X_train))[:, 0])):
                    sp_par_tmp = np.random.uniform(spectrum_parameters[0], spectrum_parameters[1])
                    lc_par_tmp = np.random.uniform(lc_parameters[0], lc_parameters[1])
                    Eqg_par_tmp = np.random.uniform(Eqg[0], Eqg[1])
                    X_train[i, :], _, _ = one_example_partial(sp_par_tmp, lc_par_tmp, Eqg_par_tmp)
                    if verbose:
                        print(f"\rExamples {completed + c + 1} / {examples_num} generated", end="")
            except Exception as e:
                print(e)
                executor.shutdown(wait=False, cancel_futures=True)
    else:
        for i in range(examples_num):
            X_train[i, :], _, _ = one_example_partial(sp_par[i], lc_par[i], Eqg_par[i])
            if verbose:
                print(f"\rExamples {i + 1} / {examples_num} generated", end="")
    return X_train, Y_train
