# Intent is to perform a high level sky subtraction using PCA decomposition of spectra from offset sky frames
# By Jonah Gannon June 2019 major update Nov 2019
# PhD student @ Swinburne university
# Works on Ubuntu 18.04 LTS

# Thanks go to Dr Ned Taylor for advice on the best way to structure the code

############################################## Import Libraries ########################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.optimize as op
from astropy.io import fits
from matplotlib import cm
import glob
import progressbar
from multiprocessing import Pool
import corner
import emcee
import scipy.ndimage as ndi
import spectres
from math import ceil
class HaltException(Exception): pass
########################################################################################################################

#Time the Code
import time
start = time.time()

# update plotting sizes
plt.rcParams.update({'font.size': 24})

############################################## User Inputs #############################################################

science_file =          '~/science_file.fits'                               # Input file to sky subtract
var_file =              '~/science_file_noise.fits'                         # Noise estimate on input file
out_file_optim =        '~/output_sky_subtracted_via_optimisation.fits'     # define a place to put the output from simple linear fit
out_file_emcee_mean =   '~/output_sky_subtracted_emcee_mean.fits'           # define a place to put the output from mean of emcee posterior
out_file_emcee_median = '~/output_sky_subtracted_emcee_median.fits'         # define a place to put the output from median of emcee posterior

template_loc = '~/template_loc/' # Location of the template for fitting, multiple templates can be used

sky_files = ['~/sky_file_1.fits',
             '~/sky_file_2.fits',
             '~/sky_file_3.fits',
             '~/sky_file_4.fits',
             '~/sky_file_5.fits',
             '~/sky_file_6.fits',
             '~/sky_file_7.fits',
             '~/sky_file_8.fits',
             '~/sky_file_9.fits',
             '~/sky_file_10.fits'] # List of the 10 sky files to use


sigma_smooth = 0. # define smoothing sigma for the template to be smoothed for for the model

# define a redshift range to fit over

z_min = 0.                  # Minimum allowable redshift
z_guess = 0.                # initial redshift guess
z_max = 0.                  # maximum allowable redshift

write_out = True            # write the spectra out at the end?
just_fit_templates = False  # just run an optimisation over multiple templates or go into full MCMC after choosing 1

########################################################################################################################

# Make sure the redshift limits make sense
if z_min >= z_max or z_guess >= z_max or z_min >= z_guess:
    raise HaltException("Check your redshift limits - they are incorrect.")

############################################## Load in data ############################################################

# import the science file and header
science_data = fits.open(science_file)[0].data

science_header = fits.open(science_file)[0].header

# import the corresponding variance

var_data = fits.open(var_file)[0].data

# define the science wavelengths

wavelength = np.linspace(science_header['CRVAL1'], science_header['CRVAL1']+science_header['CDELT1'] * (science_data.shape[0]-1), science_data.shape[0])

########################################################################################################################

# plot the data with corresponding variance

fig = plt.figure(1, figsize = (32,16))

ax = plt.subplot(111)

ax.plot(wavelength, science_data, 'c-', lw = 1)

ax.fill_between(wavelength, science_data+np.sqrt(var_data), science_data-np.sqrt(var_data), color = 'tab:red', alpha = 0.5)

plt.legend(('Science Data', '1 Sigma Fill'), prop={'size': 24})
ax.set_xlabel('Wavelength [$\AA$]')

plt.show()

####################################### Create a template file list  ###################################################

template_list = sorted(glob.glob("%s/*%s" % (template_loc, '*.fits')))

########################################################################################################################

######################################        Construct Sky PCA's         ##############################################
sky_data = []

for i in enumerate(sky_files):
    sky_data.append(fits.open(sky_files[i[0]])[0].data)

sky_matrix = np.column_stack((sky_data))
sky_matrix = sky_matrix

pca = PCA(n_components=sky_matrix.shape[1])

pcs = pca.fit_transform(sky_matrix)

eigen_spectra = np.transpose(pcs)

########################################################################################################################

######################## Define a function that clips redshifts and smooths templates ##################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def redshift_template(z):

    redshifted_template = spectres.spectres(wavelength, template_wavelength[start_index:finish_index]*(1+z), template_data[start_index:finish_index])

    return redshifted_template

########################################################################################################################

############################################## Define maximum likelihood estimators  ###################################

def lnlike(to_fit, just_give_sky_model = False, just_give_model = False):

    redshifted_template = redshift_template(z = to_fit[-1])

    # first we create our galaxy template
    continuum_model = to_fit[-3] * redshifted_template + to_fit[-2]

    # Build A Sky Model

    sky_model = to_fit[0] * eigen_spectra[0,:] + to_fit[1] * eigen_spectra[1,:] + to_fit[2] * eigen_spectra[2,:] + to_fit[3] * eigen_spectra[3,:] + to_fit[4] * eigen_spectra[4,:] + to_fit[5] * eigen_spectra[5,:] + to_fit[6] * eigen_spectra[6,:] + to_fit[7] * eigen_spectra[7,:] + to_fit[8] * eigen_spectra[8,:] + to_fit[9] * eigen_spectra[9,:]

    if just_give_sky_model:
        return sky_model

    # build a model of what we have observed, galaxy + sky
    model = sky_model + continuum_model

    if just_give_model:
        return model

    chi_squared = ((science_data - model) **2 / var_data)

    return  -0.5 * (np.sum(chi_squared))


# we note that we actually minimise the negative log likelihood and hence

neg_lnlike = lambda *args: -lnlike(*args)

########################################################################################################################

############################# Loop through all templates and all redshift choices minimising negative lnlike ###########

best_fits_model = []

best_fits_redshift = []

best_fits_lnlike = []

best_fits_results = []

ev_guess = np.linalg.norm(science_data) / np.linalg.norm(eigen_spectra[0,:])

guess = np.array([ev_guess, ev_guess, ev_guess, ev_guess, ev_guess, ev_guess, ev_guess, ev_guess, ev_guess, ev_guess, 0.07 * np.mean(science_data), 1 * np.mean(science_data), z_guess])
bounds = ((None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (0, None), (None, None), (z_min, z_max))

print("Optimising Templates...")
bar = progressbar.ProgressBar(maxval = len(template_list))

for i, template_path in enumerate(template_list):
    template_header = fits.open(template_path)[0].header
    template_data = ndi.gaussian_filter1d(fits.open(template_path)[0].data, sigma = sigma_smooth / template_header['CDELT1'])
    template_data = template_data - np.mean(template_data)
    template_wavelength = np.linspace(template_header['CRVAL1'], template_header['CRVAL1'] + (template_header['CDELT1'] * (template_data.shape[0] - 1)), template_data.shape[0])

    insurance = ceil(science_header['CDELT1'] / template_header['CDELT1']) + 1

    start_index = find_nearest(template_wavelength*(1+z_max), wavelength[0]) - insurance

    finish_index = find_nearest(template_wavelength*(1+z_min), wavelength[-1]) + insurance

    model_fit = op.minimize(neg_lnlike, guess, method='TNC',
                            args=(),
                            bounds=bounds, options ={'maxiter':5000})
    best_fits_model.append(i)
    best_fits_redshift.append(model_fit.x[-1])
    best_fits_lnlike.append(neg_lnlike(model_fit.x, False, False))
    best_fits_results.append(model_fit.x)
    bar.update(i+1)

print("")

best_fit_of_all = min(best_fits_lnlike)
index = best_fits_lnlike.index(best_fit_of_all)

template_header = fits.open(template_list[index])[0].header
template_data = ndi.gaussian_filter1d(fits.open(template_list[index])[0].data, sigma=sigma_smooth / template_header['CDELT1'])
template_data = template_data - np.mean(template_data)
template_wavelength = np.linspace(template_header['CRVAL1'], template_header['CRVAL1'] + (
            template_header['CDELT1'] * (template_data.shape[0] - 1)), template_data.shape[0])

insurance = ceil(science_header['CDELT1'] / template_header['CDELT1']) + 1

start_index = find_nearest(template_wavelength * (1 + z_max), wavelength[0]) - insurance

finish_index = find_nearest(template_wavelength * (1 + z_min), wavelength[-1]) + insurance

############################################## Plot all Optmisation Results#############################################

colour_index = np.linspace(0, 1, template_list.__len__())

fig = plt.figure(2, figsize = (16,16))
ax = plt.subplot(111)

ax.scatter(best_fits_redshift, best_fits_lnlike, color = cm.rainbow(colour_index[best_fits_model]))

ax.set_xlabel('z')

ax.set_ylabel('$\sim$ Chi Squared')

plt.show()

########################################################################################################################


############################################## Plot the Best Optimisation Fit ##########################################

best_fit = lnlike(best_fits_results[index], just_give_sky_model= False, just_give_model=True)

fig = plt.figure(3, figsize = (32,16))

ax = plt.subplot(111)

ax.plot(wavelength, science_data, 'b-', lw = 3)
ax.plot(wavelength, best_fit, 'r-', lw = 3)

plt.legend(('Science Data', 'Best Fit'), prop={'size': 24})
ax.set_xlabel('Wavelength [$\AA$]')

plt.show()

########################################################################################################################

print("Best fitting template is %s at a redshift of %.7f" % (template_list[best_fits_model[index]], best_fits_redshift[index]))

if just_fit_templates == True:
    raise KeyboardInterrupt("Just wanted to Run the Initial fitting")


# Now we are going to run some MCMC to try to explore the parameter space fully
# To see if our simple optimisation has found the best fit to our data or gotten stuck in a local minimum


################################## Define Uniform Priors ###############################################################
def lnprior(to_fit):
    return_value = 0

    for i, param in enumerate(to_fit[:-3]):
        if -10000 < param < 10000:
            return_value = return_value + 0
        else:
            return_value = return_value -np.inf

    if 0 < to_fit[-3] < 10000:
        return_value = return_value + 0
    else:
        return_value = return_value - np.inf

    if -10000 < to_fit[-2] < 10000:
        return_value = return_value + 0
    else:
        return_value = return_value - np.inf

    if z_min < to_fit[-1] < z_max:
        return_value = return_value + 0
    else:
        return_value = return_value - np.inf

    return return_value



def lnprob(to_fit, TF1, TF2):
    lp = lnprior(to_fit)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(to_fit, TF1, TF2)

########################################################################################################################


####################################### Run MCMC #######################################################################
ndim = len(eigen_spectra) + 3
nwalkers = 800
pos = [best_fits_results[index] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(False, False), pool = pool)
    print("Running MCMC Fitting...")
    sampler.run_mcmc(pos, 1000, progress = True)

samples = sampler.chain[:, 800:, :].reshape((-1, ndim))

########################################################################################################################


######################################## Plot the Parameter Posteriors #################################################

plt.rcParams.update({'font.size': 14})

best_param_mean = np.mean(samples, axis = 0)
best_param_median = np.median(samples, axis = 0)

cornerfig = corner.corner(samples, truths=best_param_median, quantiles=[0.16, 0.5, 0.84], show_titles=True)
cornerfig.subplots_adjust(top=0.94,
bottom=0.078,
left=0.063,
right=0.975,
hspace=0.05,
wspace=0.05)

cornerfig.show()

########################################################################################################################


################################ Plot the original spectra and the model fitted to them ################################

best_fit_emcee_mean =  lnlike(best_param_mean, just_give_sky_model= False, just_give_model=True)
best_fit_emcee_median = lnlike(best_param_median, just_give_sky_model= False, just_give_model=True)

fig = plt.figure(5, figsize = (32,16))

ax = plt.subplot(111)

ax.plot(wavelength, best_fit, 'g-', lw = 2)
ax.plot(wavelength, best_fit_emcee_mean, 'b-', lw = 2)
ax.plot(wavelength, best_fit_emcee_median, 'c-', lw = 2)
ax.plot(wavelength, science_data, 'r-', lw = 2)

plt.legend(('Best Fit - Optimiser', 'Best Fit - EMCEE Mean', 'Best Fit- EMCEE Median', 'Science Data'), prop={'size': 24})
ax.set_xlabel('Wavelength [$\AA$]')

plt.show()

########################################################################################################################

########################################## Plot the final spectra ######################################################

sky_optimiser = lnlike(best_fits_results[index], just_give_sky_model= True, just_give_model=False)
sky_emcee_mean = lnlike(best_param_mean, just_give_sky_model= True, just_give_model=False)
sky_emcee_median = lnlike(best_param_median,  just_give_sky_model= True, just_give_model=False)

ss_optimiser = science_data - sky_optimiser - best_fits_results[index][-2]
ss_emcee_mean =  science_data - sky_emcee_mean - best_param_mean[-2]
ss_emcee_median = science_data - sky_emcee_median - best_param_median[-2]

std_ss_optimiser = ss_optimiser
std_ss_emcee_mean = ss_emcee_mean
std_ss_emcee_median = ss_emcee_median

fig = plt.figure(7, figsize=(32,16))
ax = plt.subplot(111)

ax.plot(wavelength, ss_optimiser, 'g-', lw=2)
ax.plot(wavelength, ss_emcee_mean, 'b-', lw=2)
ax.plot(wavelength, ss_emcee_median, 'c-', lw=2)

plt.legend(("Final Spectrum - Optimiser", "Final Spectrum - EMCEE Mean", "Final Spectrum - EMCEE Median"), prop={'size': 24})
plt.show()

########################################################################################################################

############################################## Output best sky subtraction #############################################
if write_out == True:
    hdu = fits.PrimaryHDU(data=std_ss_optimiser)
    hdu.header['CRVAL1'] = wavelength[0]
    hdu.header['CDELT1'] = (wavelength[wavelength.size - 1] - wavelength[1]) / (wavelength.size - 2)
    hdu.header['CUNIT1'] = 'ANGSTROM'

    hdu.writeto(out_file_optim)


    hdu = fits.PrimaryHDU(data=std_ss_emcee_mean)
    hdu.header['CRVAL1'] = wavelength[0]
    hdu.header['CDELT1'] = (wavelength[wavelength.size - 1] - wavelength[1]) / (wavelength.size - 2)
    hdu.header['CUNIT1'] = 'ANGSTROM'

    hdu.writeto(out_file_emcee_mean)


    hdu = fits.PrimaryHDU(data=std_ss_emcee_median)
    hdu.header['CRVAL1'] = wavelength[0]
    hdu.header['CDELT1'] = (wavelength[wavelength.size - 1] - wavelength[1]) / (wavelength.size - 2)
    hdu.header['CUNIT1'] = 'ANGSTROM'

    hdu.writeto(out_file_emcee_median)

    print("Files Written Out")

########################################################################################################################

#Finish timing code
end = time.time()

runtime = end-start
print('Code Competed in: %.2f seconds' % runtime)