# -*- coding: utf-8 -*-
## this is needed for fmtsex function

#############################################################
######## Setup the environment
#############################################################

###Load packages
import os, glob, time, pickle, matplotlib, pdb, scipy, pyfits, emcee, astropysics, datetime, bisect, hashlib, itertools
from cosmocalc import cosmocalc
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import stats
import pystan, dump_rdata
from scipy import cluster
import pandas as pd

##fix for tex figures
from matplotlib import rc

##Locale
import locale

locale.setlocale(locale.LC_ALL, 'en_US.utf8')

rootdir = '/home/nes/Research/SNe/PSOs/IIps/'
photdir = rootdir + 'allphot/'
paperdir = rootdir + 'paper-flc_resubmit/'
paperdirml = rootdir + 'paper-ml_resubmit/'
pngdir = rootdir + 'png_resubmit/'
valfile = paperdir + 'values.tex'
ml_valfile = paperdirml + 'values.tex'
fdir = '/home/nes/Research/Common/Instruments/PS1/SDSS-PS1-PalomarQUEST_filters/'

## Continuous colorblind colormaps
execfile('/home/nes/Research/Software/Python/ColorblindColormaps/CBcm.py')

#############################################################
######## Useful functions
#############################################################


def numsafe(thestring):
    """ a simple function to replace numbers in a string with letters """
    return str(thestring).replace('0', 'zero').replace('1', 'one').replace(
        '2', 'two').replace('3', 'three').replace('4', 'four').replace(
            '5', 'five').replace('6', 'six').replace('7', 'seven').replace(
                '8', 'eight').replace('9', 'nine').replace('-', '').replace(
                    '/', '').replace(',', '')


def writetex(thefile, thename, thestring):
    """
    A simple function to append a latex macro to a file
    
    Parameters:
    * thefile: the tex file to append to
    * thename: The name of the macro
    * thestring: the string to write as the macro value
    """
    ##is the name safe?
    if thename.isalpha():
        ## Comment out prior lines with this macro
        nl = thename + '}'
        os.system(
            "grep -n \\" + thename + "} " + thefile +
            r" | grep -v '% \\newcommand' | sed -ne 's/ *:.*//p' | xargs -I nx sed -i 'nx s/^/% /' "
            + thefile)
        # Write new line
        f = open(thefile, 'a')
        timestamp = '  % ' + datetime.datetime.today().strftime(
            '%Y %B %d %H:%M:%S.%f')
        f.write('\\newcommand{\\' + thename + '}{' + str(thestring) + '}' +
                timestamp + '\n')
        f.close()
    else:
        raise NameError('No numbers allowed in macro name')


def textransfer(file_in, file_out, thename):
    """
    Function to transfer latex macros from one file to another, 
    commenting out old versions of the macro in the output file first
    
    Parameters:
    * file_in: the tex file to take the macro from
    * file_out: the tex file to append to
    * thename: The name of the macro
    """
    ## first comment out any pre-existing lines
    os.system(
        "grep -n \\" + thename + "} " + file_out +
        r" | grep -v '% \\newcommand' | sed -ne 's/ *:.*//p' | xargs -I nx sed -i 'nx s/^/% /' "
        + file_out)
    ## now transfer the line
    os.system("cat " + file_in + r" | grep -v '^%' | grep '\\" + thename +
              "}' >> " + file_out)


def texfig(fsize=(4, 4)):
    """
    A figure caller that will produce something with the right size and reasonable axes for a plot 
    for use in a latex document
    """
    thefig = plt.figure(figsize=fsize)
    theax = plt.axes()
    plt.subplots_adjust(left=0.16, bottom=0.13, right=0.93, top=0.94)
    return thefig, theax


def LCticks(axs, ynbins=7, myval=0.5):
    """
    Makes nice ticks for light curve plots (MJD vs magnitude)
    
    * myval: Magnitude spacing for y-axis
    """
    if type(axs) != numpy.ndarray: axs = [axs]
    the_xrange = [
        list(axs.ravel()[i].axis())[0:2] for i in range(len(axs.ravel()))
    ]
    for i in range(len(axs.ravel())):
        axs.ravel()[i].ticklabel_format(style='plain',
                                        axis='both',
                                        useOffset=False)
        axs.ravel()[i].xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
        axs.ravel()[i].xaxis.set_minor_locator(
            matplotlib.ticker.FixedLocator(
                arange(the_xrange[i][0] - 100, the_xrange[i][1] + 100, 5)))
        axs.ravel()[i].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=ynbins, prune='both'))
        axs.ravel()[i].yaxis.set_minor_locator(
            matplotlib.ticker.FixedLocator(arange(-30, 30, myval)))


def PrettyTicks(axs, N=[5, 5], mf=3, xprune='both', yprune='both'):
    """
    Makes nice ticks for generic plots, leaving log axes alone
    """
    if type(axs) != numpy.ndarray: axs = [axs]
    for i in range(len(axs)):
        axs[i].xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=N[0], prune=xprune))
        axs[i].xaxis.set_minor_locator(
            matplotlib.ticker.MaxNLocator(nbins=N[0] * mf, prune=xprune))
        axs[i].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=N[1], prune=yprune))
        axs[i].yaxis.set_minor_locator(
            matplotlib.ticker.MaxNLocator(nbins=N[1] * mf, prune=yprune))


# Each entry has: \lambda_eff, FWHM, zp(f_\lambda) erg/cm^2/s/AA, zp(f_\nu) erg/cm^2/s/hz (this one is not all filled out)
allbands={\
# UBVRIJHKL come from Bessel+ 1998

'U':[3663.,650.,4.175e-9,0.770e-20],\
'B':[4361.,890.,6.32e-9,-0.120],\
'V':[5448.,840.,3.631e-9,0],\
'R':[6407.,1580.,2.177e-9,0.186e-20],\
'I':[7980.,1540.,1.126e-9,0.444e-20],\
'J':[12500.,3300-1700.,3.147e-10,0.899e-20],#based on http://www.ifa.hawaii.edu/~tokunaga/filterSpecs.html and http://www.astro.umd.edu/~ssm/ASTR620/mags.html \
'H':[16350.,7800.-4900,1.138e-10,1.379e-20],\
'K':[22000.,3700-300.,3.961e-11,1.886e-20],\
'L':[34500.,4730,6.775e-11,2.765e-20],\
'u':[3596.,570.],\
'g':[4639.,1280.],\
'r':[6122.,1150.],\
'i':[7439.,1230.],\
'z':[8896.,1070.],\
'y':[10005.,1000], #from wikipedia, effective width is just a guess...
'W2':[1928.,657], #from Poole 2008MNRAS.383..627P\
'M2':[2246.,498],\
'W1':[2600.,693],\
'GHz 5.9':[3e18/5.9e9,nan],\
'FUV':[1528,(1786-1344)], # Galex, from Morrisey+ 2004, Table 3, the 'FWHM' column is actually Morrisey's 'bandwidth'
'NUV':[2271,(2831-1771)],
          }


def toUBVRI(u, g, r, i, z, AB=1):
    """
    A function to convert SDSS ugriz magnitudes to UBVRI (IN AB MAGNITUDES) using Table 2 of Blanton and Roweis 2007
    http://adsabs.harvard.edu/abs/2007AJ....133..734B
    This conversion is derived from galaxies (and may be different for other spectral types)...
    
    The AB output mags can be converted to Vega using the AB switch
    This conversion uses Table 1 of Blanton & Roweis (2007)
    """
    U = u - 0.0682 - 0.0140 * ((u - g) - 1.2638)
    B = g + 0.2354 + 0.3915 * ((g - r) - 0.6102)
    V = g - 0.3516 - 0.7585 * ((g - r) - 0.6102)
    R = r - 0.0576 - 0.3718 * ((r - i) - 0.2589)
    I = i - 0.0647 - 0.7177 * ((i - z) - 0.2083)

    if AB == 0:  # Convert to Vega
        U += -0.79
        B += 0.09
        V += -0.02
        R += -0.21
        I += -0.45

    return [U, B, V, R, I]


def toUBVRI_99em(u, g, r, i, z, AB=0):
    """
    A function to convert PS1 ugriz magnitudes to UBVRI in Vega magnitudes (actually, just V and I band)
    using SN IIP S-corrections calculated based on a +37 day SN 1999em spectrum from Leonard+2002
    
    A linear color term (based on r-i color for I-band and g-r color for V-band)
    
    See 4/4/2014 notes for details on derivations of these laws
    """
    U = nan
    B = nan
    V = g + -0.4218 * ((g - r) - -0.2702)
    R = nan
    I = i + -0.4399 * ((r - i) - -0.9505)

    if AB == 1:
        raise NotImplementedError

    return [U, B, V, R, I]


def bandf(mag,
          magerr,
          band,
          ftype=None,
          mode='lambda',
          units='cgs',
          z=0.,
          N=1e4):
    """
    Given a magnitude and band, convert to a flux density in ftype ('AB', 'Vega', or 'muJy') 
    and ergs/cm^2/s/AA (mode='lambda') or ergs/cm^2/s/Hz (mode='nu')
    
    If ftype is not specified, makes reasonable assumptions
    """
    ##Guest AB/Vega
    if ftype == None:
        if band in ['u', 'g', 'r', 'i', 'z']: ftype = 'AB'
        elif band in ['U', 'V', 'B', 'R', 'I', 'J', 'H', 'K', 'Y']:
            ftype = 'Vega'
        elif 'GHz' in band:
            ftype = 'muJy'
        else:
            ftype = 'AB'
    mag_MC = random.normal(mag, magerr, N)
    if ftype == 'AB':
        if mode == 'lambda':
            flux = 10**(
                (mag_MC + 48.574) / -2.5) * (2.998e18) / (allbands[band][0])**2
        elif mode == 'nu':
            flux = 10**((mag_MC + 48.574) / -2.5)
        else:
            print "bandf: Mode not recognized"
            flux = [nan]
    elif ftype == 'Vega':
        if mode == 'lambda':
            zp = allbands[band][2]
            #flux=10**((mag_MC+21.1+zp)/-2.5)
            flux = 10**((mag_MC) / -2.5) * zp
        elif mode == 'nu':
            zp = allbands[band][3]
            flux = 10**(
                (mag_MC) / -2.5) * zp * (2.998e18) / (allbands[band][0])**2
            #flux=10**((mag_MC+48.598+zp)/-2.5)
        else:
            print "bandf: Mode not recognized"
            flux = [nan]
    elif ftype == 'muJy':
        if mode == 'lambda':
            flux = 1e-17 * mag_MC * 3e10
        elif mode == 'nu':
            flux = 1e-17 * mag_MC
        else:
            print "bandf: Mode not recognized"
            flux = [nan]
    else:
        print "bandf: ftype not recognized"
        flux = [nan]
    if units == 'cgs': flux *= 1
    elif units == 'kms':
        if mode == 'lambda': flux *= 1e-7 * 100**2 * 1e-4
        elif mode == 'nu': flux *= 1e-7 * 100**2
    else: print "bandfunits not recognized"
    ##Redshift correction
    flux = flux / (1 + z)
    return [median(flux), std(flux)]


def CCMextinct(l, EBmV, Rv=3.1, mag=0):
    """
    Calculate flux supression at a given wavelength based on the M.W. extinction law of
    Cardelli, Clayton, and Mathis (1989ApJ...345..245C).
    
    Parameters:
    * l: Wavelength (list/array, in Angstroms)
    * EBmV: Redenning E(B-V)
    * Rv: Ratio of total to selective extinction
    * mag: Return A_\lambda (mag) instead of flux multiplier
    """
    ## Calculate inverse microns
    x = 1 / (array(l) / 1e4)
    y = x - 1.82
    ## Calculate coefficients at each wavelength
    a = zeros(len(x))
    b = zeros(len(x))
    ### IR -> Optical
    sel = where((x > 0.3) & (x < 1.1))
    a[sel] = 0.574 * x[sel]**1.61
    b[sel] = -0.527 * x[sel]**1.61
    sel = where((x > 1.1) & (x < 3.3))
    a[sel] = 1 + 0.17699 * y[sel] - 0.50447 * y[sel]**2 - 0.02427 * y[
        sel]**3 + 0.72085 * y[sel]**4 + 0.01979 * y[sel]**5 - 0.77530 * y[
            sel]**6 + 0.32999 * y[sel]**7
    b[sel] = 1.41338 * y[sel] + 2.28305 * y[sel]**2 + 1.07233 * y[
        sel]**3 - 5.38434 * y[sel]**4 - 0.62251 * y[sel]**5 + 5.30260 * y[
            sel]**6 - 2.09002 * y[sel]**7
    ### UV and FUV
    Fa = -0.04473 * (x - 5.9)**2 - 0.009779 * (x - 5.9)**3
    Fb = 0.2130 * (x - 5.9)**2 + 0.1207 * (x - 5.9)**3
    sel = where(x < 5.9)
    Fa[sel] = zeros(len(sel[0]))
    Fb[sel] = zeros(len(sel[0]))
    sel = where((x > 3.3) & (x < 8))
    a[sel] = 1.752 - 0.316 * x[sel] - 0.104 / (
        (x[sel] - 4.67)**2 + 0.341) + Fa[sel]
    b[sel] = -3.090 + 1.825 * x[sel] + 1.206 / (
        (x[sel] - 4.62)**2 + 0.263) + Fb[sel]
    sel = where((x > 8) & (x < 10))
    a[sel] = -1.073 - 0.628 * (x[sel] - 8) + 0.137 * (
        x[sel] - 8)**2 - 0.070 * (x[sel] - 8)**3
    b[sel] = 13.670 + 4.257 * (x[sel] - 8) - 0.420 * (
        x[sel] - 8)**2 + 0.374 * (x[sel] - 8)**3
    ##Calculate extinction in magnitudes
    AV = EBmV * Rv
    Al = (a + b / Rv) * (AV)
    if mag:
        return Al
    else:
        return 10**(Al / 2.5)


def distmod(z, H0=70):
    """
    Distance modulus assuming standard cosmology (using cosmocalc)
    H0 defaults to 70
    """
    return 5 - 5 * log10(cosmocalc(float(z), H0=H0, WM=0.27)['DL_Mpc'] * 1e6)


def pgrepv(tfile, outfile=None, hcharacter='#', checkcol=6):
    """
    Pythonic version of grep one liner to remove comments / header from a file.
    Meant for use with CmdStan output files
    
    Returns a string corresponding to tfile with any lines starting with hcharacter removed.
    
    if outfile is specified, writes output to that file instead
    
    If checkcol is !=0, makes sure that checkcol is an integer value to weed out bad rows.  
	checkcol=6 by default (note that it starts at 1, not 0 - awk format, not python)
	, checking the n_divergent value
    """
    with open(tfile, 'r') as fn:
        lines = fn.readlines()
        lines2 = []
        for line in lines:
            if line[0] != hcharacter and line != '\n' and (
                    checkcol == 0 or
                (line[0:5] == 'lp__,'
                 or len(line.split(',')[checkcol - 1].strip()) == 1)):
                lines2.append(line)
    outdata = ''.join(lines2)
    if outfile == None:
        return outdata
    else:
        with open(outfile, 'w') as fn_out:
            fn_out.write(outdata)


def myloadcsv(fn):
    """
    CSV parser file
    Assumes the first line is the header
    """
    ## Open file
    with open(fn, 'r') as fno:
        outdict = {}
        i = 0
        ## walk through file
        for line in fno.readlines():
            ## parse header
            if i == 0:
                header = line.strip('\n').split(',')
                for v in header:
                    outdict[v] = []
            ## parse data
            else:
                data = line.split(',')
                ## Check for bad data
                if len(data) != len(header):
                    print('myloadcsv ' + fn + ': line #' + str(i) +
                          ' is bad, ignoring')
                else:
                    ##Record data
                    for j in range(len(header)):
                        outdict[header[j]].append(float(data[j]))
            i += 1
    ## convert to arrays
    for j in range(len(header)):
        outdict[header[j]] = array(outdict[header[j]])
    return outdict


def texUpDown(x, n=2):
    """
    Take a list of distribution summary statistics formatted like [-1sigma,median,+1sigma] 
    and spit out a tex-formatted string like x^{+sig}_{-sig}
    where each element is formatted like %0.nf where n is specified as a parameter
    """
    if sum(isnan(array(x))) > 0:
        return '\\ldots'
    else:
        if n < 0:
            x = around(x, n)
            n = 0
        fstring = '0.' + str(n) + 'f'
        return ('%' + fstring) % x[1] + '^{+' + (
            '%' + fstring) % x[2] + '}_{-' + ('%' + fstring) % x[0] + '}'


def texUpDown_dist(x, n=2):
    """
    Helper function for texUpDown that takes a distribution array directly instead of percentile values
    """
    vs = percentile(x, [16, 50, 84])
    xd = [vs[1] - vs[0], vs[1], vs[2] - vs[1]]
    return texUpDown(xd, n=n)


def savedict(d, froot):
    """
    Save a python dictionary (d) of 2D arrays or single values to a series of text files (one for each key)
    with prefix froot
    """
    for key in d:
        f = open(froot + '_' + key, 'w')
        if isscalar(d[key]):
            f.write(str(d[key]) + '\n')
        else:
            for i in range(len(d[key])):
                if isscalar(d[key][i]): line = str(d[key][i]) + '\n'
                else: line = ','.join([str(o) for o in d[key][i]]) + '\n'
                f.write(line)
        f.close()


def rstd(x, axis=0):
    """
    Calculate 86 - 14th percentile value for a distribution x
    
    This corresponds to std for a gaussian distribution
    """
    return (percentile(x, 84, axis=axis) - percentile(x, 16, axis=axis)) / 2.


#############################################################
######## SN IIP K-corrections
#############################################################

## load Nugent templates
Nugent_d = np.genfromtxt(
    '/home/nes/Research/SNe/PSOs/IIps/Nugent/sn2p_flux.v1.2.dat',
    names=['t', 'l', 'f'])
Nugent_t = unique(Nugent_d['t'])
### interpolate over times
#Nugent_m=interp2d(Nugent_d['t'],Nugent_d['l'],Nugent_d['f'])

## load PS1 filter curves
PS1fc = {}
for b in 'grizy':
    PS1fc[b] = np.genfromtxt(fdir + b + '_PS1.res', names=['l', 'f'])

## Load the Bessel / Johnson-Cousins filters from http://spiff.rit.edu/classes/phys440/lectures/filters/filters.html
BESSfc = {}
for b in 'UBVRI':
    BESSfc[b] = np.genfromtxt(rootdir + 'literature/Bessel/bess-' + b.lower() +
                              '.pass',
                              names=['l', 'f'])


###### For S corrections
def calc_Nug_color(band1, band2, Nday=61., tilt=0., altspec=None):
    ls = arange(3000, 10000, 5)
    ## Load filters
    VegaF = {'U': -0.79, 'B': 0.09, 'V': -0.02, 'R': -0.21, 'I': -0.45}
    if band1.lower() == band1:
        filt1 = interp1d(PS1fc[band1]['l'],
                         PS1fc[band1]['f'],
                         fill_value=0,
                         bounds_error=0)
        mod1 = 0
    else:
        filt1 = interp1d(BESSfc[band1]['l'],
                         BESSfc[band1]['f'],
                         fill_value=0,
                         bounds_error=0)
        mod1 = VegaF[band1]
    if band2.lower() == band2:
        filt2 = interp1d(PS1fc[band2]['l'],
                         PS1fc[band2]['f'],
                         fill_value=0,
                         bounds_error=0)
        mod2 = 0
    else:
        filt2 = interp1d(BESSfc[band2]['l'],
                         BESSfc[band2]['f'],
                         fill_value=0,
                         bounds_error=0)
        mod2 = VegaF[band1]
    ## Tilt it
    tilt_x0 = mean([allbands[band1][0], allbands[band2][0]])
    tilt_f = lambda A, x: A * (x - tilt_x0)
    ## Load SNe IIP template
    if altspec == None:
        Nsel = where(Nugent_d['t'] == Nday)
        Nx = Nugent_d['l'][Nsel]
        Ny = Nugent_d['f'][Nsel]
    else:
        Sdata = np.genfromtxt(altspec, names=['l', 'f'])
        Nx = Sdata['l']
        Ny = Sdata['f']
    SNIIPf = interp1d(Nx,
                      Ny * (Nx)**2 * 10**tilt_f(tilt, Nx),
                      fill_value=0,
                      bounds_error=0)
    color = -2.5 * log10(
        sum(filt1(ls) * SNIIPf(ls) / sum(filt1(ls))) /
        sum(filt2(ls) * SNIIPf(ls) / sum(filt2(ls)))) + mod1 - mod2
    return color


def calc_Nug_Scor(band1,
                  band2,
                  c_band1,
                  c_band2,
                  target_color,
                  Nday=61.,
                  altspec=None):
    """
    Calculate S correction between band1 and band2 using the Nugent, using the calc_Nug_color function
    
    Tilts the spectrophotometric template to match the target_color specified between c_band1 and c_band2
    
    If altspec is specified, uses a different flambda spectral file instead of the Nugent template
    """
    opt_tilt_f = lambda x: target_color - calc_Nug_color(
        c_band1, c_band2, Nday=Nday, tilt=x, altspec=altspec)
    opt_tilt = fsolve(opt_tilt_f, 0)
    Scor = calc_Nug_color(band1,
                          band2,
                          Nday=61.,
                          tilt=opt_tilt,
                          altspec=altspec)
    return Scor


###### K corrections
VI_color_law = lambda test_color, x: (
    (test_color / 0.1) * -0.121) * (x - 10000) / 10000


def Kcor_IIP(b, z, t, test_color=0, A_V=None):
    """
    Calculate K-corrections for PS1 filters using the SN IIP spectral templates from P. Nugent
    
    Parameters:
    * b: Band of observation (grizy)
    * z: Redshift of SN
    * t: Epoch (rest frame days since explosion)
    * test_color: V-I color difference (in mag) to apply to the Nugent models 
      to test the effect of intrinsic color variation (see 4/3/2014 notes and then the update on 11/8/2014)
    * A_V: Extinction to apply to the Nugent template
    
    If t<0, returns 0
    If t>411 (last of Nugent models), returns t=411 value
    """
    if t < 0: return 0
    elif t > max(Nugent_t): t = max(Nugent_t) - 0.01
    ## Load PS1 filter curve
    l = PS1fc[b]['l']
    R = PS1fc[b]['f']
    ## Load Nugent template interpolated to nearest time
    N_N = len(Nugent_d['l'][Nugent_d['t'] == 0])
    N_l = np.zeros(N_N)
    N_f = np.zeros(N_N)
    N_l = Nugent_d['l'][where(Nugent_d['t'] == 0.)]
    t1 = Nugent_t[bisect.bisect(Nugent_t, t) -
                  1]  # the last Nugent_t value less than t
    t2 = Nugent_t[bisect.bisect(Nugent_t, t)]
    f1 = Nugent_d['f'][Nugent_d['t'] ==
                       t1]  # the flux corresponding to N_l at t1
    f2 = Nugent_d['f'][Nugent_d['t'] == t2]
    N_f = f1 + (f2 - f1) * (t - t1) / (t2 - t1)
    ## Apply color difference
    f_color_test_cor = VI_color_law(test_color, N_l)
    ## Apply extinction
    if A_V is not None:
        extinct_curve = CCMextinct(N_l, A_V / 3.1, Rv=3.1, mag=0)
    else:
        extinct_curve = np.ones(len(N_l))
    ## Interpolate Nugent template values to filter wavelength
    f = interp1d(N_l, N_f * (10**f_color_test_cor) / extinct_curve)
    ## Calculate K correction
    return -2.5 * log10(1 /
                        (1 + z) * sum(l * f(l) * R) / sum(l * f(l *
                                                                (1 + z)) * R))


def vKcor_IIP(b, z, t):
    """
    Vectorized version of Kcor_IIP, so you can pass an array of times
    
    Parameters:
    * b: Band of observation (grizy)
    * z: Redshift of SN
    * t: Epoch (rest frame days since explosion)
    
    If t<0, returns 0
    If t>411 (last of Nugent models), returns t=411 value    
    """
    ## Establish full template Kcorrection curve
    vK = vectorize(lambda x: Kcor_IIP(b, z, x))
    tt = arange(0, 411)
    tK = vK(tt)
    ## Interpolate model K cure
    fK = interp1d(tt, tK, bounds_error=False, fill_value=tK[-1])
    ## Calculate K correction
    out = fK(t)
    ## Fix out-of-bounds values
    out[t < 0] = 0.
    return out


#############################################################
######## Stan light curve model
#############################################################


### To avoid recompiling the stan model, establish a global variable to store the Pystan fit object
global compiled_stan_model
compiled_stan_model = None  # for a populated object, the format will be [checksum,fit object]

global compiled_stan_model_ind
compiled_stan_model_ind = None


def pystan_SPII_tm(SNdata,
                   stanmodel=(rootdir + 'stan/SIIP_t_multilevel.stan'),
                   chains=10,
                   iterations=1000,
                   Kcor=None,
                   get_neff=0,
                   stan_init='0',
                   verbose=0,
                   parallel=1,
                   fixhyper=0,
                   dump_data=0,
                   fluxoffset=1,
                   **kwargs):
    """
    Wrapper function to run a hierarchical Pystan fit on a set of SN light curve data.
    
    Ues pystan_SPII_tm_getLC to identify relevant data from observing seasons.
    
    Parameters:
    * SNdata: Record array with SN data (including fluxes, absolute magnitudes) in the aphot format
    
    Optional parameters:
    * stanmodel: model to fit (stan/C code; default: "SIIP.stan")
    * chains: Number of chains to run with stan (default: 10)
    * iterations: Iterations to run with stan (default: 1000)
    * stan_init: Initialize to zeros or random ('0' by default, or choose 'random' or 'myprior' for hand-coded initialization)
    * Kcor: An array of K corrections (shape = [N_SNe,N_filters,N_tsteps]) from t=0..N_tsteps days with steps = 1.0 days.  If not given, zeros are used.
    * get_neff: Return effective sample statistics for each variable [0,1, or 'some' for just the sampling parameters dictionary]
    * verbose: Turn on verbose output from Stan
    * parallel: Instruct pystan to run this many chains in parallel
    * fixhyper: Fix the hyperparameter values (i.e. specify h_* values as data instead of parameter initializations).  Only usable with fixed initialization
    * dump_data: Output y_in to a stan data dump file of this name
    * fluxoffset: Divide fluxes by 1e7 before passing to Stan?
    
    Remaining arguments are passed to pystan.stan
    """
    #### Load cached model if available
    global compiled_stan_model
    with open(stanmodel, 'r') as obj:
        modeltext = obj.read()
    if compiled_stan_model != None and compiled_stan_model[0] == hashlib.md5(
            modeltext).hexdigest():
        print "pystan_SPII_tm: Past compiled model found and model file unchanged..."
        past_fit = compiled_stan_model[1]
    elif compiled_stan_model != None and compiled_stan_model[0] != hashlib.md5(
            stanmodel).hexdigest():
        past_fit = None
        print "pystan_SPII_tm: Past model found, but model file has changed; must recompile"
    else:
        print "pystan_SPII_tm: No past model found model; must recompile"
        past_fit = None
    #### Prepare data, if nibjs>0
    events = unique(SNdata['event'])
    if len(events) > 0:
        ## Pick observations in right time range and in optical bands
        #sel_b=((SNdata['band']=='g') | (SNdata['band']=='r') | (SNdata['band']=='i') | (SNdata['band']=='z') | (SNdata['band']=='y'))
        dp = [SPII_dpeak_first(name) for name in events]
        dp_all = np.zeros(len(SNdata))
        for i in range(len(SNdata)):
            dp_all[i] = dp[where(events == SNdata['event'][i])[0]]
        #sel=where(sel_b & (SNdata['MJD']>(dp_all-dpeak_a)) & (SNdata['MJD']<(dp_all+dpeak_b)))
        SNdata2 = []
        for i in range(len(events)):
            SNdata2.append(
                pystan_SPII_tm_getLC(events[i], dp[i], bands='grizy'))
        SNdata2 = np.hstack(SNdata2)
        ## Exclude null fluxes
        sel = where(SNdata2['raw_flux'] != 0)
        SNdata2 = SNdata2[sel]
        ## Get redshift list
        z = [
            SNdata2['z'][where(SNdata2['event'] == event)[0][0]]
            for event in events
        ]
        ## Get zero point lists
        old_mzs = np.zeros([len(events), 5])
        for k in range(len(events)):
            for j in range(5):
                try:
                    old_mzs[k, j] = get_r_mzero(events[k],
                                                dp[k],
                                                band='grizy'[j])
                except:
                    old_mzs[k, j] = 0
        mzs = fix_mzs(old_mzs)
        ## Get filter list
        J = intbands(SNdata2['band'])
        ## Get distance modulus list
        dms = [distmod(z[i]) for i in range(len(z))]
        ## Get SN ID list
        K = np.zeros(len(SNdata2), dtype=int)
        Kdic = {}
        for i in range(1, 1 + len(events)):
            Kdic[events[i - 1]] = i
        for i in range(len(SNdata2)):
            K[i] = Kdic[SNdata2['event'][i]]
        ## Check Kcor list
        if Kcor == None:
            Kcor = np.zeros([len(SNdata2), 5, 200])
        ## Calculate duringseason predictors
        dseason = pystan_SPII_tm_getdseason(events)
        ## Convert flux to luminosity
        dms_byN = array([dms[K[k] - 1] for k in range(len(K))])
        zm_byN = array([mzs[K[k] - 1, J[k] - 1] for k in range(len(K))])
        magmod = dms_byN + zm_byN
        fL = SNdata2['raw_flux'] * 10**(magmod / -2.5)
        dfL = SNdata2['raw_flux_err'] * 10**(magmod / -2.5)
        ## Put in stan input format
        if fluxoffset: dby = 1e7
        else: dby = 1
        y_in = {
            'N_obs': len(SNdata2),
            'N_SN': len(events),
            'N_filt': 5,
            't': SNdata2['MJD'],
            'fL': fL / dby,
            'dfL': dfL / dby,
            'z': array(z),
            't0_mean': array(dp),
            'mzero': array(mzs),
            'distmod': array(dms),
            'J': J,
            'SNid': K,
            'Kcor_N': shape(Kcor)[2],
            'Kcor': Kcor,
            'fluxscale': 1e7,  # roughly percentile(fL,90)
            'duringseason': dseason
        }
    else:  #no data
        y_in = {
            'N_obs': 1,
            'N_SN': 1,
            'N_filt': 5,
            't': np.zeros([1]),
            'f': np.zeros([1]),
            'df': np.ones([1]),
            'z': np.zeros([1]),
            't0_mean': np.zeros([1]),
            'mzero': np.zeros([1, 5]),
            'distmod': np.zeros([1]),
            'J': np.ones([1], dtype='int'),
            'SNid': np.ones([1], dtype='int'),
            'Kcor_N': 1,
            'Kcor': np.zeros([1, 5, 1]),
            'fluxscale': 1,
            'duringseason': np.zeros([1], dtype='int')
        }

    if dump_data != 0 and 'pickle' in dump_data:
        with open(dump_data, 'w') as f:
            pickle.dump(y_in, f)
    elif dump_data != 0:
        with open(dump_data, 'w') as f:
            dump_rdata.dump_to_rdata(f, **y_in)

    ## Initialize parameter values
    if stan_init == 'myprior':
        stan_init = pystan_SPII_tm_init  # backwards compatibility
    if stan_init == 0 or stan_init == '0':
        sinit = 0
    elif stan_init == 'random':
        sinit = 'random'
    elif hasattr(stan_init,
                 '__call__'):  ## check that a function has been passed
        if fixhyper:
            null, hyperinit = stan_init(len(events), fixhyper=1)
            for key in hyperinit:
                y_in[key] = hyperinit[key]
            sinit = lambda: stan_init(len(events), fixhyper=2)
        else:
            sinit = lambda: stan_init(len(events), fixhyper=0)
    else:
        raise Exception("Unrecognized init mode")

    #### Do sampling
    fit_params = {
        "data": y_in,
        "iter": iterations,
        "chains": chains,
        "verbose": verbose,
        "init": sinit,
        "n_jobs": parallel
    }
    for key in fit_params:
        kwargs[key] = fit_params[key]

    if past_fit == None:
        fit = pystan.stan(file=stanmodel, **kwargs)
        checksum = hashlib.md5(modeltext).hexdigest()
        compiled_stan_model = [checksum, fit]
    else:
        fit = pystan.stan(fit=past_fit, **kwargs)

    ## Get neff and other sampler statistics if desired
    if get_neff == 1:
        all_neff = [
            pystan.chains.ess(fit.sim, i)
            for i in range(len(fit.sim['fnames_oi']))
        ]
        neffs = {k: (v) for k, v in zip(fit.sim['fnames_oi'], all_neff)}
        ## Add in rhat
        ## avoid zerodivisionerror
        all_rhat = [nan] * len(fit.sim['fnames_oi'])
        for i in range(len(fit.sim['fnames_oi'])):
            try:
                all_rhat[i] = pystan.chains.splitrhat(fit.sim, i)
            except ZeroDivisionError:
                all_rhat[i] = nan
        for k, v in zip(fit.sim['fnames_oi'], all_rhat):
            neffs['rhat_' + k] = v
        ndiv = [
            fit.get_sampler_params()[c]['n_divergent__'] for c in range(chains)
        ]
        return [events, fit.extract(), [neffs, ndiv, fit.get_sampler_params()]]
    elif get_neff == 'some':
        return [events, fit.extract(), fit.get_sampler_params()]
    elif get_neff == 0:
        return [events, fit.extract()]


def pystan_SPII_tm_ind(SNdata,
                       stanmodel=(rootdir +
                                  'stan/SIIP_t_multilevel_mod5_lin_ind.stan'),
                       chains=10,
                       iterations=1000,
                       Kcor=None,
                       get_neff=0,
                       stan_init='0',
                       verbose=0,
                       parallel=1,
                       dump_data=0,
                       fluxoffset=1,
                       **kwargs):
    """
    Wrapper function to run a non-hierarchical, individual SN Pystan fit on light curve data.
    
    Uses pystan_SPII_tm_getLC to identify relevant data from observing seasons.
    
    Parameters:
    * SNdata: Record array with SN data (including fluxes, absolute magnitudes) in the aphot format
    
    Optional parameters:
    * stanmodel: model to fit (stan/C code; default: "SIIP.stan")
    * chains: Number of chains to run with stan (default: 10)
    * iterations: Iterations to run with stan (default: 1000)
    * stan_init: Initialize to zeros or random ('0' by default, or choose 'random' or 'myprior' for hand-coded initialization)
    * Kcor: An array of K corrections (shape = [N_SNe,N_filters,N_tsteps]) from t=0..N_tsteps days with steps = 1.0 days.  If not given, zeros are used.
    * get_neff: Return effective sample statistics for each variable [0,1, or 'some' for just the sampling parameters dictionary]
    * verbose: Turn on verbose output from Stan
    * parallel: Instruct pystan to run this many chains in parallel
    * dump_data: Output y_in to a stan data dump file of this name
    * fluxoffset: Divide fluxes by 1e7 before passing to Stan?
    
    Remaining arguments are passed to pystan.stan
    """
    #### Load cached model if available
    global compiled_stan_model_ind
    with open(stanmodel, 'r') as obj:
        modeltext = obj.read()
    if compiled_stan_model_ind != None and compiled_stan_model_ind[
            0] == hashlib.md5(modeltext).hexdigest():
        print "pystan_SPII_tm: Past compiled model found and model file unchanged..."
        past_fit = compiled_stan_model_ind[1]
    elif compiled_stan_model_ind != None and compiled_stan_model_ind[
            0] != hashlib.md5(stanmodel).hexdigest():
        past_fit = None
        print "pystan_SPII_tm: Past model found, but model file has changed; must recompile"
    else:
        print "pystan_SPII_tm: No past model found model; must recompile"
        past_fit = None
    #### Prepare data, if nibjs>0
    events = unique(SNdata['event'])
    if len(events) > 1:
        Exception("pystan_SPII_tm_ind: Multiple events submitted??")
    else:
        name = events[0]
    ## Pick observations in right time range and in optical bands
    dp = SPII_dpeak_first(name)
    SNdata2 = pystan_SPII_tm_getLC(name, dp, bands='grizy')
    ## Exclude null fluxes
    sel = where(SNdata2['raw_flux'] != 0)
    SNdata2 = SNdata2[sel]
    ## Get redshift
    z = SNdata2['z'][0]
    ## Get zero point lists
    old_mzs = np.zeros([5])
    for j in range(5):
        try:
            old_mzs[j] = get_r_mzero(name, dp, band='grizy'[j])
        except:
            old_mzs[j] = 0
    mzs = fix_mzs(old_mzs)
    ## Get filter list
    J = intbands(SNdata2['band'])
    J = J.flatten()
    ## Get distance modulus
    dms = distmod(z)
    ## Check Kcor list
    if Kcor == None:
        Kcor = np.zeros([5, 200])
    ## Calculate duringseason predictors
    dseason = pystan_SPII_tm_getdseason([name])[0]
    ## Convert flux to luminosity
    zm_byN = array([mzs[J[k] - 1] for k in range(len(SNdata2))])
    magmod = dms + zm_byN
    fL = SNdata2['raw_flux'] * 10**(magmod / -2.5)
    dfL = SNdata2['raw_flux_err'] * 10**(magmod / -2.5)
    ## Put in stan input format
    if fluxoffset: dby = 1e7
    else: dby = 1
    y_in = {
        'N_obs': len(SNdata2),
        'N_filt': 5,
        't': SNdata2['MJD'],
        'fL': fL / dby,
        'dfL': dfL / dby,
        'z': z,
        't0_mean': dp,
        'mzero': array(mzs),
        'distmod': dms,
        'J': J,
        'Kcor_N': shape(Kcor)[1],
        'Kcor': Kcor,
        'fluxscale': 1e7,  # roughly percentile(fL,90)
        'duringseason': dseason
    }

    if dump_data != 0 and 'pickle' in dump_data:
        with open(dump_data, 'w') as f:
            pickle.dump(y_in, f)
    elif dump_data != 0:
        with open(dump_data, 'w') as f:
            dump_rdata.dump_to_rdata(f, **y_in)

    ## Initialize parameter values
    if stan_init == 'myprior':
        stan_init = pystan_SPII_tm_init  # backwards compatibility
    if stan_init == 0 or stan_init == '0':
        sinit = 0
    elif stan_init == 'random':
        sinit = 'random'
    elif hasattr(stan_init,
                 '__call__'):  ## check that a function has been passed
        if fixhyper:
            null, hyperinit = stan_init(len(events), fixhyper=1)
            for key in hyperinit:
                y_in[key] = hyperinit[key]
            sinit = lambda: stan_init(len(events), fixhyper=2)
        else:
            sinit = lambda: stan_init(len(events), fixhyper=0)
    else:
        raise Exception("Unrecognized init mode")

    #### Do sampling
    fit_params = {
        "data": y_in,
        "iter": iterations,
        "chains": chains,
        "verbose": verbose,
        "init": sinit,
        "n_jobs": parallel
    }
    for key in fit_params:
        kwargs[key] = fit_params[key]

    if past_fit == None:
        fit = pystan.stan(file=stanmodel, **kwargs)
        checksum = hashlib.md5(modeltext).hexdigest()
        compiled_stan_model_ind = [checksum, fit]
    else:
        fit = pystan.stan(fit=past_fit, **kwargs)

    ## Get neff and other sampler statistics if desired
    if get_neff == 1:
        all_neff = [
            pystan.chains.ess(fit.sim, i)
            for i in range(len(fit.sim['fnames_oi']))
        ]
        neffs = {k: (v) for k, v in zip(fit.sim['fnames_oi'], all_neff)}
        ## Add in rhat
        ## avoid zerodivisionerror
        all_rhat = [nan] * len(fit.sim['fnames_oi'])
        for i in range(len(fit.sim['fnames_oi'])):
            try:
                all_rhat[i] = pystan.chains.splitrhat(fit.sim, i)
            except ZeroDivisionError:
                all_rhat[i] = nan
        for k, v in zip(fit.sim['fnames_oi'], all_rhat):
            neffs['rhat_' + k] = v
        ndiv = [
            fit.get_sampler_params()[c]['n_divergent__'] for c in range(chains)
        ]
        return [events, fit.extract(), [neffs, ndiv, fit.get_sampler_params()]]
    elif get_neff == 'some':
        return [events, fit.extract(), fit.get_sampler_params()]
    elif get_neff == 0:
        return [events, fit.extract()]


def pystan_SPII_tm_getLC(name,
                         dpeak,
                         bands='grizy',
                         btwdays=50.,
                         remove_dupes=0):
    """
  Pick out light curve points from the full SN observing seasons before, during, and after a SN.
  
  Uses the aphot database
  
  Parameters:
  * name: SN event name
  * dpeak: MJD date of SN peak, e.g. as returned by SPII_dpeak_first
  * bands: Filter bands to return from aphot
  * btwdays: Number of days considered to seperate observing seasons
  * remove_dupes: If >0, remove any duplicate points, using this value as the window (in days) to identify them
	    (useful for filtering out objects subtracted against multiple templates)
	    (This feature assumes each point will have at most one duplicate)
  """
    out_sel = []
    ## step through bands
    for b in bands:
        isel = where((aphot['event'] == name) & (aphot['band'] == b))
        if len(isel[0]) > 0:
            ## determine observing season edges
            delay = aphot['MJD'][isel][1:] - aphot['MJD'][isel][:-1]
            pivot = where(delay > btwdays)[0] + 1
            ## assign season IDs
            sIDs = [sum(i + 1 > pivot) for i in range(len(isel[0]))]
            ## select data in and around the SN season
            SNsID = sIDs[argmin(abs(dpeak - aphot['MJD'][isel]))]
            band_out_sel = where((sIDs == SNsID) | (sIDs == SNsID - 1)
                                 | (sIDs == SNsID + 1))
            ## add to output
            out_sel.append(isel[0][band_out_sel[0]])
    ## Filter duplicates
    if len(out_sel) == 0:
        return aphot[[]]
    else:
        outdata = aphot.copy()[hstack(out_sel)]
        if remove_dupes > 0:
            ## step through bands:
            for b in bands:
                isel = where(outdata['band'] == b)[0]
                ## sort
                ts = outdata['MJD'][isel]
                ao = argsort(ts)
                ## find similar timestamped points
                toffset = ts[ao][1:] - ts[ao][:-1]
                dupe_sel = where(toffset < remove_dupes)[0]
                # Average the duplicates along the numeric columns - not updating the values for some reason
                #for key in ['MJD','raw_flux','raw_flux_err','flux','flux_err','m','dm']:
                #pdb.set_trace()
                #outdata[isel][ao][key][dupe_sel] = mean([outdata[isel][ao][key][dupe_sel],outdata[isel][ao][key][1:][dupe_sel]],axis=0)
                # Remove the duplicates
                outdata = np.delete(outdata, isel[ao][dupe_sel + 1], axis=0)
            return outdata
        else:
            return outdata


def pystan_SPII_tm_getfit(sout, name):
    """
    Given the output from pystan_SPII_tm (sout = [events, fit]), return the fit samples for a particular SN (name)
    """
    [events, fit] = sout
    ## identify which SN this was within the fit (k=??)
    sid = list(events).index(name)
    s_fit = fit.copy()
    if 'alpha' in s_fit:
        fix_keys = [
            'pt0', 't1', 'tp', 't2', 'td', 'alpha', 'beta1', 'beta2', 'betadN',
            'betadC', 'Yb', 'V', 'Mp', 'M1', 'M2', 'Md', 'mpeak', 't0'
        ]  #,'Mpr'
        for key in fix_keys:
            if len(shape(s_fit[key])) == 3:
                s_fit[key] = s_fit[
                    key][:, :,
                         sid]  # note that new linear model has N_SN and N_F switched, as of 3/2014
            else:
                s_fit[key] = s_fit[key][:, sid]
    elif 'lalpha' in s_fit:
        fix_keys = [
            'pt0', 't1', 'tp', 't2', 'td', 'lalpha', 'lbeta1', 'lbeta2',
            'lbetadN', 'lbetadC', 'Yb', 'V', 'Mp', 'M1', 'M2', 'Md', 'mpeak',
            't0'
        ]  #,'Mpr'
        for key in fix_keys:
            s_fit[key] = s_fit[key][:, sid]
    return s_fit


def pystan_SPII_tm_init(Nsn, Nfilt=5, fixhyper=0):
    """
    Generate reasonable initial parameters for the SNIIP hierarchical light curve model
    
    Parameters:
    * Nsn: Number of SNe in the model
    * Nfilt: Number of filters (default=5)
    * fixhyper: If 1, return hyperparameters in a separate dictionary; if 2, do not return them
    """
    params = {}
    params['pt0'] = random.normal(0, 1, Nsn)
    params['raw_t1'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_t2'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_td'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_tplateau'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_lalpha'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_lbeta1'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_lbeta2'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_lbetadN'] = random.normal(0, .2, [Nsn, Nfilt])
    params['raw_lbetadC'] = random.normal(0, .2, [Nsn, Nfilt])
    params['Mpr'] = random.normal(0, .5, [Nsn, Nfilt])
    params['Yb'] = random.normal(0, 100, [Nsn, Nfilt])
    params['V'] = abs(random.normal(100, 10, [Nsn, Nfilt]))
    params['h_t1'] = abs(random.normal(3, .1, Nfilt))
    params['h_t1_s'] = abs(random.normal(.5, .1, Nfilt))
    params['h_t2'] = abs(random.normal(70, .5, Nfilt))
    params['h_t2_s'] = abs(random.normal(20, .1, Nfilt))
    params['h_tplateau'] = 80 + abs(random.normal(20, 5, Nfilt))
    params['h_tplateau_s'] = abs(random.normal(15, .5, Nfilt))
    params['h_td'] = abs(random.normal(20, 1, Nfilt))
    params['h_td_s'] = abs(random.normal(10, .5, Nfilt))
    params['h_lalpha'] = abs(random.normal(.5, .05, Nfilt))
    params['h_lalpha_s'] = abs(random.normal(.4, .02, Nfilt))
    params['h_lbeta1'] = -abs(random.normal(2, .1, Nfilt))
    params['h_lbeta1_s'] = abs(random.normal(.7, .05, Nfilt))
    params['h_lbeta2'] = -abs(random.normal(1.8, .1, Nfilt))
    params['h_lbeta2_s'] = abs(random.normal(.5, .03, Nfilt))
    params['h_lbetadN'] = -abs(random.normal(1, .05, Nfilt))
    params['h_lbetadN_s'] = abs(random.normal(.3, .005, Nfilt))
    params['h_lbetadC'] = -abs(random.normal(2, .1, Nfilt))
    params['h_lbetadC_s'] = abs(random.normal(.2, .001, Nfilt))
    params['h_Mprdiff'] = random.normal(0, .05, Nfilt)
    params['h_Mprdiff_s'] = abs(random.normal(.5, .02, Nfilt))

    if fixhyper == 0:
        return params
    elif fixhyper == 1:
        hypers = {}
        for key in params.keys():
            if key[0:2] == 'h_':  #  and key[0:10]!='h_tplateau'
                hypers[key] = params.pop(key)
        return params, hypers
    elif fixhyper == 2:
        for key in params.keys():
            if key[0:2] == 'h_':  #  and key[0:10]!='h_tplateau'
                params.pop(key)
        return params


def fix_mzs(mzs):
    """
    Fix an array of mzero values to remove zeros.
    
    This happens when an object has non-detections, but no detections,
    in some bands
    
    If this fixing isn't done, you end up with inf flux values when you 
    try to normalize to a common zero point
    """
    if len(shape(mzs)) == 2:
        for k in range(len(mzs)):
            sel = where(mzs[k] == 0)
            mzs[k][sel] = mean(mzs[k][where(mzs[k] > 0)])
    elif len(shape(mzs)) == 1:
        sel = where(mzs == 0)
        mzs[sel] = mean(mzs[where(mzs > 0)])
    return mzs


def mat_SPII(t, fit, MJD=0):
    """
    Calculate the simple SN IIP magnitude for a series of N times t
    given a sample of M model draws for Mp, alpha, beta, and V
    
    If MJD=0, takes t to represent time since explosion
    If MJD=1, takes t to represent MJD
    
    returns a NxM matrix
    """
    if MJD:
        p_t = (tile(t,
                    (len(fit['alpha']), 1)).T - tile(fit['t0'], (len(t), 1)))
    else:
        p_t = (tile(t, (len(fit['alpha']), 1)).T)
    p_tp = (tile(fit['tp'], (len(t), 1)))
    p_td = (tile(fit['td'], (len(t), 1)))
    p_Mf = (tile(fit['Mf'], (len(t), 1)))
    p_alpha = (tile(fit['alpha'], (len(t), 1)))
    p_beta = (tile(fit['beta'], (len(t), 1)))
    p_betad = (tile(fit['betad'], (len(t), 1)))
    #p_eps=(normal(0,tile(fit['V'],(len(t),1))**.5))
    p_Yb = (tile(fit['Yb'], (len(t), 1)))

    m_rise = (p_Mf / (p_tp)**p_alpha * (p_t)**p_alpha) * ((p_t < p_tp) &
                                                          (p_t > 0))
    m_rise[isnan(m_rise)] = 0
    m_plateau = (p_Mf * exp(-p_beta * (p_t - p_tp))) * ((p_t > p_tp) &
                                                        (p_t < p_tp + p_td))
    m_late = (p_Mf * exp(-p_beta *
                         (p_td)) * exp(-p_betad *
                                       (p_t - p_tp - p_td))) * (p_t >
                                                                (p_tp + p_td))

    return (p_Yb + m_rise + m_plateau + m_late).T


def mat_SPII_m(t, fit, b, MJD=0, z=None, fluxscale=1e7):
    """
    Calculate the simple SN IIP flux for a series of N times t
    for a band b given a stan model fit
    
    If MJD=0, takes t to represent time since explosion
    If MJD=1, takes t to represent MJD
    
    returns a NxM matrix
    """
    j = 'grizy'.find(b)
    t = array(t)
    if MJD:
        p_t = (tile(t.astype('float'), (len(fit['t1'][:, j]), 1)).T -
               tile(fit['t0'], (len(t), 1)))
    else:
        p_t = (tile(t.astype('float'), (len(fit['t1'][:, j]), 1)).T)
    p_t1 = (tile(fit['t1'][:, j], (len(t), 1)))
    p_tp = (tile(fit['tp'][:, j], (len(t), 1)))
    p_t2 = (tile(fit['t2'][:, j], (len(t), 1)))
    p_td = (tile(fit['td'][:, j], (len(t), 1)))
    p_M1 = (tile(fit['M1'][:, j], (len(t), 1)))
    p_M2 = (tile(fit['M2'][:, j], (len(t), 1)))
    p_Mp = (tile(fit['Mp'][:, j], (len(t), 1)))
    p_Md = (tile(fit['Md'][:, j], (len(t), 1)))
    if 'alpha' in fit:
        p_alpha = (tile(fit['alpha'][:, j], (len(t), 1)))
        p_beta1 = (tile(fit['beta1'][:, j], (len(t), 1)))
        p_beta2 = (tile(fit['beta2'][:, j], (len(t), 1)))
        p_betadN = (tile(fit['betadN'][:, j], (len(t), 1)))
        p_betadC = (tile(fit['betadC'][:, j], (len(t), 1)))
    elif 'lalpha' in fit:
        p_alpha = exp(tile(fit['lalpha'][:, j], (len(t), 1)))
        p_beta1 = exp(tile(fit['lbeta1'][:, j], (len(t), 1)))
        p_beta2 = exp(tile(fit['lbeta2'][:, j], (len(t), 1)))
        p_betadN = exp(tile(fit['lbetadN'][:, j], (len(t), 1)))
        p_betadC = exp(tile(fit['lbetadC'][:, j], (len(t), 1)))
    p_Yb = (tile(fit['Yb'][:, j], (len(t), 1)))

    m_rise = p_M1 * (p_t / p_t1)**p_alpha * ((p_t < p_t1) & (p_t > 0))
    m_rise[isnan(m_rise)] = 0
    m_slowrise = (p_M1 * exp(p_beta1 * (p_t - p_t1))) * ((p_t > p_t1) &
                                                         (p_t < p_tp + p_t1))
    m_plateau = (p_Mp * exp(-p_beta2 * (p_t - p_t1 - p_tp))) * (
        (p_t > p_tp + p_t1) & (p_t < p_t2 + p_tp + p_t1))
    m_lateNi = (p_M2 * exp(-p_betadN * (p_t - p_t1 - p_tp - p_t2))) * (
        (p_t > p_t2 + p_tp + p_t1) & (p_t < p_td + p_t2 + p_tp + p_t1))
    m_lateCo = (p_Md * exp(-p_betadC * (p_t - p_t1 - p_tp - p_t2 - p_td))) * (
        p_t > p_t2 + p_tp + p_t1 + p_td)

    return fluxscale * (m_rise + m_slowrise + m_plateau + m_lateNi +
                        m_lateCo).T


def mat_SPII_m_mag(name, t, fit, b, MJD=0, z=0.):
    """
    Calculate the simple SN IIP magnitude for a series of N times t
    for a band b given a stan multilevel model fit
    
    If MJD=0, takes t to represent time since explosion
    If MJD=1, takes t to represent MJD
    If z is given, returns absolute magnitude
    
    returns a NxM matrix
    """
    # Derive zero point
    ## Now this is already applied during modeling
    #sel=where((aphot['event']==name) & (aphot['limit']==0) & (aphot['band']==b))
    #m0=(aphot[sel]['m']+2.5*log10(aphot[sel]['raw_flux']))[0]

    ## Get fluxes
    p_f = mat_SPII_m(t, fit, b, MJD=MJD)  #,z=z)

    ## Convert to mags
    p_f[p_f <= 0] = 1e-20
    p_m = -2.5 * log10(p_f)  #+m0
    p_m = np.ma.masked_array(p_m, np.isnan(p_m))

    ## Apply distance modulus
    ## Now this is already applied during modeling
    #if z>0: p_m+=distmod(z)

    return p_m


def mat_SPII_q(t, f, df, fit):
    """
    Calculate the outlier probability under the simple SN IIP gauddian mixture model
    for a series of N times t (MJD), magnitudes m, and mag. uncertainties dm given a sample of M model draws
    
    returns a MxN matrix
    """
    d_f = mat_SPII(t - median(fit['t0']), fit)
    pq_fg = scipy.stats.norm.pdf(
        f, d_f,
        sqrt(df**2 + matrix(fit['V']).T))  # probability under data model
    pq_bg = scipy.stats.norm.pdf(
        f,
        matrix(fit['Yb']).T, sqrt(
            (df**2 + matrix(fit['Vb']).T)))  # probability under outlier model
    q = median(pq_bg / (pq_bg + pq_fg), axis=0)
    return q


def SPII_dpeak_brightest(name):
    """
    Estimate the explosion date of a SN IIP based on the brightest detection in the band with the most detections
    """
    flens = {}
    for b in 'grizy':
        sel = where((aphot['limit'] == 0) & (aphot['event'] == name)
                    & (aphot['band'] == b))
        flens[b] = len(sel[0])
    fmost = flens.keys()[argmax(flens.values())]
    if fmost > 2:
        sel = where((aphot['limit'] == 0) & (aphot['event'] == name)
                    & (aphot['band'] == fmost))
        dpeak = aphot['MJD'][sel][argmin(aphot['m'][sel])]
    else:
        dpeak = None
    return dpeak


def SPII_dpeak_first(name):
    """
    Estimate the explosion date of an SN IIP based on the first detection within a cluster of detections
    """
    ## Manually specify some problematic ones
    if name == '2012-C-370407':
        dpeak = 56000
    elif name == '2012-C-370519':
        dpeak = 56000
    elif name == '2013-K-580098':
        dpeak = 56604
    else:
        ## Identify brightest point
        sel = where((aphot['event'] == name) & (aphot['limit'] == 0)
                    & ((aphot['band'] == 'g') | (aphot['band'] == 'r')
                       | (aphot['band'] == 'i') | (aphot['band'] == 'z')))
        m_max = min(aphot['m'][sel])
        ## Find clusters of detections
        sel = where((aphot['event'] == name) & (aphot['limit'] == 0)
                    & ((aphot['band'] == 'g') | (aphot['band'] == 'r')
                       | (aphot['band'] == 'i') | (aphot['band'] == 'z'))
                    & (aphot['m'] < m_max + 2))
        t = aphot['MJD'][sel]
        tgrid = arange(min(t), max(t))
        dgrid = zeros(len(tgrid))
        for i in range(len(tgrid)):
            dgrid[i] = sum((t > tgrid[i] - 50) & (t < tgrid[i] + 50))
        ## Identify date of biggest cluster
        bclust = tgrid[argmax(dgrid)]
        ## Identify first detection within the cluster
        dpeak = min(t[where((t > bclust - 200) & (t < bclust + 100))])
    return dpeak


def pystan_SPII_tm_getdseason(events, thresh=20, dp_func=SPII_dpeak_first):
    """
    Figure out which events in a list went off during an observing season, as opposed to behind the sun
    
    Parameters:
    * events: list of event names
    * thresh: Number of days before first detection to require another data point to qualify as the same observing season (default = 20)
    * dp_func: Function to use to identify the date of the first detection (SPII_dpeak_first by default)
    """
    dsea = []
    for i in range(len(events)):
        ## manually override some
        if events[i] == '2013-D-500012':
            dsea.append(0)
        elif events[i] == '2013-K-580098':
            dsea.append(0)
        else:
            ## identify earliest detection
            ed = dp_func(events[i])
            ## identify point before earliest detection
            peds = aphot['MJD'][(aphot['event'] == events[i])
                                & (aphot['MJD'] < ed)]
            if len(peds) > 0: ped = max(peds)
            else: ped = -inf
            if ed - ped < thresh:
                dsea.append(1)
            else:
                dsea.append(0)
    return array(dsea)


def SPII_plot_flux(name,
                   b,
                   dpeak,
                   fit,
                   MJD=0,
                   plotlines=0,
                   multi=0,
                   ML=0,
                   dt=1,
                   ax=None,
                   plot_median=1,
                   plot_contours=1,
                   mag=0,
                   c_med='k',
                   c_ML='r',
                   usedetlims=1,
                   Kcor=1):
    """
    Optional Parameters:
    
    * MJD: Show on MJD axis instead of days since explosion
    * plotlines: Show this many sample models from the fit (default = 0)
    * multi: if yes, treats fit as a multilevel model fit (i.e. use mat_SPII_m)
    * ML: Plot maximum likelihood model with dashed red line
    * ax: Use an existing axis to plot instead of creating a new one
    * plot_median: Show the median of the posterior model? (thick black line)
    * plot_contours: Show 1 and 2 sigma contours for posterior model
    * mag: Show magnitude scale instead of flux (default=0)
    * c_med: Color for posterior median line (default: black)
    * c_ML: Color for max likelihood line (default: red)
    * usedetlims: Use PS1 detection limits established from full dataset rather than 3sigma upper limits from individual non-detections
    * Kcor: Apply K-corrections to data
    """
    # Load data
    sel = where((aphot['event'] == name) & (aphot['band'] == b)
                & (aphot['MJD'] > dpeak - 50) & (aphot['MJD'] < dpeak + 300))
    z = aphot[sel]['z'][0]

    # Establish date grid
    if MJD:
        p_t = arange(median(fit['t0']) - 20, median(fit['t0']) + 500, dt)
    else:
        p_t = arange(-20, 500, dt)

    ## Calculate model posterior
    if multi:
        p_m = mat_SPII_m_mag(name, p_t, fit, b, MJD=MJD, z=z)
        if mag == 0:  ## convert back to flux
            magmod = distmod(z) + get_r_mzero(
                name, SPII_dpeak_first(name), band=b)
            p_m = 10**((p_m - magmod) / -2.5)
    else:
        if mag:
            raise Exception(
                "Magnitude plots for single-band light curve fits not implemented"
            )
        else:
            p_m = mat_SPII(p_t, fit, MJD=MJD)

    p_m = np.ma.masked_array(p_m, np.isnan(p_m))

    ## If MJD=1, I should time dilate the model to match the photometry
    if MJD:
        p_t += (
            p_t - median(fit['t0'])
        ) * z  # note this needs to be z, not 1+z, since it's a correction

    ## Plot contours
    if ax == None:
        fig = plt.figure()
        ax = plt.axes()

    if plot_contours:
        p_m_l2 = scipy.stats.mstats.scoreatpercentile(p_m, 5)
        p_m_l1 = scipy.stats.mstats.scoreatpercentile(p_m, 16)
        p_m_u2 = scipy.stats.mstats.scoreatpercentile(p_m, 95)
        p_m_u1 = scipy.stats.mstats.scoreatpercentile(p_m, 84)
        ax.fill_between(p_t, p_m_l2, p_m_u2, color='0.5', alpha=0.2)
        ax.fill_between(p_t, p_m_l1, p_m_u1, color='0.5', alpha=0.5)
    if plot_median:
        p_m_m = scipy.stats.mstats.scoreatpercentile(p_m, 50)
        ax.plot(p_t, p_m_m, c=c_med, lw=2)

    ## If plotlines, plot random sample of fits
    for i in range(plotlines):
        m = np.random.randint(len(fit['td']))
        ax.plot(p_t, p_m[m], alpha=0.2, c='r')

    ## Plot maximum likelihood model
    if ML:
        ax.plot(p_t,
                p_m[argmax(fit['lp__'])],
                alpha=0.5,
                c=c_ML,
                lw=2,
                ls='dashed')

    ## Plot data
    sel_d = where((aphot['band'] == b) & (aphot['event'] == name))
    t = aphot['MJD'][sel_d]

    ## Kcorrection - this has to be done in rest frame time since explosion
    if Kcor: Kc = vKcor_IIP(b, z, (t - median(fit['t0'])) / (1 + z))
    else: Kc = np.zeros(len(t))

    if MJD == 0:
        t = t - median(fit['t0'])

    if mag:
        ## Are any observations limits?
        sel_l = (aphot['limit'][sel_d] == 1)
        ##detections
        ax.errorbar(t[sel_l == 0],
                    aphot['m'][sel_d][sel_l == 0] + distmod(z) +
                    array(Kc)[sel_l == 0],
                    aphot['dm'][sel_d][sel_l == 0],
                    fmt='o',
                    mec='none',
                    color=bandc(b, col=1),
                    capsize=0)
        ##limits
        if usedetlims: lms = PS1_detlims[b] * ones(sum(sel_l))
        else: lms = aphot['m'][sel_d][sel_l]

        ax.plot(t[sel_l],
                lms + distmod(z),
                'v',
                mec='none',
                color=bandc(b, col=1),
                ms=2**2)
    else:
        Kc = 10**(Kc / -2.5)
        ax.errorbar(t,
                    aphot['raw_flux'][sel_d] * Kc,
                    aphot['raw_flux_err'][sel_d] * Kc,
                    fmt='o',
                    mec='none',
                    color=bandc(b, col=1),
                    capsize=0)

    ## Show background fit
    if mag == 0:
        pYb = percentile(fit['Yb'] * 1e7 / 10**(magmod / -2.5), [16, 84])
        ax.fill_between([min(p_t) - 100, max(p_t) + 100], [pYb[0], pYb[0]],
                        [pYb[1], pYb[1]],
                        color=bandc(b, col=1),
                        alpha=0.2)

    ## Format the plot
    if MJD: ax.set_xlabel('MJD')
    else: ax.set_xlabel('Time since t0 (rest frame days)')

    if mag: ax.set_ylabel('AB mag (' + b + ' band)')
    else: ax.set_ylabel('PS1 flux (' + b + ' band)')

    ax.set_title(pso_shortname_dict[name])
    if MJD:
        if mag:
            ax.axis(
                [median(fit['t0']) - 30,
                 median(fit['t0']) + 300, -14, -20])
        else:
            ax.axis([
                median(fit['t0']) - 30,
                median(fit['t0']) + 300, -5000,
                max(p_m_u2) * 1.2
            ])
    else:
        if mag:
            ax.axis([-30, 300, -14, -20])
        else:
            ax.axis([-30, 300, -5000, max(p_m_u2) * 1.2])


def SPII_plot_tm(name, b, sout, subset='all', **kwargs):
    """
    Plots data and model for SN in one band using results from a "SIIP_t_multilevel.stan" model fit
    
    Parameters:
    * name: The name of the SN to be plotted
    * b: The band to be plotted
    * sout: List returned by pystan_SPII_tm from fit (contains [events,fit])
    * subset: Plot results for just the stated range of samples (default: 'all', to do one chain do e.g. [0:100])
    
    note: dpeak is obtained using SNquickload, so global variables will be re-assigned!
    
    Remaining arguments are passed along to SPII_plot_flux
    """
    SNquickload(name)
    s_fit = pystan_SPII_tm_getfit(sout, name)
    if subset != 'all':
        for key in s_fit:
            s_fit[key] = s_fit[key][subset[0]:subset[1]]
    SPII_plot_flux(name, b, dpeak, s_fit, **kwargs)


def SPII_plot_mag(name, b, dpeak, fit, plotlines=0, multi=0, ML=0, dt=1):
    """
    Parameters:
    
    * multi: if yes, treats fit as a multilevel model fit (i.e. use mat_SPII_m)
    * ML: Plot maximum likelihood model with dashed red line
    """
    # Load data
    sel = where((aphot['event'] == name) & (aphot['band'] == b)
                & (aphot['MJD'] > dpeak - 50) & (aphot['MJD'] < dpeak + 500))
    z = aphot[sel]['z'][0]

    # Establish date grid
    p_t = arange(median(fit['t0']) - 20, median(fit['t0']) + 500, dt)

    ## Calculate model posterior without K corrections
    if multi:
        p_m = mat_SPII_m_mag(name, p_t, fit, b, MJD=1, z=z, Kcor=0)
    else:
        # Derive zero point
        sel = where((aphot['event'] == name) & (aphot['limit'] == 0)
                    & (aphot['band'] == b))
        m0 = (aphot[sel]['m'] + 2.5 * log10(aphot[sel]['raw_flux']))[0]
        p_f = mat_SPII(p_t, fit, MJD=1)
        ## Convert to absolute mags
        p_f[p_f <= 0] = 1e-20
        p_m = -2.5 * log10(p_f) + m0 + distmod(z)
        p_m = np.ma.masked_array(p_m, np.isnan(p_m))

    p_m_m = scipy.stats.mstats.scoreatpercentile(p_m, 50)
    p_m_l2 = scipy.stats.mstats.scoreatpercentile(p_m, 5)
    p_m_l1 = scipy.stats.mstats.scoreatpercentile(p_m, 16)
    p_m_u2 = scipy.stats.mstats.scoreatpercentile(p_m, 95)
    p_m_u1 = scipy.stats.mstats.scoreatpercentile(p_m, 84)

    ## Plot contours
    plt.figure()
    plt.fill_between(p_t, p_m_l2, p_m_u2, color='0.5', alpha=0.2)
    plt.fill_between(p_t, p_m_l1, p_m_u1, color='0.5', alpha=0.5)
    plt.plot(p_t, p_m_m, c='k', lw=2)

    ## If plotlines, plot random sample of fits
    for i in range(plotlines):
        m = np.random.randint(len(fit['td']))
        plt.plot(p_t, p_m[m], alpha=0.2, c='r')

    ## Plot maximum likelihood model
    if ML:
        plt.plot(p_t,
                 p_m[argmax(fit['lp__'])],
                 alpha=0.5,
                 c='r',
                 lw=2,
                 ls='dashed')  ## Plot data

    sel_d = where((aphot['band'] == b) & (aphot['event'] == name)
                  & (aphot['limit'] == 0))
    t = aphot['MJD'][sel_d]
    plt.errorbar(t,
                 aphot['m'][sel_d] + distmod(z),
                 aphot['dm'][sel_d],
                 fmt='o',
                 mec='none',
                 color=bandc(b, col=1),
                 capsize=0)
    ## Show non-detections
    sel = where((aphot['band'] == b) & (aphot['event'] == name)
                & (aphot['limit'] == 1))
    for i in sel[0]:
        plt.axvline(aphot['MJD'][i], ls='dashed', color=bandc(b, col=1))

    ## Format the plot
    plt.xlabel('MJD')
    plt.ylabel('Absolute magnitude (' + b + ' band)')
    plt.title(name)
    plt.axis([median(fit['t0']) - 30, median(fit['t0']) + 300, -12, -22])


def intbands(J):
    """
    Helper function to convert a string array of grizy bands to integers
    g' = 1
    r' = 2
    i' = 3
    z' = 4
    y' = 5
    """
    J2 = np.zeros(len(J), dtype='int')
    J2[J == 'g'] = 1
    J2[J == 'r'] = 2
    J2[J == 'i'] = 3
    J2[J == 'z'] = 4
    J2[J == 'y'] = 5
    return J2


def get_r_mzero(name, dpeak, band='r'):
    """
    Helper function to get r-band zero point magnitude for a supernova (name)
    """
    sel = where((aphot['event'] == name) & (aphot['MJD'] > dpeak - 50)
                & (aphot['MJD'] < dpeak + 300) & (aphot['band'] == band)
                & (aphot['limit'] == 0))
    dms = sort(aphot['m'][sel] + 2.5 * log10(aphot['raw_flux'][sel]))
    #if dms[-1] - dms[0] > 0.05: pdb.set_trace()
    return dms[0]


def SNquickload(lname, getfit=0):
    global name, dpeak, t, m, dm, f, df, z, J, lim

    name = lname
    dpeak = SPII_dpeak_first(name)
    ## Load data
    sel = where((aphot['event'] == name) & (aphot['MJD'] > dpeak - 50)
                & (aphot['MJD'] < dpeak + 300)
                & ((aphot['band'] == 'g') | (aphot['band'] == 'r')
                   | (aphot['band'] == 'i') | (aphot['band'] == 'z')
                   | (aphot['band'] == 'y')))
    t = aphot[sel]['MJD']
    m = aphot[sel]['m']
    dm = aphot[sel]['dm']
    f = aphot[sel]['raw_flux']
    df = aphot[sel]['raw_flux_err']
    lim = aphot[sel]['limit']
    z = aphot[sel]['z'][0]
    J = intbands(aphot[sel]['band'])

    if getfit:
        global fit
        fit = np.load(rootdir + 'mod5_lin_ind/fit4_' + name +
                      '.npy')[1]  #.item(0)


def get_fit_SN(sfit, snames, name):
    """
    Extract fit parameters for a particular SN (name) from within a heirarchical model fit (sfit) for many SNe (snames)
    """
    new_fit = {}
    n = where(array(snames) == name)[0]
    if len(n) == 0:
        raise Exception("get_fit_SN: Name '" + name + "' not in snames")
    for key in sfit:
        s = shape(sfit[key])
        # Is this a SN-specific parameter?
        if len(s) > 1 and list(s)[1] == len(snames):
            # Is this a per-band parameter?
            if len(s) == 3:
                new_fit[key] = sfit[key][:, n].reshape(list(s)[0], -1)
            else:
                new_fit[key] = sfit[key][:, n].reshape(list(s)[0])
    return new_fit


def get_indfit_SN(snames,
                  file_name=[rootdir + 'mod5_lin_ind/fit_resubmit_', '.npy']):
    """
    Function to load a suite of individual SN fits and combine them into one sfit object.
    
    Parameters:
    * snames: List of SN names to load
    * file_name: list of [prefix,suffix] to append to file names for loading
    
    Note: expects every parameter to have identical sampling lengths
    """
    for i in range(len(snames)):
        name = snames[i]
        fit = np.load(file_name[0] + name + file_name[1])[1]  #.item(0)
        if i == 0:
            sfit = fit
            for p in sfit.keys():
                sfit[p] = expand_dims(sfit[p], 1)
        else:
            for p in sorted(sfit.keys()):
                #if the axis lengths are not the same, and this is not unidimensional, this isn't something that should be ported (e.g. mm and fL)...
                b = expand_dims(fit[p], 1)
                if (len(shape(b)) == 2) or (shape(b)[-1] == shape(
                        sfit[p])[-1]):
                    sfit[p] = np.concatenate([sfit[p], b], axis=1)
                else:
                    print 'get_indfit_SN: Excluding key ' + p
                    sfit.pop(p)
    return sfit


def get_marg_post(param, sV, b=None, bands='grizy', oktypes=None, multi=1):
    """
    Get the full marginalized posterior information for a given LC parameter
    
    Parameters:
    
    * param: stan fit parameter
    * sV: version of stan output to use (e.g. '5')
    * b: Band to use for K-dimensional parameters (None or string)
    * bands: Full set of bands to refer to (default: 'grizy')
    * oktypes: Use only SNe whose ttype is in this list (e.g. ['SNIIp'])
    * multi: Use hierarchical model fit instead of individual fits?  
    """
    allp = []
    if multi:
        snames, sfit = np.load('stan/sfit_tm_' + sV + '.npy')
    for name in snames:
        if oktypes == None or ttype[name] in oktypes:
            SNquickload(name)
            if multi:
                fit = get_fit_SN(sfit, snames, name)
            else:
                fit = np.load('stan/sfit_' + name + '_' + sV + '.npy').item(0)
            if b == None:
                p = fit[param]
            else:
                p = fit[param][:, bands.find(b)]
            allp.append(p)
    return array(allp).T


def getCmdStanSamples(gstring, warmup=0, nodiv=1):
    """
    Load CmdStan sampler output from a set of chain files
    
    Parameters
    * gstring: glob string to identify chain files
    * warmup: range of samples to exclude from output dictionary (burnin length)
    * nodiv: If yes, remove divergent samples before output
    """
    files = glob.glob(gstring)
    Nch = len(files)
    outarray = None
    prev_end_point = 0
    for fnum in range(len(files)):
        fs = sorted(files)[fnum]
        ## Remove header
        nohead_file = fs.rsplit('/', 1)[0] + '/' + 'nohead_' + fs.rsplit(
            '/', 1)[1]
        pgrepv(fs, outfile=nohead_file)
        ## Load file
        fdata = myloadcsv(nohead_file)
        N = len(fdata['lp__'])
        if N <= warmup:
            N = 0
        elif N > warmup:
            N = N - warmup
        ## Initialize the output dictionary once
        if outarray == None:
            ### Get header line
            header = fdata.keys()
            ## Get list of parameters
            params = unique([c.split('.')[0] for c in header])
            ## Find dimensions for each parameter:
            pdim = {}
            for p in params:
                ## Get a parameter name of this type
                subps = [c for c in header if p + '.' in c]
                ## Is this a solitary parameter
                if len(subps) == 0:
                    pdim[p] = [N]
                else:
                    ## Count the dimensions
                    Ndim = len(subps[0].split('.'))
                    if Ndim == 1:
                        pdim[p] = [N]  # redundant with above, I think
                    elif Ndim == 2:
                        subsubps1 = [int(c.split('.')[1]) for c in subps]
                        pdim[p] = [N, max(subsubps1)]
                    elif Ndim == 3:
                        subsubps1 = [int(c.split('.')[1]) for c in subps]
                        subsubps2 = [int(c.split('.')[2]) for c in subps]
                        pdim[p] = [N, max(subsubps1), max(subsubps2)]
            ## Initialize the dictionary
            outarray = {}
            for p in params:
                outarray[p] = np.zeros(pdim[p])
            outarray['chain#'] = np.zeros(N)
        else:
            if N > 0:
                ## expand the arrays
                for p in outarray.keys():
                    nshape = list(shape(outarray[p]))
                    nshape[0] = N
                    outarray[p] = np.concatenate(
                        [outarray[p], np.zeros(nshape)], axis=0)
        #### Read in data
        if N > 0:
            crange = [prev_end_point, prev_end_point + N]
            for p in params:
                ## check dimensions of parameter
                if len(pdim[p]) == 1:
                    outarray[p][crange[0]:crange[1]] = fdata[p][-N:]
                elif len(pdim[p]) == 2:
                    for j in range(pdim[p][1]):
                        outarray[p][crange[0]:crange[1],
                                    j] = fdata[p + '.' + str(j + 1)][-N:]
                elif len(pdim[p]) == 3:
                    for j in range(pdim[p][1]):
                        for k in range(pdim[p][2]):
                            outarray[p][crange[0]:crange[1], j,
                                        k] = fdata[p + '.' + str(j + 1) + '.' +
                                                   str(k + 1)][-N:]
            outarray['chain#'][crange[0]:crange[1]] = fnum
            prev_end_point += N
    ##remove divergent samples
    if nodiv:
        sel = where(outarray['n_divergent__'] == 0)
        for p in outarray.keys():
            outarray[p] = outarray[p][sel]
    return outarray


#############################################################
######## Stan light curve statistics
#############################################################


def SNstat_t0(fit, z):
    """
    Calculate the explosion date of a SN 
    (Note: time parameters are already calculated in rest frame in the stan model)
    """
    return fit['t0']


def SNstat_tp(fit, z, band=1):
    """
    Calculate the plateau duration of a SN (tp+t2) using r-band (band=1) by default, in rest frame
    (Note: time parameters are already calculated in rest frame in the stan model)
    """
    ##calculate duration
    pm = fit['tp'][:, band] + fit['t2'][:, band]
    return pm


def SNstat_dm15(fit, z, name, band='r', dt=0.1, tmax=150):
    """
    Calculate the decline rate (delta m_15)
    
    Uses r' band by default
    dt is time spacing (in days) for model grid
    (Note: time parameters are already calculated in rest frame in the stan model)
    """
    if band in 'grizy':
        ## Calculate purely based on the plateau phase decline rate (beta2)
        #b2=fit['beta2'][:,'grizy'.find(band)]
        #dm15 = 2.5*log10(exp(-b2*15.))
        ## Calculate based on light curve models
        pt = arange(1, tmax, dt)
        pm = mat_SPII_m_mag(name, pt, fit, band, MJD=0, z=z)
        sel = where(
            pt < tmax - 15 * dt)  #don't search past the end of the grid
        pmax = argmax(-pm[:, sel[0]], axis=1)

        dm15 = pm[range(len(pm)), pmax] - pm[range(len(pm)),
                                             pmax + int(15 / dt)]

        return dm15
    else:
        raise Exception('SNstat_dm15: band not recognized')


def SNstat_trise(fit, z, band='r'):
    """
    Calculate the rise time as a sum of t1 and tp in the specified band
    (Note: time parameters are already calculated in rest frame in the stan model)
    """
    t1 = fit['t1'][:, 'grizy'.find(band)]
    tp = fit['tp'][:, 'grizy'.find(band)]
    return (t1 + tp)


def SNstat_Lbolo(fit, tt, z, name, bands='griz', AV_cor=1):
    """
    Calculate the bolometric luminosity from g-z band for a SN by SED interpolation
    
    Returns energy in ergs
    
    Parameters:
    * Fit: Stan samples
    * tt: Time samples to return luminosity values for
    * z: redshift
    * bands: Photometric filters (default = 'grizy')
    * AV_cor: If true, apply extinction correction using Kasen's redenning formula
    """
    ##Get E(B-V)
    N = len(fit[fit.keys()[0]])
    if AV_cor:
        EBV = Kasen_EBV(name, fit, z)
    else:
        EBV = np.zeros(N)
    ##Calculate model lightcurve in each band:
    allLC = {}
    offset = zeros([len(bands), N])
    for j in range(len(bands)):
        ## Call model
        allLC[bands[j]] = mat_SPII_m_mag(name, tt, fit, bands[j],
                                         MJD=0) - distmod(z)
        ## Calculate extinction
        if AV_cor:
            A_l = CCMextinct([allbands[bands[j]][0]], EBV, Rv=3.1, mag=1)
        else:
            A_l = np.zeros(len(fit[fit.keys()[0]]))
        offset[j] = +5 * log10(allbands[bands[j]][0]) + 2.406 - A_l
    ##Step through MCMC samples
    flc = zeros([N, len(tt)])
    ##Step through bands
    for j in range(len(bands)):
        flc += 10**((allLC[bands[j]] + np.meshgrid(offset[j], tt)[0].T) /
                    -2.5) * allbands[bands[j]][1]
    flc = flc * 4 * pi * (cosmocalc(z)['DL_cm'])**2
    return flc


def SNstat_L50(fit, z, name, bands='griz', AV_cor=1):
    """
    Calculate the pseudo-bolometric luminosity at 50 days
    """
    L50 = SNstat_Lbolo(fit, [50], z, name, bands=bands, AV_cor=AV_cor)
    return L50[:, 0]


def SNstat_Ebolo(fit, z, name, bands='griz', dt=0.5, dmax=500):
    """
    Calculate the bolometric luminosity from g-z band for a SN by SED interpolation
    
    Returns energy in ergs
    
    Parameters:
    * Fit: Stan samples
    * tt: Time samples to return luminosity values for
    * z: redshift
    * bands: Photometric filters (default = 'griz')
    
    (Note: time parameters are already calculated in rest frame in the stan model)
    """
    L = SNstat_Lbolo(fit, arange(0.1, dmax, dt), z, name, bands=bands)
    return sum(L, axis=1)


SNstatpack = {
    't0': SNstat_t0,
    'tp': SNstat_tp,
    'dm15': SNstat_dm15,
    'trise': SNstat_trise,
    'SNstat_L50': SNstat_L50,
    'E_bolo': SNstat_Ebolo
}

#############################################################
######## Progenitor property estimation using formulae from
######## Kasen & Woosley 2009 (Astrophysical Journal 703 2205)
#############################################################

### Record array with Kasen et al. Table 1 data
thed = [('Mi', 'float'), ('Z', 'float'), ('Mf', 'float'), ('R0', 'float'),
        ('MFe', 'float'), ('XHe', 'float'), ('~Mej', 'float')]
Kasen_T1_data=[\
#Mi	Z	Mf	R0	MFe	XHe

[12 ,1.0 ,10.9  ,625 ,1.365 ,0.30],\
[15 ,1.0 ,12.8  ,812 ,1.482 ,0.33],\
#[15	,0.1	,13.3	 ,632	,1.462	,0.33],\
[20 ,1.0 ,15.9 ,1044 ,1.540 ,0.38],\
[25 ,1.0 ,15.8 ,1349 ,1.590 ,0.45]]

Kasen_T1 = array([], dtype=thed)
for i in range(len(Kasen_T1_data)):
    Kasen_T1.resize(Kasen_T1.shape[0] + 1)
    for j in range(len(thed) - 1):
        Kasen_T1[-1][thed[j][0]] = Kasen_T1_data[i][j]
    ## Estimate ejecta mass as Mf - MFe
    Kasen_T1[-1]['~Mej'] = Kasen_T1_data[i][2] - Kasen_T1_data[i][4]


def Kasen_T1_interp(key1, key2):
    """
    Use Kasen & Woosley (2009) Table 1 to interpolate between their model grid and return
    stellar progenitor parameters as a function of another parameter
    
    e.g. Kasen_T1_interp('Mi','R0')(14) = 749.7 R_solar
    """
    #return interp1d(Kasen_T1[key1],Kasen_T1[key2],bounds_error=0,fill_value=nan)
    return lambda x: polyval(polyfit(Kasen_T1[key1], Kasen_T1[key2], 2), x)


Pastorello_F13 = np.genfromtxt(rootdir +
                               'literature/Pastorello04/Figure13.txt')
Pastorello_F13I = interp1d(Pastorello_F13[:, 0], Pastorello_F13[:, 1])


def Pastorello_MNi(LSN, t):
    """
    Using Figure 13 of Pastorello et al. (2004MNRAS.347...74P), and the equation on their p.89,
    estimate M_Ni for a SN given its pseudo-bolometric luminosity relative to SN 1987a
    
    Parameters:
    * LSN: OIR pseudo-bolometric luminosity of the SN at a time t
    * t: Time in rest frame days since explosion
    """
    return 0.075 * LSN / 10**Pastorello_F13I(t)


def Pastorello_calc_MNi(fit, z, name, t_min=125, t_max=None):
    """
    Using Figure 13 of Pastorello et al. (2004MNRAS.347...74P), and the equation on their p.89,
    estimate M_Ni for a SN given its stan fit.
    
    This works by identifying the time at which the SN luminosity is best constrained using a grid search
    
    Parameters:
    * fit: Stan fit pickle
    * z: Redshift
    * name: SN name for looking up values in aphot
    * LSN: OIR pseudo-bolometric luminosity of the SN at a time t
    * t_min,t_max: Minimum and maximum time for grid search(days since explosion)
	If t_max==None, the time of the last detection is used (or t=300; whichever is earlier)
    """
    ## Get t_max
    if t_max == None:
        t_all = aphot['MJD'][where((aphot['limit'] == 0)
                                   & (aphot['event'] == name))]
        t_max = max(t_all)
        if t_max > 300: t_max = 300
    ## Get bolometric luminosities
    tt = arange(t_min, t_max, 2)
    Ls = SNstat_Lbolo(fit, tt, z, name)
    L_down, L_up = percentile(Ls, [5, 95], axis=0)
    ## Identify best constrained time
    best = argmin(L_up - L_down)
    t = tt[best]
    LSN = Ls[:, best]
    # Return M_Ni
    return Pastorello_MNi(LSN, t)


def Kasen_Eq11(L50, t_p, M_Ni, verbose=0, expand_extrap=0):
    """
    Given L50, t_p, and M_Ni, return M_in, R_0, E_51, M_ej
    Using Eq. 12 & 13 from Kasen & Woosley (2009)
    
    Parameters:
    * L50: log10 of luminosity at 50 days past explosion
    * t_p: Plateau duration (rest frame days)
    * M_Ni: Nickel mass (solar masses)
    * verbose: verbosity for cobyla output (default=0; no printing)
    * expand_extrap: Number of solar masses to artifically expand the Kasen model grid for extrapolation
	(if this is 0, uses 12 - 25 Msol)
    
    Returns:
    * Min: Initial mass in units of 10 solar masses
    * R0: Radius in solar radii
    * Mej: Ejecta mass in units of solar masses (this is a poor estimate - M_final - M_Fe_core)
    * E_51: Log of energy in units of 10^51 ergs 
    * tp0: True plateau duration (in rest frame days corrected for M_Ni contamination)
    """

    ## Define system of equations
    def depends(Min):
        R0 = Kasen_T1_interp('Mi', 'R0')(10 * Min)
        Mej = Kasen_T1_interp('Mi', '~Mej')(10 * Min)
        E51 = (10**L50 / (1.49e42 * Min**0.77))**(1 / 0.82)
        tp0 = 128 * E51**-0.26 * Min**0.11
        return R0, Mej, E51, tp0

    def Eqs(p):
        Min = p[0]
        R0, Mej, E51, tp0 = depends(Min)
        Eq3 = t_p - tp0 * (
            1 + 22. * M_Ni * E51**-0.5 * (R0 / 500.)**-1 * Mej**0.5
        )**(
            1 / 6.
        )  # note: .35 -> 22 to fix Kasen's Eq. 13, see notes from 1/11/2014
        return Eq3**2

    ## Define constraints
    C1 = lambda p: p[0] - (min(Kasen_T1['Mi']) / 10. - expand_extrap
                           )  # Min > 12
    C2 = lambda p: (max(Kasen_T1['Mi']) / 10. + expand_extrap) - p[0
                                                                   ]  # Min <25
    ## Constrained multivariate fit
    p = scipy.optimize.fmin_cobyla(Eqs, [2.], [C1, C2], iprint=verbose)
    R0, Mej, E51, tp0 = depends(p[0])
    return p[0], R0, Mej, E51, tp0


def Kasen_calc(name,
               fit,
               N=100,
               allback=1,
               optreturns=1,
               bolo_correct=1.,
               AV_cor=True,
               expand_extrap=0):
    """
    Convenience function to automate calculating SN progenitor parameters using the formulae from Kasen & Woosley (2009)
    
    Parameters:
    * name: For loading SN data from aphot
    * fit: stan fit pickle
    * N: Number of samples to pick from the stan chain (default=100)
    * optreturns: Return subsamples from L50,t_p,M_Ni in addition to physical parameters
    * bolo_correct: Multiplicative correction factor to apply to L50 to convert from pseudo-bolometric to full bolometric luminosity (default=1)
    * AV_cor: If true, apply extinction correction using Kasen's redenning formula
    * expand_extrap: Number of solar masses to artifically expand the Kasen model grid for extrapolation
	(if this is 0, uses 12 - 25 Msol)
    
    Returns: 
    * Min: Initial mass in solar masses
    * R0: Radius in solar radii
    * Mej: Ejecta mass in units of solar masses (this is a poor estimate - M_final - M_Fe_core)
    * E_51: Log of energy in units of 10^51 ergs 
    * tp0: True plateau duration (in rest frame days corrected for M_Ni contamination)
    
    Optional returns:
    * L50
    * t_p
    * M_Ni
    """
    SNquickload(name)
    L50 = SNstat_L50(fit, z, name, AV_cor=AV_cor) * bolo_correct
    t_p = SNstat_tp(fit, z)
    M_Ni = Pastorello_calc_MNi(fit, z, name)

    Min = np.zeros(N)
    R0 = np.zeros(N)
    Mej = np.zeros(N)
    E51 = np.zeros(N)
    tp0 = np.zeros(N)
    js = random.randint(0, len(L50), N)
    for i in range(N):
        j = js[i]
        Min[i], R0[i], Mej[i], E51[i], tp0[i] = Kasen_Eq11(
            log10(L50[j]), t_p[j], M_Ni[j], expand_extrap=expand_extrap)

    if optreturns:
        return 10 * Min, R0, Mej, E51, tp0, [L50[js], t_p[js], M_Ni[js]]
    else:
        return 10 * Min, R0, Mej, E51, tp0


Kasen_Eq15 = lambda V: 0.52 + 0.03 * (V + 17.5)


def Kasen_EBV(name, fit, z):
    """
    Estimate E(B-V) redenning for a SN IIp by comparison to 
    the color predicted by Eq. 15 of Kasen & Woosley (2009)
    """
    ### Get absolute photometry
    try:
        mg = mat_SPII_m_mag(name, [50], fit, 'g', MJD=0, z=z)
        mr = mat_SPII_m_mag(name, [50], fit, 'r', MJD=0, z=z)
        mi = mat_SPII_m_mag(name, [50], fit, 'i', MJD=0, z=z)
        mz = mat_SPII_m_mag(name, [50], fit, 'z', MJD=0, z=z)
    except:
        print "Kasen_EBV: Photometry error for " + name
        null = ones([len(fit[fit.keys()[0]]), 1]) * nan
        mg, mr, mi, mz = null, null, null, null

    ## Convert to Landolt/Vega
    U, B, V, R, I = toUBVRI_99em(nan, mg, mr, mi, mz,
                                 AB=0)  # Assuming Kasen used Vega mags
    Vs = V[:, 0]
    Is = I[:, 0]

    ## Calculate Kasen's predicted values
    Kasen_VI = Kasen_Eq15(Vs)

    ## The difference is the redenning
    EVI = (Vs - Is) - Kasen_VI

    ## Convert to E(B-V)
    EBV = EVI / 1.38  # via Tammann et al. (2003A&A...404..423T)

    ## Prevent negative numbers
    EBV[EBV < 0] = 0

    return EBV


#############################################################
######## Load photometry database
#############################################################

aphot = np.load(photdir + 'allphot_galex_5.npy')

## fix a redshift
aphot['z'][aphot['event'] == '2011-A-120333'] = 0.15

allevents = unique(aphot['event'])

if '' in allevents:
    print "WARNING\nPhotometry (probably Galex) exists for an unnamed object and will be ignored."
    allevents = np.delete(allevents, where(allevents == ''), axis=0)

#### Adjust fluxes to common zero point, mzero=30
## Get filter list
J = intbands(aphot['band'])
## Get SN ID list
K = np.zeros(len(aphot), dtype=int)
Kdic = {}
for i in range(1, 1 + len(allevents)):
    Kdic[allevents[i - 1]] = i
for i in range(len(aphot)):
    if aphot['event'][i] != '': K[i] = Kdic[aphot['event'][i]]
## Get zero point lists
old_mzs = np.zeros([len(allevents), 5])
for k in range(len(allevents)):
    dp = SPII_dpeak_first(allevents[k])
    for j in range(5):
        try:
            old_mzs[k, j] = get_r_mzero(allevents[k], dp, band='grizy'[j])
        except:
            old_mzs[k, j] = 0
## Apply correction
fcorrect = 10**((old_mzs[K - 1, J - 1] - 30) / -2.5)
fcorrect[
    abs(fcorrect) >
    10] = 0  # some objects are messed up - they have all non-detections and therefore no real zero point definition, so don't touch them

aphot['raw_flux'] *= fcorrect
aphot['raw_flux_err'] *= fcorrect

## Determine detection limits
PS1_bands = unique(aphot['band'])
PS1_detlims = {}
for b in PS1_bands:
    m = percentile(
        aphot['m'][where((aphot['band'] == b) & (aphot['limit'] == 0))], 95)
    PS1_detlims[b] = m

## For light curve statistics
LCpars = np.load('PS1_lcfitpars_u3.npy').item(0)
ttype = np.load('PS1_ttype6.npy').item(0)
ttype_u = np.load('PS1_ttype_u2.npy').item(0)
pso_name_dict = np.load('PS1_names.npy').item(0)
pso_coords_dict = np.load('PS1_coords.npy').item(0)
pso_shortname_dict = np.load('PS1_shortnames.npy').item(0)


################################
########### For running the model
################################

def runstan(model,
            nobjs='all',
            samples=20e3,
            chains=5,
            parallel=5,
            delta=.25,
            thin=10,
            sinit='myprior',
            fixstep=0,
            get_neff=1,
            max_tree=15,
            startstep=None,
            indmodel=0,
            tdict=ttype,
            **kwargs):
    ## how many objects to include?
    SN_IIps = [obj for obj in allevents if 'iip' in tdict[obj].lower()]
    if nobjs == 'all':
        goto = len(SN_IIps)
    else:
        goto = nobjs

    cond = []
    ts = arange(0, 200)
    Kcors = np.zeros([goto, 5, len(ts)])
    for i in range(goto):
        name = SN_IIps[i]
        cond.append((aphot['event'] == name))
        if name == '2011-A-120333': z = 0.15
        else: z = aphot['z'][cond[-1]][0]
        if isnan(z) == 0:
            for j in range(5):
                vKcor = vectorize(lambda x: Kcor_IIP('grizy'[j], z, x))
                Kcors[i, j] = vKcor(ts)

    sel = where(sum(array(cond), axis=0) == 1)
    subdata = aphot.copy()[sel]
    subdata['z'][subdata['event'] == '2011-A-120333'] = 0.15

    if fixstep == 0:
        if startstep == None:
            mcontrol = {'adapt_delta': delta, 'max_treedepth': max_tree}
        else:
            mcontrol = {
                'adapt_delta': delta,
                'max_treedepth': max_tree,
                'stepsize': startstep
            }
        mwarmup = samples / 2
    else:
        mcontrol = {
            'adapt_engaged': False,
            'stepsize': fixstep,
            'max_treedepth': max_tree
        }
        mwarmup = 0

    a = pystan_SPII_tm(subdata,
                       stan_init=sinit,
                       stanmodel=(rootdir + 'stan/' + model),
                       chains=chains,
                       iterations=samples,
                       Kcor=Kcors,
                       get_neff=get_neff,
                       control=mcontrol,
                       thin=thin,
                       parallel=parallel,
                       fixhyper=0,
                       warmup=mwarmup,
                       **kwargs)

    try:
        ## Show some diagnostic plots
        global snames, sfit, sneff
        snames, sfit, null = a

        plotrandomLCs(5, a=a, mag=0)

        if get_neff == 1:
            sne = a[2][-1]

        elif get_neff == 'some':
            sne = a[2]

        if get_neff != 0:
            plt.figure()
            plt.plot(
                array([sne[c]['n_divergent__'] for c in range(len(sne))]).T)
            plt.ylabel('ndivergent')
            plt.axis([-1, len(sne[0]['n_divergent__']), -0.1, 1.1])

            plt.figure()
            plt.semilogy(
                array([sne[c]['stepsize__'] for c in range(len(sne))]).T)
            plt.ylabel('stepsize')

            plt.figure()
            plt.plot(array([sne[c]['treedepth__'] for c in range(len(sne))]).T)
            plt.ylabel('treedepth')
    except:
        return a

    return a


def runstan_ind(model,
                name,
                samples=20e3,
                chains=5,
                parallel=5,
                delta=.25,
                thin=10,
                sinit='myprior',
                fixstep=0,
                get_neff=1,
                max_tree=15,
                startstep=None,
                indmodel=0,
                saveit=0,
                **kwargs):
    cond = []
    ts = arange(0, 200)
    Kcors = np.zeros([5, len(ts)])
    cond.append((aphot['event'] == name))
    z = aphot['z'][cond[-1]][0]
    if isnan(z) == 0:
        for j in range(5):
            vKcor = vectorize(lambda x: Kcor_IIP('grizy'[j], z, x))
            Kcors[j] = vKcor(ts)

    sel = where(sum(array(cond), axis=0) == 1)
    subdata = aphot.copy()[sel]

    ## can't run the fit if there's no redshift
    if sum(isnan(subdata['z'])) > 0:
        print "runstan_ind: FAILURE FOR " + name + ": No redshift"
        return None

    if fixstep == 0:
        if startstep == None:
            mcontrol = {'adapt_delta': delta, 'max_treedepth': max_tree}
        else:
            mcontrol = {
                'adapt_delta': delta,
                'max_treedepth': max_tree,
                'stepsize': startstep
            }
        mwarmup = samples / 2
    else:
        mcontrol = {
            'adapt_engaged': False,
            'stepsize': fixstep,
            'max_treedepth': max_tree
        }
        mwarmup = 0

    a = pystan_SPII_tm_ind(subdata,
                           stan_init=sinit,
                           stanmodel=(rootdir + 'stan/' + model),
                           chains=chains,
                           iterations=samples,
                           Kcor=Kcors,
                           get_neff=get_neff,
                           control=mcontrol,
                           thin=thin,
                           parallel=parallel,
                           warmup=mwarmup,
                           **kwargs)
    if saveit != 0:
        np.save(rootdir + saveit + name, a)

    ## Show some diagnostic plots
    global snames, sfit, sneff
    snames, sfit, null = a
    SNquickload(name)  # to get dpeak

    for j in range(5):
        try:
            SPII_plot_flux(name,
                           'grizy'[j],
                           dpeak,
                           sfit,
                           plotlines=25,
                           plot_median=0,
                           Kcor=1,
                           MJD=1,
                           multi=1)
            if saveit != 0:
                plt.savefig(rootdir + saveit + name + '_' + 'grizy'[j] +
                            '.pdf')
        except:
            print "Problem with " + name + '_' + 'grizy'[j]

    if get_neff == 1:
        sne = a[2][-1]

    elif get_neff == 'some':
        sne = a[2]

    if get_neff != 0:
        plt.figure()
        plt.plot(array([sne[c]['n_divergent__'] for c in range(len(sne))]).T)
        plt.ylabel('ndivergent')
        plt.axis([-1, len(sne[0]['n_divergent__']), -0.1, 1.1])
        if saveit != 0: plt.savefig(rootdir + saveit + name + '_treedepth.pdf')

        plt.figure()
        plt.semilogy(array([sne[c]['stepsize__'] for c in range(len(sne))]).T)
        plt.ylabel('stepsize')
        if saveit != 0: plt.savefig(rootdir + saveit + name + '_stepsize.pdf')

        plt.figure()
        plt.plot(array([sne[c]['treedepth__'] for c in range(len(sne))]).T)
        plt.ylabel('treedepth')
        if saveit != 0: plt.savefig(rootdir + saveit + name + '_treedepth.pdf')

    return a


def plotrandomLCs(N=10, a=None, **kwargs):
    if a != None:
        snames, sfit, sneff = a
    for i in range(N):
        spick = random.randint(0, shape(sfit['pt0'])[1])
        bpick = 'grizy'[random.randint(0, 5)]
        try:
            SPII_plot_tm(snames[spick],
                         bpick, [snames, sfit],
                         plotlines=25,
                         plot_median=0,
                         multi=1,
                         Kcor=1,
                         MJD=1,
                         **kwargs)
        except:
            print "Problem with " + snames[spick] + '_' + bpick


def testdiv(data):
    Nchains = len(sneff[2])
    Nc = len(sneff[2][0]['n_divergent__']) / 2
    plt.figure()
    for chain in range(Nchains):
        samples = data[(chain * Nc):Nc * (chain + 1)]
        sdiv = sneff[2][chain]['n_divergent__'][Nc:]

        bins = np.linspace(
            min(samples) - .1 * abs(min(samples)),
            max(samples) + .1 * (max(samples)), 25)
        idig = numpy.digitize(samples, bins)
        adv = array([mean(sdiv[idig == i]) for i in range(len(bins))])
        col = CBcdict[CBcdict.keys()[chain]]
        plt.plot(bins, adv, label='Chain ' + str(chain), c=col)
        plt.plot(samples, sdiv, 'o', c=col)


def getLCp(key,
           j,
           b,
           oktype='',
           bands='grizy',
           eventsel=allevents,
           typedic=ttype_u):
    """
    Get all parameter values from the LCpars dictionary for the parameter "key" and band b
    
    j options:
    * j=0: -1 sigma
    * j=1: median
    * j=2: +1 sigma
    """
    if b == None:
        return array([
            LCpars[name][key][j] for name in eventsel
            if ((typedic[name] == 'SN' + oktype) or (oktype == ''))
        ])
    else:
        return array([
            LCpars[name][key][b][j] for name in eventsel
            if ((typedic[name] == 'SN' + oktype) or (oktype == ''))
        ])


def plotphase(key1,
              key2,
              b1=None,
              b2=None,
              cut=None,
              fig=None,
              ax=None,
              oktypes=['IIp', 'IIn', 'IIb', 'IIl'],
              okcolors=['r', 'purple', 'b', 'orange', 'k'],
              alpha=1.,
              returndata=0,
              eventsel=allevents):
    """
    Plot a phase space diagram for all SNe in the specified parameters
    
    Returns fig,ax
    
    Parameters:
    * key1,key2: Parameters to plot on x and y axes (strings)
    * b1,b2: Band of parameter to use (integers or None)
    * cut: Cuts to make (integer for # sigma or list for cut on uncertainties in each axis)
    * fig: Existing figure to plot on
    * ax: Existing axes to plot on
    * oktypes: Types of SNe to show (e.g. ['IIp','IIl'])
    * alpha: Alpha (opacity) for errorbar plot
    * returndata: Return [x,y,dx1,dx2,dy1,dy2] after fig,ax?
    * eventsel: Event selection array to pass to getLCp
    """
    if ax == None:
        fig = plt.figure()
        ax = plt.axes()
    else:
        fig = ax.figure
    for i in range(len(oktypes)):
        x = getLCp(key1, 1, b1, oktype=oktypes[i], eventsel=eventsel)
        dx1 = getLCp(key1, 0, b1, oktype=oktypes[i], eventsel=eventsel)
        dx2 = getLCp(key1, 2, b1, oktype=oktypes[i], eventsel=eventsel)
        y = getLCp(key2, 1, b2, oktype=oktypes[i], eventsel=eventsel)
        dy1 = getLCp(key2, 0, b2, oktype=oktypes[i], eventsel=eventsel)
        dy2 = getLCp(key2, 2, b2, oktype=oktypes[i], eventsel=eventsel)
        #pdb.set_trace()
        errx = (dx2 + dx1) / 2.
        erry = (dy2 + dy1) / 2.
        nnan = ((isnan(x) == 0) & (isnan(y) == 0))
        if cut == None:
            sel = where(nnan)
        elif cut > 0 and (type(cut) == int or type(cut) == float):
            sel = where(nnan & (errx / x < cut) & (erry / y < cut))
        elif type(cut) == list and len(cut) == 2:
            sel = where(nnan & (errx < cut[0]) & (erry < cut[1]))
        else:
            print "cut type not recognized - ignoring"
            sel = where(x > -inf)

        ## Don't plot labels if only one SN type is being shown
        if len(oktypes) > 1:
            l = oktypes[i]
        else:
            l = None

        ax.errorbar(x[sel],
                    y[sel],
                    xerr=[dx1[sel], dx2[sel]],
                    yerr=[dy1[sel], dy2[sel]],
                    fmt='o',
                    label=l,
                    capsize=0,
                    mew=0,
                    color=okcolors[i],
                    alpha=alpha)

    ax.legend(numpoints=1, ncol=2, prop={'size': 9})
    ax.set_xlabel(key1)
    ax.set_ylabel(key2)
    if returndata: return fig, ax, [x, y, dx1, dx2, dy1, dy2]
    else: return fig, ax


def calc_rhat(sfit, p, takelast=None, sdim1=None, sdim2=None, sdim3=None):
    """
    Calculate rhat for a getCmdStanSamples object
    
    This function arrows for chains of different lengths
    
    Parameters
    * sfit: getCmdStanSamples object
    * p: the name of the parameter of interest (string)
    * takelast: Take only the last *takelast* samples of each chain?
    * sdim1: For multidimensional parameters, take this element within the first axis
    * sdim2: For multidimensional parameters, take this element within the second axis
    * sdim3: For multidimensional parameters, take this element within the third axis
    """
    ## Pick parameter
    if sdim1 == None:
        data = sfit[p]
    elif sdim2 == None:
        data = sfit[p][:, sdim1]
    elif sdim3 == None:
        data = sfit[p][:, sdim1, sdim2]
    else:
        data = sfit[p][:, sdim1, sdim2, sdim3]
    ## separate out the chains
    chains = unique(sfit['chain#'])
    m = len(chains)
    sep_samps = []
    for i in chains:
        if takelast == None:
            sep_samps.append(data[sfit['chain#'] == i])
        else:
            sep_samps.append(data[sfit['chain#'] == i][-takelast:])
    sep_samps = sep_samps
    ## calculate rhat
    s2 = array([var(sep_samps[i]) for i in range(m)])  # within chain variance
    W = mean(s2)  # mean within chain variance
    thetabar = array([mean(sep_samps[i]) for i in range(m)])  # chain means
    n = len(hstack(sep_samps)) / m  # average number of samples per chain
    B = sum((thetabar - mean(thetabar))**2) / (m - 1) * n
    Vtheta = (1 - 1. / n) * W + 1. / n * B
    rhat = sqrt(Vtheta / W)
    return rhat


def sum_of_lognorm(locs, scales, N=1e5):
    """
    Determine the distribution of a sum of lognormal random variables using Monte Carlo
    
    Parameters:
    * locs: Vector of lognormal location parameters, passed to np.random.lognorm
    * scales: Vector of lognormal scale parameters (same length as locs)
    * N: Number of samples to use in Monte Carlo simulation
    """
    ## Simulate lognormal draws
    ys = np.zeros([len(locs), N])
    for i in range(len(locs)):
        ys[i] = random.lognormal(locs[i], scales[i], N)
    ## Fit kde
    gkde = stats.gaussian_kde(sum(ys, axis=0))
    return gkde


def fmtsex(text, dec=0):
    """
    simple function to format a sexigesimal RA/DEC in hh:mm:ss.s format
    
    Parameters:
    * dec: Set this to append a sign to the DEC
    """
    ##catch h m s format
    if 'h' in text or 'd' in text:
        text = text.replace('h',
                            ':').replace('m',
                                         ':').replace('s',
                                                      '').replace('d', ':')
    ##already has colons?
    if ':' in text:
        outtext = ':'.join([obj.zfill(2) for obj in text.split(':')])
        ##space seperated?
    else:
        outtext = ':'.join([obj.zfill(2) for obj in text.split()])
    ##make sure the arcseconds have two decimal places
    arcsec = ('%0.2f' % float(outtext.split(':')[2]))
    outtext = ':'.join(outtext.split(':')[0:2]) + ':' + arcsec.zfill(5)
    ##add sign to dec
    if dec == 1:
        outtext = outtext.replace('–', '-')
        if outtext[0] != '+' and outtext[0] != '-': outtext = '+' + outtext
    return outtext


def sextodeg(RA, DEC):
    """
    convert sexigesimal coordinates to decimals
    """
    ##standardize format
    RA = fmtsex(RA)
    DEC = fmtsex(DEC, dec=1)
    ##Determine DEC sign
    if '-' in DEC: dsign = -1
    else: dsign = 1
    ##split
    RAs = array(RA.split(':'), 'f')
    DECs = abs(array(DEC.split(':'), 'f'))
    ##do math
    RAd = (RAs[0] + RAs[1] / 60. + RAs[2] / 60. / 60) * 15.
    DECd = (DECs[0] + DECs[1] / 60. + DECs[2] / 60. / 60) * dsign
    return [RAd, DECd]


def sextodegl(RAlist, DEClist):
    """convert a list of sexigesimal coordinates to decimal"""
    RAd = []
    DECd = []
    for i in range(len(RAlist)):
        try:
            RA = RAlist[i]
            DEC = DEClist[i]
            [newRA, newDEC] = sextodeg(RA, DEC)
            RAd.append(newRA)
            DECd.append(newDEC)
        except:
            print 'Failed on #' + str(i)
            RAd.append(nan)
            DECd.append(nan)
    return [RAd, DECd]


def degtosex(RA, DEC):
    """
    convert decimal coordinates to sexigesimal
    
    """
    tRA = RA / 15.
    RAh = str(int(tRA))
    RAm = str(int(((tRA - int(tRA)) * 60.)))
    RAs = str('%.2f' % ((tRA - int(tRA) - int(RAm) / 60.) * 60. * 60))
    tDEC = abs(DEC)
    DECh = str(int(tDEC))
    DECm = str(int(((tDEC - int(tDEC)) * 60.)))
    DECs = str('%.1f' % ((tDEC - int(tDEC) - int(DECm) / 60.) * 60. * 60))
    strRA = RAh.zfill(2) + ':' + RAm.zfill(2) + ':' + RAs.zfill(5)
    strDEC = DECh.zfill(2) + ':' + DECm.zfill(2) + ':' + DECs.zfill(4)
    if DEC < 0: strDEC = '-' + strDEC
    else: strDEC = '+' + strDEC
    return [strRA, strDEC]


def eqDist(RA, DEC, RA0, DEC0, unit='min'):
    """
    Simple equatorial distance calculator for sexigesimal or decimal coordinates
    format expected: 00;40:41.2 or decimal
    reference: http://www.skythisweek.info/angsep.pdf
    """
    ##convert to deg, if it's not already
    try:
        radeg = float(RA)
    except:
        radeg = sum(array(RA.split(':'), 'f') * [1, 1 / 60., 1 /
                                                 (60 * 60.)]) * 15
    try:
        decdeg = float(DEC)
    except:
        decdeg = sum(array(DEC.split(':'), 'f') * [1, 1 / 60., 1 / (60 * 60.)])
    try:
        ra0deg = float(RA0)
    except:
        ra0deg = sum(
            array(RA0.split(':'), 'f') * [1, 1 / 60., 1 / (60 * 60.)]) * 15
    try:
        dec0deg = float(DEC0)
    except:
        dec0deg = sum(
            array(DEC0.split(':'), 'f') * [1, 1 / 60., 1 / (60 * 60.)])
    ##radians
    rarad = radeg * pi / 180
    decrad = decdeg * pi / 180
    ra0rad = ra0deg * pi / 180
    dec0rad = dec0deg * pi / 180
    ##calculate distance
    delR = rarad - ra0rad
    delD = decrad - dec0rad
    top = cos(dec0rad)**2 * sin(delR)**2 + (
        cos(decrad) * sin(dec0rad) - sin(decrad) * cos(dec0rad) * cos(delR))**2
    bottom = sin(decrad) * sin(dec0rad) + cos(decrad) * cos(dec0rad) * cos(
        delR)
    dist = 180 / pi * arctan(sqrt(top) / bottom)
    ##return distance in arcmin
    if unit == 'min':
        outdist = dist * 60
    elif unit == 'deg':
        outdist = dist
    elif unit == 'rad':
        outdist = dist * pi / 180.
    elif unit == 'sec':
        outdist = dist * 60. * 60.
    return outdist


def PS1namer(RA, DEC):
    """
    Given RA and DEC in RR:RR:RR.x and DD:DD:DD.x format, output PS1 object names
    in the format defined here: http://ps1sc.ifa.hawaii.edu/PS1wiki/index.php/PS1SC_Publication_Policy#PS1-Lite_papers_:_use_of_small_amounts_of_PS1_data_and_papers_that_use_PS1_data_for_calibration_only
    
    PSO JRRR.rrrr+DD.dddd 
    """
    ## Convert to decimal
    d_ra, d_dec = sextodeg(RA, DEC)
    if d_dec < 0:
        d_sign = '-'
    else:
        d_sign = '+'
    full_ra = '%0.6f' % d_ra
    full_dec = '%0.6f' % abs(d_dec)
    sep_ra = full_ra.split('.')
    sep_dec = full_dec.split('.')
    return 'PSO J' + (sep_ra[0]).zfill(3) + '.' + sep_ra[1][0:4] + d_sign + (
        sep_dec[0]).zfill(2) + '.' + sep_dec[1][0:4]
