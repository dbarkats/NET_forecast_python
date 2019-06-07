#! /usr/bin/env python

from optparse import OptionParser
import dateutil.parser as dparser


# TODO:
    # add capability to read arbitrary bandpasses
    # add different fake bandpass shapes than rectange
    # add standard outputs
    # add results in different units

 # To run reduc_NET, you need to define:
 #  opts.band.v_cen
 #  opts.band.frac_bw
 #  opts.atm.site
 #  opts.bolo.T0

if __name__ == '__main__':
    usage = '''
  
    
    '''
    parser = OptionParser(usage=usage)
    
    parser.add_option("--band_center",
                      dest="v_cen",
                      default = 95e9,
                      type=float,
                      help="--band_center. Band center of the band in Hz. Default: 95e9 Hz")
    
    parser.add_option("--band_width",
                      dest="frac_bw",
                      type=float,
                      default = 0.25,
                      help="--band_width: fractional bandwidth of the band. Defaults: 0.25 ")
    
    parser.add_option("--site",
                      dest="site",
                      type=str,
                      default = 'SP',
                      help="--site: site name. Default: SP for South Pole.")

    parser.add_option("--bath_temp",
                      dest="T0",
                      type = float,
                      default=0.250,
                      help="--bath_temp: bolometer bath temperature in K. Default: 0.250K ")

    parser.add_option("-v","--verbose",
                      dest="verbose",
                      action="store_true",
                      default=False,
                      help="-v will add verbosity, default = False")
 
    parser.add_option("-V","--Version",
                      dest="version",
                      action="store_true",
                      default=False,
                      help="-V: print the version of the python script")
    
    (options, args) = parser.parse_args()


    import NETlib as nl
    n = nl.NETlib(options.verbose)

    if options.version:
        version = n.version
        print('NETlib.py version %s'%version)
        exit()

    opts = {}
    opts['band']={}
    opts['band']['v_cen'] = options.v_cen
    opts['band']['frac_bw']=options.frac_bw
    
    opts['atm'] = {}
    opts['atm']['site'] = options.site
    
    opts['bolo']={}
    opts['bolo']['T0']=options.T0
    
    n.calc_NET(opts)
#    
# 
#  outfile = sprintf('nets_%sGHz_%s_T0%03dmK.mat',band.name,site,T0*1e3)
#  save(outfile,'nets','freqs','freqs','frac_bws','atm','band','bolo','s')

