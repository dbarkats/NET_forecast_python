#import matplotlib as mpl
#mpl.use('Agg')
from pylab import *
import os,sys
import traceback
from constants import constants
(h,k,c,Tcmb) = constants()

class NETlib():
    """
    class with all necesary libraries to compute bolometer NET

    """
    
    def __init__(self, verbose =True):
        """

        symlink from NETlib.py to the latest version NETlib_v0.X.py

        """        
        self.path = '/n/home01/dbarkats/work/20170801_S4_NET_forecast_python/'
        self.amversion = 9.2
        self.version = 0.2
        self.verbose = verbose
        self.orig_stdout = sys.stdout
        if self.verbose==False:
            self.f = open('output.txt','w')
            sys.stdout = self.f
            
    def getVersion(self):

        scriptFile = os.readlink(self.path+'NETlib.py')
        self.version = scriptFile.split('_')[1].split('.py')[0]
        
    def calc_NET(self,opts):
        
        print "###  NETlib.py version: %s, am version: %s"%(self.version, self.amversion)
        print '' 
        print "Summary of selected options:"
        (atm,band,bolo,s) = self.define_instrument(opts=opts)
        pwv = atm['pwv']

        print("###   v_cen: %.1fGHz, frac_bw: %.2f, v_lo: %.1fGHz, v_hi: %.1fGHz" 
              %(band['v_cen']/1e9, band['frac_bw'], band['v_lo']/1e9, band['v_hi']/1e9))
        print("###   site: '%s', el: %.1f deg, alt: %4.1f meters, pwv: %2.3f mm"
              %(atm['site'], atm['el'], atm['alt'], pwv/1000.))
        print('###   T0: %.3fK, Tc: %.3fK, sf: %.1f, beta: %.1f, Rtes: %.3f, Rshunt: %.3f' 
              %(bolo['T0'], bolo['Tc'],bolo['sf'],bolo['beta'],bolo['Rtes'],bolo['Rshunt']))
        print " "
  
        (Q,Qv,nep1,dPdT) = self.calc_photon_noise(atm,band,s) 
        (bolo,nep2) = self.calc_det_noise(bolo,Q)
  
        # sum up total noise
        nep = {}
        nep['photon'] = sqrt(nep1['shot']**2 + nep1['bose']**2) ;
        nep['detector'] = sqrt(nep2['phonon']**2 + nep2['shunt']**2 + nep2['tes']**2) ;
        nep['tot'] = sqrt(nep['photon']**2 + nep['detector']**2) ;
  
        print('nep.shot:                        %3.3e [W/sqrt(Hz)]'%nep1['shot'])
        print('nep.bose:                        %3.3e [W/sqrt(Hz)]'%nep1['bose'])
        print('nep.phot (shot+bose):            %3.3e [W/sqrt(Hz)]'%nep['photon'])
        print('nep.phonon:                      %3.3e [W/sqrt(Hz)]'%nep2['phonon'])
        print('nep.shunt:                       %3.3e [W/sqrt(Hz)]'%nep2['shunt'])
        print('nep.tes:                         %3.3e [W/sqrt(Hz)]'%nep2['tes'])
        print('nep.detector (phonon+shunt+tes): %3.3e [W/sqrt(Hz)]'%nep['detector'])
        print('nep.total:                       %3.3e [W/sqrt(Hz)]'%nep['tot'])
        print('dPdTcmb: %3.4f pW/K, and dPdTrj: %3.4f pW/K'%(dPdT['cmb']*1e12,dPdT['rj']*1e12))
        print('Gc: %3.2f pW/K'%(bolo['Gc']*1e12))
        print('')

        # we say we are 'photon noise limited' if NEP_photon >~ NEP_detector
        # calc experiment sensitivity to uK_cmb
        net = 1e6* nep['tot'] / (dPdT['cmb'] * sqrt(2))  #  uK_cmb sqrt(s)
        print('NET: %6.2f uK sqrt(s)'%(net))
        
        if not(self.verbose):
            sys.stdout = self.orig_stdout
            self.f.close()

        return net,pwv

    def getAtmosphere(self, atm = None):
        """
        atm is a dictionnary containing the following keys
        site: site name
        el : elevation of the observation through the atmosphere
        
        file: file containing the am output
        v: array of frequency
        dv: frequency step
        Tx: array of Transmission coef
        
        """
        atmdir = '/n/home01/dbarkats/work/20170801_S4_NET_forecast_python/am_spectra/'
        
        zones = ['Antarctic','Arctic','tropical','northern_midlatitude','southern_midlatitude']
        zones_alts = array(range(0,11)+[20,30,40,50])*1000

        if atm == None:
            atm = {}
        if 'site' not in atm.keys():
            atm['site'] = 'SP' # South Pole  default site
        if 'el' not in atm.keys():
            atm['el'] = 90 # el=90 is the default elevation
        if 'alt' not in atm.keys():
            atm['alt'] = 0  # altitude=0m is default for zones
        
        if atm['site'] == 'SP':
            name = 'SPole'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 2835
        elif atm['site'] == 'CP':
            name = 'ALMA'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 5060
        elif atm['site'] == 'ACT':
            name = 'ACT'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 5145
        elif atm['site'] == 'SU':
            name = 'Summit'
            atm['alt'] = 3200
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
        elif atm['site'] == 'TI':
            name = 'Ali'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 5000
        elif atm['site'] == 'MK':
            name = 'MaunaKea'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 4100
        elif atm['site'] == 'LMT':
            name = 'LMT'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 4600
        elif atm['site'] == 'MA':
            name = 'CambridgeMA'
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt'] = 50
        elif atm['site'] == 'SD':
            name = 'Sardinia'    
            atm['file']='%s_annual_50_el%s.out'%(name,str(atm['el']))
            atm['alt']  = 600
        elif atm['site'] in zones:  # Zones
            name = atm['site']
            i1 = sort(argsort(abs(zones_alts-atm['alt']))[0])
            atm['alt'] = zones_alts[i1]
            atm['file']='%s_annual_trunc%d_el%s.out'%(name,atm['alt'],str(atm['el']))
        else:
            print >> self.orig_stdout, 'ERROR: atm.site must be one of: SP, CP, SU, TI for the moment'
            sys.exit()
        
        # load in an atmospheric transmission spectrum from site
        x = genfromtxt(atmdir+atm['file'],delimiter='',dtype=None, comments='%')
        atm['v'] = x[1:,0]*1e9; # in  GHz
        atm['dv'] = atm['v'][1] - atm['v'][0] # freq step size in Hz
        
        # hard coded output list is v [GHz], tau, Tx, Trj[K], Tb[K]
        atm['Tx'] = x[1:,2];  #Tx
        atm['Trj']=x[1:,3]  #Trj

        errfile =atm['file'].replace('.out','.err')
        inErrFile = open(atmdir+errfile,'r')
        for line in inErrFile:
            if line[0] == '#' and 'dry_air' in line:
                atm['dry_air'] = float(line.split()[2])
            if line[0] == '#' and 'h2o' in line:
                atm['pwv'] = float(next(inErrFile).split()[1][1:])
            if line[0] == '#' and 'o3' in line:
                atm['o3'] = float(line.split()[2])
  
        return atm

    def define_instrument(self, opts):

        """
        [atm,band,bolo,s]
        Define all the relevant instrument parameters
        input is just atmospheric Tx spectra
        output is:
           - s structure which gives us the layers of the instrument  
           - add a fake first layer above atmosphere for CMB with emissivity = 1 and transmission = 1
           - band.v_cen band.frac_bw, band.v_lo, band.v_hi  
           - bolo structure defines the device parameters
        """
            
        # define atmosphere 
        atm = self.getAtmosphere(opts['atm']) ;

        # define bandpass
        if 'band' in opts.keys():
            band = opts['band'];
        else:
            band = {}
            band['v_cen'] = 95e9; # GHz
            band['frac_bw'] = 0.24;  # fractional BW, (delta_v/v)
        band = self.calc_band(band);

        # define bolometer device parameters 
        bolo={}
        if 'T0' in opts['bolo'].keys():
            bolo['T0'] = opts['bolo']['T0']
        else:
            bolo['T0']  = 0.250;   #K, physical temp of antenna

        if 'Tc' in opts['bolo'].keys():
            bolo['Tc'] = opts['bolo']['Tc']
        else:
            bolo['Tc']  = 2 *  bolo['T0']  # bath temperature

        if 'sf' in opts['bolo'].keys():
            bolo['sf'] = opts['bolo']['sf']
        else:
            if band['v_cen'] < 60e9:
                bolo['sf'] = 2.0 ;     # safety factor
            elif band['v_cen'] < 110e9:
                bolo['sf'] = 2.5 ;
            else:
                bolo['sf'] = 3.0 ;

        if 'beta' in opts['bolo'].keys():
            bolo['beta'] = opts['bolo']['beta']
        else:
            bolo['beta'] = 2.0 # conductance scales at T^beta

        if 'Rtes' in opts['bolo'].keys():
            bolo['Rtes'] = opts['bolo']['Rtes']
        else:
            bolo['Rtes'] = 0.05 #  Ohm
        
        if 'Rshunt' in opts['bolo'].keys():
            bolo['Rshunt'] = opts['bolo']['Rshunt']
        else:
            bolo['Rshunt'] = 0.003 #  Ohm
        
        bolo['Ldc'] = 20;        # loop gain
        bolo['L'] = bolo['Ldc']/(bolo['Ldc']-1); # correction factor for shunt noise

       ## Define layers
       # each layer has a T, eps , and Tx
       # T is physical temperature of element
       # eps is emissivity of the element (from 0 to 1) or absorption coef.
       # Tx is transmission coef of element (Tx = 1 - eps)
       # optical efficiency is Tx through the instrument.
        
        Tx1 = ones(size(atm['Tx']));
        if 's' in  opts.keys():
            s = opts['s']
            print('###   User-defined opts for instrument layers')
        else:
            print('###   Default instrument layers')
            s={}
            for k in ['name','T','eps','Tx']:
                s[k]=[]
            
            if band['v_cen'] < 60e9:
                s['name'].append('window');  s['T'].append(280);       s['eps'].append(0.01) 
                s['name'].append('blocker1');s['T'].append(150);       s['eps'].append(0.01) 
                s['name'].append('blocker2');s['T'].append(70);        s['eps'].append(0.01) 
                s['name'].append('blocker3');s['T'].append(30);        s['eps'].append(0.01) 
                s['name'].append('lenses');  s['T'].append(5);         s['eps'].append(0.15) 
                s['name'].append('antenna'); s['T'].append(bolo['T0']);s['eps'].append(0.60) 

            elif band['v_cen'] < 110e9:
                s['name'].append('window');  s['T'].append(280);       s['eps'].append(0.02) 
                s['name'].append('blocker1');s['T'].append(150);       s['eps'].append(0.01) 
                s['name'].append('blocker2');s['T'].append(70);        s['eps'].append(0.01) 
                s['name'].append('blocker3');s['T'].append(30);        s['eps'].append(0.02) 
                s['name'].append('lenses');  s['T'].append(5);         s['eps'].append(0.15) 
                s['name'].append('antenna'); s['T'].append(bolo['T0']);s['eps'].append(0.60) 

            elif band['v_cen'] < 183e9:
                s['name'].append('window');  s['T'].append(280);       s['eps'].append(0.03) 
                s['name'].append('blocker1');s['T'].append(150);       s['eps'].append(0.01) 
                s['name'].append('blocker2');s['T'].append(70);        s['eps'].append(0.01) 
                s['name'].append('blocker3');s['T'].append(30);        s['eps'].append(0.02) 
                s['name'].append('lenses');  s['T'].append(5);         s['eps'].append(0.15) 
                s['name'].append('antenna'); s['T'].append(bolo['T0']);s['eps'].append(0.60)
            
            else:
                s['name'].append('window');  s['T'].append(280);       s['eps'].append(0.04) 
                s['name'].append('blocker1');s['T'].append(150);       s['eps'].append(0.03) 
                s['name'].append('blocker2');s['T'].append(70);        s['eps'].append(0.03) 
                s['name'].append('blocker3');s['T'].append(30);        s['eps'].append(0.03) 
                s['name'].append('lenses');  s['T'].append(5);         s['eps'].append(0.15) 
                s['name'].append('antenna'); s['T'].append(bolo['T0']);s['eps'].append(0.60)

        # multiply all eps (custom layers and default layers) by Tx1 so it's the right size
        for k, eps in enumerate(s['eps']):
            s['eps'][k] = Tx1 * eps

        # insert cmb and atmosphere layers to custom and default layers
        s['name'].insert(0,'atm');s['T'].insert(0,250);s['Tx'].insert(0,atm['Tx']);s['eps'].insert(0,1-atm['Tx']);
        s['name'].insert(0,'cmb');s['T'].insert(0,Tcmb);s['Tx'].insert(0,Tx1);s['eps'].insert(0,Tx1);
        
        #TODO: change atm['T'] to be consistent with atm['Trj']

        # set eps of last element to 0 in band and 1 outside band
        s['eps'][-1][find(atm['v'] < band['v_lo'])]=1; 
        s['eps'][-1][find(atm['v']> band['v_hi'])]=1;
        
        # define Tx of all other layers
        for i in range(shape(s['Tx'])[0],size(s['name'])):
            s['Tx'].append(1-s['eps'][i] )
            
        return [atm,band,bolo,s]


    def  calc_photon_noise(self,atm,band,s):
        
        """
        calculates the photon noise contribution to the NEP

        """
  
        # define indices of bandpass
        q=find( (atm['v'] < band['v_hi']) & (atm['v'] > band['v_lo'])) 

        Nv = size(atm['Tx'])  # number of frequency points
        Nl = size(s['name'])  # number of instrument layers
        Tx_tot = ones(Nv) ;  # total transmission coef
        Tx_inst = ones(Nv) ; # instrument-only transmission coef

        # compute total and instrument Tx coef.
        for i in range(size(s['name'])):
            Tx_tot = Tx_tot * s['Tx'][i];
            if i >= 2:
                Tx_inst = Tx_inst * s['Tx'][i]

        # plot total transmission
        if(0):
            clf()
            plot(atm['v']/1e9,Tx_inst,'b','linewidth',3)
            plot(atm['v']/1e9,Tx_tot,'k','linewidth',3)

         # get W/K conversion factor
        dPdT = self.calc_dPdT(atm,Tx_tot)
        
        # calculate optical power per unit wavelength at detector from each source:
        # this is the Planck brightness times the source emissivity 
            
        class Power:
            tot = 0  # power incident from all elements between cmb and detector:
            inst = 0 # power incident only from instrument-related parts
            Trjtot = 0
            Trjinst = 0

        Q = Power();
        Qvtot = 0
        
        s['Txm']= []
        s['cumTxm']= []
        s['Qv']= []
        s['Q']= []
        s['Trj'] = []
        
        if (self.verbose):
            print '###   Summary of instrument layers and photon noise'

        for i in range(Nl):
            trans = ones(Nv);
            for j in range(i+1,Nl):
                trans = trans * s['Tx'][j];

            s['Txm'].append(mean(s['Tx'][i][q]))
            s['cumTxm'].append(mean(trans[q])) ;
            s['Qv'].append(self.planck_v(atm['v'],s['T'][i],1) * s['eps'][i] * trans) ;
            # integrate over v to get total optical power
            s['Q'].append(sum(s['Qv'][i]) * atm['dv'])  # in W
            s['Trj'].append(s['Q'][i]/dPdT['rj'])  # in Krj
            if(self.verbose):
                print('%12s Tx:%2.2f, cumulTx to det: %2.3f, Power:%2.3f [pW], Trj: %5.2f [Krj]'
                     %(s['name'][i], s['Txm'][i], s['cumTxm'][i], s['Q'][i]*1e12, s['Trj'][i]));

            Q.tot = Q.tot + s['Q'][i] ;
            Q.Trjtot = Q.Trjtot + s['Trj'][i] ;
            Qvtot = Qvtot + s['Qv'][i] ;
            if i >= 2:
                Q.inst = Q.inst + s['Q'][i];
                Q.Trjinst = Q.Trjinst + s['Trj'][i];

        if self.verbose:
            print('                                       Total Power:%2.3f [pW], Trj: %5.2f [Krj]'
                  %(Q.tot *1e12, Q.Trjtot))
            print('                                   Inst-only Power:%2.3f [pW], Trj: %5.2f [Krj]'
                  %(Q.inst*1e12, Q.Trjinst));
            print (' ')
        nep = {}
        nep['shot'] = sqrt(2* sum(Qvtot * h *atm['v'])*atm['dv'] ) ; # W / sqrt(Hz)
        nep['bose'] = sqrt(2* sum(Qvtot**2)*atm['dv'] )    ; # W / sqrt(Hz)
        
        return  Q,Qvtot,nep,dPdT


    def calc_det_noise(self,bolo,Q):

        nep = {}
        bolo['Gc'] = self.calc_Gc(bolo,Q) ;
        nep['phonon'] = self.calc_phonon_noise(bolo) ;
        (nep['shunt'],nep['tes']) = self.calc_johnson_noise(bolo, Q)
        
        return bolo, nep

    def calc_Gc(self, bolo, Q):
        
        Gc = bolo['sf']*Q.tot/bolo['Tc'] * (1+bolo['beta'])/(1-(bolo['T0']/bolo['Tc'])**(1+bolo['beta'])) # W/K
        return Gc

    def calc_phonon_noise(self,bolo):
        

        D = 1-(bolo['T0']/bolo['Tc'])    # see Mather eq 34
        F = sqrt(1 - (1+bolo['beta']/2)*D + (bolo['beta']+2)*(3*bolo['beta']+2)/12*D**2) 
        nep_phonon = sqrt(4 * k * bolo['Gc'] * bolo['Tc']**2 * F**2)      # W / sqrt(Hz)
        
        return nep_phonon
  
    def calc_johnson_noise(self,bolo, Q):

        I_0 = sqrt(Q.tot*(bolo['sf']-1) / bolo['Rshunt']) # Amps
        nep_shunt = sqrt(4*k*bolo['T0']*bolo['Rshunt']*I_0**2/bolo['L']**2)  # W / sqrt(Hz)
        nep_tes = sqrt(4*k*bolo['Tc']*bolo['Rtes']*I_0**2/bolo['Ldc']**2)      # W / sqrt(Hz)
    
        return nep_shunt, nep_tes


    def calc_band(self, band):
        """
        recalculates the band edges given  v_cen and a frac_bw

        """
        band['bw'] = band['v_cen']*band['frac_bw']    # BW
        band['v_lo'] = band['v_cen'] - band['bw']/2;  # band edge low
        band['v_hi'] = band['v_cen'] + band['bw']/2;  # band edge high
        
        return band


    def planck_v(self,v,T,sm=1):
        """
         calculates the Planck spectral radiance as a function of frequency 
         and Temperature of the emitter
         v is an array of frequencies in Hz
         T is a single temperature in K
         if sm = 0, the result is the generic Planck blackbody spectral radiance
         if sm = 1, the result is the single moded expression. This is the default.

        """
        x=(h*v)/(k*T)
  
        # spectral radiance  (W m^-2 sr^-1 Hz^-1)
        Bv = ( (2*h*v**3) / c**2 ) * ( 1 / ( exp(x) - 1 ) );

        # Power per unit bandwidth absorbed by a detector is
        # Qv =  A*Omega * Bv
        # Antenna theorem states A*Omega = lambda^2 so
        # Qv = lambda^2 * 2* h * v^3/c^2 *  1 ./ ( exp((v.*h)./(k.*T)) - 1 ) );
        # Qv = h*v *  1 ./ ( exp((v.*h)./(k.*T)) - 1 ) );
        # Qv is the single moded expression 
        # for the power  per unit frequency absorbed by a single moded antenna, 1 polarization (that is 
        # where the factor of 1/2 disappeared)

        # single moded expression
        Qv = (h*v) * ( 1 / ( exp(x) - 1 ) );

        if(sm):
            return Qv;
        else:
            return Bv;

    def planck_dIdT(self, v,T,sm = 1):

        """
        Partial derivative of the Planck function with respect
        to temperature at given temperature and frequency.
        
        Input in Hz and K
        Output in W m^-2 Hz^-1 ster^-1 K^-1

        """
        x=(h*v)/(k*T);        
        dIdT=((2*h**2*v**4)/(c**2*k*T**2))*(exp(x)/(exp(x)-1)**2);

        # the factor of 0.5* c^2/v^2 is to convert to single moded power per unit freq.
        if sm == 1:
            dIdT = dIdT * 0.5 * c**2 /v**2 ; 

        return dIdT

    def calc_dPdT(self,atm,Tx_tot):
  
        #dT = .0001;
        #Qv_cmb_plus_dT = planck_v(atm['v'],T+dT,1) * Tx_tot ;
        #Qv_cmb =         planck_v(atm['v'],T,1)  * Tx_tot;
        #dPdT = (sum(Qv_cmb_plus_dT)*atm['dv'] - sum(Qv_cmb)*atm['dv'])/dT  # W/K_cmb

        # 27/03/2017: dB replaces the numerical derivative 
        # with the planck_dIdT function which provides the analytical derivative

        v = atm['v']
        dv = atm['dv']

        dPdT = {}
        dPdT['cmb'] = nansum(Tx_tot * self.planck_dIdT(v,Tcmb))*dv;
        dPdT['rj'] = nansum(Tx_tot * self.planck_dIdT(v,1000))*dv;
        dPdT['rj2cmb'] = dPdT['cmb']/dPdT['rj'];

        return dPdT


######
# the following methods should move to amLib.py
####
    def calcPgnd(self,Hgnd):
        """
        calc Pgnd in mBar
        given Hgnd in meters
        """
        Pgnd = 10* 101.29 * ((15.04 - .00649 * Hgnd + 273.1)/288.08)**5.256
        return Pgnd
    
    def truncateProfile(self,l, Hgnd):
        """
        given l layer, find  last layer 
        when we trunctate at Hgrnd height in meters
        """
        
        # find 2 closest layers in height
        Hbase = Hgnd
        s=abs(array(l.Hbase)-Hgnd)
        i1,i2 = sort(argsort(s)[:2])
        Pinterp = [l.Pbase[i1], l.Pbase[i2]]
        Pbase = interp(Hgnd,[l.Hbase[i2], l.Hbase[i1]],[l.Pbase[i2],l.Pbase[i1]])
        Tbase = interp(Pbase,Pinterp,[l.Tbase[i1],l.Tbase[i2]])
        h2obase = interp(Pbase,Pinterp,[l.h2o[i1],l.h2o[i2]])
        o3base = interp(Pbase,Pinterp,[l.o3[i1],l.o3[i2]])

        #print Pbase, Pinterp
        #print Hbase, l.Hbase[i1:i2+1]
        #print Tbase,  l.Tbase[i1:i2+1]
        #print h2obase, l.h2o[i1:i2+1]
        #print o3base, l.o3[i1:i2+1]

        return i1,i2, Pbase, Tbase, h2obase, o3base
    
    def rewriteAmc(self,filename,Hgnd = 0):   
        
        L = self.readAmc(filename)
        i1,i2, Pbase, Tbase, h2obase, o3base = self.truncateProfile(L,Hgnd)
        
        filebase = filename.split('.amc')[0]
        filebase_trunc = '%s_trunc%d'%(filebase,Hgnd)
        if os.path.isfile('am_spectra/%s.amc'%filebase_trunc):
            return filebase_trunc
        Hgnd = int(Hgnd)
        g = open('am_spectra/%s.amc'%filebase_trunc,'w')
        f = open('am_spectra/%s.amc'%filebase,'r')

        lines = f.readlines()
        last_layer = False
        done = False
        for line in lines:
            if done ==True: 
                break
            if (line.startswith('#')) or (line.startswith('?')) or (line == '\n'):
                g.write(line)
                continue
            sline = line.split()
            if sline[0] =='Pbase':
                if float(sline[1]) == L.Pbase[i2]:
                    last_layer = True
                if last_layer == True:
                    newline = line.replace(sline[1],'%3.3f'%Pbase,1)
                    newline = newline.replace(sline[6],'%3.0f'%Hgnd)
                    g.write(newline)
                else:
                    g.write(line)        
            elif sline[0] =='Tbase' :
                if last_layer == True:
                    g.write(line.replace(sline[1],'%3.3f'%Tbase,1))
                else:
                    g.write(line) 

            elif (sline[0] =='column') and (sline[1] =='dry_air'):
                g.write(line)

            elif (sline[0] =='column') and (sline[1] =='h2o'):
                if last_layer == True:
                    g.write(line.replace(sline[3],'%3.3e'%h2obase,1))
                else:
                    g.write(line) 
            elif (sline[0] =='column') and (sline[1] =='o3'):
                if last_layer == True:
                    g.write(line.replace(sline[3],'%3.3e'%o3base,1))
                    done = True
                else:
                    g.write(line) 
            else:
                g.write(line)
            
        f.close()
        g.close()
        return filebase_trunc
        
                    
    def readAmc(self,filename):
        """
        """
        class Layer:
            Pbase = []
            Hbase = []
            Tbase = []
            o3 = []
            h2o = []
            dry_air = []
        L = Layer();
        
        f = open('am_spectra/%s'%filename,'r')
        lines = f.readlines()

        for line in lines:
            if (line.startswith('#')) or (line.startswith('?')) or (line == '\n'):
                continue
            sline = line.split()
            if sline[0] =='Pbase' :
                L.Pbase.append(float(sline[1]))
                L.Hbase.append(float(sline[6]))
            elif sline[0] =='Tbase' :
                L.Tbase.append(float(sline[1]))
            elif (sline[0] =='column') and (sline[1] =='dry_air'):
                L.dry_air.append(sline[2]) 
            elif (sline[0] =='column') and (sline[1] =='h2o'):
                L.h2o.append(float(sline[3]))
            elif (sline[0] =='column') and (sline[1] =='o3'):
                L.o3.append(float(sline[3]))
        
        f.close()
        
        return L
           
