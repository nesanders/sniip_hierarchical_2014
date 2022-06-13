data {
  int<lower=0> N_obs;                             // number of observations 
  int<lower=0> N_SN;                              // number of objects (SNe)
  int<lower=0> N_filt;                            // number of photometric filters
  vector[N_obs] t;                                // MJD of observation
  vector[N_obs] fL;                               // Luminosity (linear brightness)
  vector[N_obs] dfL;                              // Uncertainty in brightness (flux)
  vector[N_SN] z;                                 // Redshift of each object
  vector[N_SN] t0_mean;                           // Guess at pt0 for each object
  matrix[N_SN, N_filt] mzero;                     // r-band magnitude zero point of each object
  vector[N_SN] distmod;                           // Distance modulus (mag) of each object
  int<lower=1,upper=N_filt> J[N_obs];             // Integer array representing band of observation
  int<lower=1,upper=N_SN> SNid[N_obs];            // Integer array representing object (supernova) within sample
  int<lower=0> Kcor_N;                            // Number of time steps in the K-correction array
  real Kcor[N_SN, N_filt,Kcor_N];                 // K correction curves per object and filter (mag)
  real<lower=0> fluxscale;                        // Float to adjust peak flux variable by (something like the 90th percentile flux value should work)
  vector<lower=0,upper=1>[N_SN] duringseason;     // Was the SN discovered during a season or between seasons?
}

transformed data {
  vector[N_filt] prior_t_hF[4];                           // Prior over time-filter effects
  vector[N_filt] prior_t_hF_s[4];
  vector[N_filt] prior_r_hF[5];                           // Prior over rate-filter effects
  vector[N_filt] prior_r_hF_s[5];
  
  // t1 filter priors
  for (i in 1:N_filt) {
    prior_t_hF[1,i] <- 0;
    prior_t_hF_s[1,i] <- 0.1;
  }
  // tp filter priors
  prior_t_hF[2,1] <- -1;
  prior_t_hF[2,2] <- -0.5;
  prior_t_hF[2,3] <- 0;
  prior_t_hF[2,4] <- 0.5;
  prior_t_hF[2,5] <- 1;
  for (i in 1:N_filt) {prior_t_hF_s[2,i] <- 0.1;}
  // t2 filter priors
  for (i in 1:N_filt) {
    prior_t_hF[3,i] <- 0;
    prior_t_hF_s[3,i] <- 0.1;
  }
  // td filter priors
  for (i in 1:N_filt) {
    prior_t_hF[4,i] <- 0;
    prior_t_hF_s[4,i] <- 0.1;
  }
  
  // lalpha filer priors
  for (i in 1:N_filt) {
    prior_r_hF[1,i] <- 0;
    prior_r_hF_s[1,i] <- 0.1;
  }
  // lbeta1 filer priors
  prior_r_hF[2,1] <- 2;
  prior_r_hF[2,2] <- 1;
  prior_r_hF[2,3] <- 0;
  prior_r_hF[2,4] <- -0.5;
  prior_r_hF[2,5] <- -1;
  for (i in 1:N_filt) {prior_r_hF_s[2,i] <- 0.1;}
  // lbeta2 filer priors
  prior_r_hF[3,1] <- 1;
  prior_r_hF[3,2] <- 0.3;
  prior_r_hF[3,3] <- 0;
  prior_r_hF[3,4] <- -1;
  prior_r_hF[3,5] <- -1;
  for (i in 1:N_filt) {prior_r_hF_s[3,i] <- 0.1;}
  // lbetadN filer priors
  for (i in 1:N_filt) {
    prior_r_hF[4,i] <- 0;
    prior_r_hF_s[4,i] <- 0.1;
  }
  // lbetadC filer priors
  for (i in 1:N_filt) {
    prior_r_hF[5,i] <- 0;
    prior_r_hF_s[5,i] <- 0.1;
  }
  
}

parameters {  
  //// Time parameters
  vector[4] t_hP;                                 // Parameter-level time parameters, not including t0
  vector<lower=0>[4] sig_t_hP;                    // Parameter-level time sigmas
  vector[N_filt] t_hF[4];                         // Filter-level time parameters
  vector<lower=0>[N_filt] sig_t_hF[4];            // Filter-level time sigmas
  vector[N_SN * N_filt] t_hSNF[4];                // SN-filter time interactions
  vector<lower=0>[N_SN * N_filt] sig_t_hSNF[4];   // SN-filter time interaction sigmas
  
  //// Rate parameters
  vector[5] r_hP;                                 // Parameter-level rate parameters, not including t0
  vector<lower=0>[5] sig_r_hP;                    // Parameter-level rate sigmas
  vector[N_filt] r_hF[5];                         // Filter-level rate parameters
  vector<lower=0>[5] sig_r_hF[5];                 // Filter-level rate sigmas
  vector[N_SN * N_filt] r_hSNF[5];                // SN-filter rate interactions
  vector<lower=0>[N_SN * N_filt] sig_r_hSNF[5];   // SN-filter rate interaction sigmas
  
  //// Flux parameters
  real M_h;                                       // Top level flux parameter
  real<lower=0> sig_M_h;                          // Top level flux sigma
  vector[N_filt] M_hF;                            // Filter-level flux parameters
  vector<lower=0>[N_filt] sig_M_hF;               // Filter-level flux sigmas
  vector[N_SN * N_filt] M_hSNF;                   // SN-filter flux interactions
  vector<lower=0>[N_SN * N_filt] sig_M_hSNF;      // SN-filter flux interaction sigmas
  
  //// Background parameters
  real Y_h;                                      // Top level background parameter
  real<lower=0> sig_Y_h;                         // Top level background sigma
  vector[N_SN * N_filt] Y_hSNF;                    // Local background parameter
  vector<lower=0>[N_SN * N_filt] sig_Y_hSNF;       // Local background sigma

  //// t0 parameters
  real t0s_h;                                   // Top level short-time parameter
  real<lower=0> sig_t0s_h;                      // Top level short-time sigma
  vector[N_SN] t0s_hSN;                         // SN-level short-time parameter
  vector<lower=0>[N_SN] sig_t0s_hSN;            // SN-level short-time sigma
  real t0l_h;                                   // Top level long-time parameter
  real<lower=0> sig_t0l_h;                      // Top level long-time sigma
  vector[N_SN] t0l_hSN;                         // SN-level long-time parameter
  vector<lower=0>[N_SN] sig_t0l_hSN;            // SN-level long-time sigma
  
  //// Intrinsic scatter parameters
  real<lower=0> V_h;                            // Top level scatter parameter
  vector<lower=0>[N_filt] V_hF;                 // Filter-level scatter parameters
  vector<lower=0>[N_SN * N_filt] V_hSNF;          // SN-filter scatter interactions
}


transformed parameters {
    vector[N_obs] mm;                         // SNIIP model flux
    vector[N_obs] dm;                         // SNIIP model uncertainty
    
    // Define local light curve parameters
    vector<upper=0>[N_SN] pt0;
    matrix<lower=0>[N_SN, N_filt] t1;
    matrix<lower=0>[N_SN, N_filt] t2;
    matrix<lower=0>[N_SN, N_filt] td;
    matrix<lower=0>[N_SN, N_filt] tp;
    matrix[N_SN, N_filt] lalpha; 
    matrix[N_SN, N_filt] lbeta1;
    matrix[N_SN, N_filt] lbeta2;
    matrix[N_SN, N_filt] lbetadN;
    matrix[N_SN, N_filt] lbetadC; 
    matrix[N_SN, N_filt] Mp; 
    matrix[N_SN, N_filt] Yb;
    matrix<lower=0>[N_SN, N_filt] V;
    // Define flux parameters
    matrix<lower=0>[N_SN, N_filt] M1;
    matrix<lower=0>[N_SN, N_filt] M2;
    matrix<lower=0>[N_SN, N_filt] Md;
    
    // Calculate transformed explosion date offset
    for (l in 1:N_SN) {
      if (duringseason[l] == 1) {
        pt0[l] <- -exp( t0s_h + sig_t0s_h * ( t0s_hSN[l] .* sig_t0s_hSN[l] )); 
      } else {
        pt0[l] <- -exp( t0l_h + sig_t0l_h * ( t0l_hSN[l] .* sig_t0l_hSN[l] ));
      }
    }
    
    // Transformed hierarchical (bottom-level) parameters
    for (i in 1:N_filt) { // Step through filters
	
        for (j in 1:N_SN) { // Step through SNe
            t1[j,i] <- exp( log(1) + t_hP[1] + sig_t_hP[1] * (
                       t_hF[1,i] * sig_t_hF[1,i] 
                     + sig_t_hSNF[1,(i-1)*N_SN+j] * t_hSNF[1,(i-1)*N_SN+j]
                       ));
	    
            tp[j,i] <- exp( log(10) + t_hP[2] + sig_t_hP[2] * (
                       t_hF[2,i] * sig_t_hF[2,i] 
                     + sig_t_hSNF[2,(i-1)*N_SN+j] * t_hSNF[2,(i-1)*N_SN+j]
                       ));
   
            t2[j,i] <- exp( log(100) + t_hP[3] + sig_t_hP[3] * (
                       t_hF[3,i] * sig_t_hF[3,i] 
                     + sig_t_hSNF[3,(i-1)*N_SN+j] * t_hSNF[3,(i-1)*N_SN+j]
                       ));

            td[j,i] <- exp( log(10) + t_hP[4] + sig_t_hP[4] * (
                       t_hF[4,i] * sig_t_hF[4,i] 
                     + sig_t_hSNF[4,(i-1)*N_SN+j] * t_hSNF[4,(i-1)*N_SN+j]
                       ));

            lalpha[j,i] <- -1 + ( r_hP[1] + sig_r_hP[1] * (
                           r_hF[1,i] * sig_r_hF[1,i] 
                         + sig_r_hSNF[1,(i-1)*N_SN+j] * r_hSNF[1,(i-1)*N_SN+j]
                           ));

            lbeta1[j,i] <- -4 + ( r_hP[2] + sig_r_hP[2] * (
                           r_hF[2,i] * sig_r_hF[2,i] 
                         + sig_r_hSNF[2,(i-1)*N_SN+j] * r_hSNF[2,(i-1)*N_SN+j]
                           ));

            lbeta2[j,i] <- -4 + ( r_hP[3] + sig_r_hP[3] * (
                           r_hF[3,i] * sig_r_hF[3,i] 
                         + sig_r_hSNF[3,(i-1)*N_SN+j] * r_hSNF[3,(i-1)*N_SN+j]
                           ));

            lbetadN[j,i] <- -3 + ( r_hP[4] + sig_r_hP[4] * (
                            r_hF[4,i] * sig_r_hF[4,i] 
                          + sig_r_hSNF[4,(i-1)*N_SN+j] * r_hSNF[4,(i-1)*N_SN+j]
                            ));

            lbetadC[j,i] <- -5 + ( r_hP[5] + sig_r_hP[5] * (
                            r_hF[5,i] * sig_r_hF[5,i] 
                          + sig_r_hSNF[5,(i-1)*N_SN+j] * r_hSNF[5,(i-1)*N_SN+j]
                            ));

            Mp[j,i] <- exp(M_h + sig_M_h * (
                           M_hF[i] * sig_M_hF[i] 
                         + sig_M_hSNF[(i-1)*N_SN+j] * M_hSNF[(i-1)*N_SN+j]
                           ));

            Yb[j,i] <- Y_h + sig_Y_h * (Y_hSNF[(i-1)*N_SN+j] .* sig_Y_hSNF[(i-1)*N_SN+j]);

            V[j,i] <- V_h * V_hF[i] * V_hSNF[(i-1)*N_SN+j]; //  * V_hSN[j]

        }
    }

    // Calculate flux turnover points    
    M1 <- Mp ./ exp( exp(lbeta1) .* tp );
    M2 <- Mp .* exp( -exp(lbeta2) .* t2 );
    Md <- M2 .* exp( -exp(lbetadN) .* td );

    for (n in 1:N_obs) {      
        // Definitions
        real N_SNc;                                      // K correction
        int Kc_up;                                       // K correction helper variable
        int Kc_down;                                     // K correction helper variable
        real t_exp;                                      // time since explosion
        int j;                                           // band of this observation
        int k;                                           // object ID this observation
        real mm_1;
        real mm_2;
        real mm_3;
        real mm_4;
        real mm_5;
        real mm_6;
        
        j <- J[n];
        k <- SNid[n];
        t_exp <- ( t[n] - (t0_mean[k] + pt0[k]) ) / (1 + z[k]);
        
        // Model magnitudes
        // before explosion
        if (t_exp<0) {                                                                      
            mm_1 <- Yb[k,j];
        } else {
            mm_1 <- 0;
        }
        // Explosion rise phase
	if ((t_exp>=0) && (t_exp < t1[k,j])) {                                                                      
            mm_2 <- Yb[k,j] + M1[k,j] * pow(t_exp / t1[k,j] , exp(lalpha[k,j]));
        } else {
            mm_2 <- 0;
        }
        // Slow rise phase
        if ((t_exp >= t1[k,j]) && (t_exp < t1[k,j] + tp[k,j])) {                                                                      
            mm_3 <- Yb[k,j] + M1[k,j] * exp(exp(lbeta1[k,j]) * (t_exp - t1[k,j]));
        } else {
            mm_3 <- 0;
        }
        // Plateau phase
        if ((t_exp >= t1[k,j] + tp[k,j]) && (t_exp < t1[k,j] + tp[k,j] + t2[k,j])) {                                                                      
            mm_4 <- Yb[k,j] + Mp[k,j] * exp(-exp(lbeta2[k,j]) * (t_exp - t1[k,j] - tp[k,j]));
        } else {
            mm_4 <- 0;
        }
        // Ni ->= Co decay phase
        if ((t_exp >= t1[k,j] + tp[k,j] + t2[k,j]) && (t_exp < t1[k,j] + tp[k,j] + t2[k,j] + td[k,j])) {                                                                      
            mm_5 <- Yb[k,j] + M2[k,j] * exp(-exp(lbetadN[k,j]) * (t_exp - t1[k,j] - tp[k,j] - t2[k,j]));
        } else {
            mm_5 <- 0;
        }
        // Co ->= Fe decay phase
        if (t_exp >= t1[k,j] + tp[k,j] + t2[k,j] + td[k,j]) {                                                                      
            mm_6 <- Yb[k,j] + Md[k,j] * exp(-exp(lbetadC[k,j]) * (t_exp - t1[k,j] - tp[k,j] - t2[k,j] - td[k,j]));
        } else {
            mm_6 <- 0;
        }
        // Model + data uncertainty
        dm[n] <- sqrt(pow(dfL[n],2) + pow(V[k,j],2));
        
        // K correction
        if (t_exp<0) {
            N_SNc <- 0;
        } else if  (t_exp<Kcor_N-2){                    //Linearly interpolate between values
            Kc_down <- 0;
            while ((Kc_down+1) < t_exp) {               // find floor(t_exp) - since no (int) floor function exists
		Kc_down <- Kc_down + 1; 
            }
            Kc_up <- Kc_down+1;
            N_SNc <- Kcor[k,j,Kc_down+1] + (t_exp - floor(t_exp)) * (Kcor[k,j,Kc_up+1]-Kcor[k,j,Kc_down+1]);
        } else {                                        // use final value for correction
            N_SNc <- Kcor[k,j,Kcor_N];
        }
        mm[n] <- (mm_1+mm_2+mm_3+mm_4+mm_5+mm_6) / (pow(10, N_SNc/(-2.5)));
    }

}

model {
  
    
    //// PRIORS


    //// Hyperparameter priors

    // Top level means
    t0s_h ~ normal(0, 0.5);                             // Within-season delay ~ 1 day
    sig_t0s_h ~ cauchy(0, 0.1);
    t0l_h ~ normal(log(100), 1);                       // Out-of-season delay ~ 100 days
    sig_t0l_h ~ cauchy(0, 0.1);
    V_h ~ cauchy(0, 0.001);                               // Typical scatter
    Y_h ~ normal(0, 0.1);                               // Typical background level
    sig_Y_h ~ cauchy(0, 0.01);
    M_h ~ normal(0, 1);                                // Flux scale
    sig_M_h ~ cauchy(0, 0.1);

    // Pooling priors
    t_hP ~ normal(0,0.1);                             // Prior width on top-level durations of 50%
    sig_t_hP ~ cauchy(0, 0.1);
    for (i in 1:4) {                                  // Times
      t_hF[i] ~ normal(prior_t_hF[i], prior_t_hF_s[i]);
      sig_t_hF[i] ~ cauchy(0, 0.1);
      t_hSNF[i] ~ normal(0,1);
      sig_t_hSNF[i] ~ cauchy(0, 0.1);
    }
    
    r_hP ~ normal(0,1);                              // Rates
    sig_r_hP ~ cauchy(0, 0.1);
    for (i in 1:5) {
      r_hF[i] ~ normal(prior_r_hF[i], prior_r_hF_s[i]);
      sig_r_hF[i] ~ cauchy(0, 0.1);
      r_hSNF[i] ~ normal(0,1);
      sig_r_hSNF[i] ~ cauchy(0, 0.1);
    }

    M_hF ~ normal(0,1);
    sig_M_hF ~ cauchy(0, 0.1);
    M_hSNF ~ normal(0,1);
    sig_M_hSNF ~ cauchy(0, 0.1);
    
    
    Y_hSNF ~ normal(0,1);                       // Backgrounds
    sig_Y_hSNF ~ cauchy(0, 0.1);
    
    V_hF ~ cauchy(0, 0.1);                       // Scatter
    V_hSNF ~ cauchy(0, 0.1);
    
    t0s_hSN ~ normal(0,1);                           // Explosion date delays
    sig_t0s_hSN ~ cauchy(0, 0.1);
    t0l_hSN ~ normal(0,1);
    sig_t0l_hSN ~ cauchy(0, 0.1);

    //   //// LIKELIHOOD
    fL ~ normal(mm,dm);
}

// reproduce some of the calculations from the model so their samples are output
generated quantities {
    vector[N_SN] t0;    
    vector[N_obs] fL_out;   
    matrix[N_SN, N_filt] mpeak;
    matrix<lower=0>[N_SN, N_filt] tplateau;

    t0 <- pt0 + t0_mean;
    fL_out <- fL * fluxscale;
    
    // Calculate peak magnitudes relative to r
    // Cannot vectorize because log10 is only defined for reals
    for (k in 1:N_SN) {
      for (j in 1:N_filt) {
        mpeak[k,j] <- -2.5*log10(Mp[k,j] * fluxscale); // + distmod[k] + mzero[k,j];
      }
    }
      
    // Calculate plateau rise phase duration
    tplateau <- tp + t2;
}


