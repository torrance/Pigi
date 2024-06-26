/*
 *
 * This file accompanied with the header file specfun.h is a partial
 * C translation of the Fortran code by Zhang and Jin following
 * original description:
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *       COMPUTATION OF SPECIAL FUNCTIONS
 *
 *          Shanjie Zhang and Jianming Jin
 *
 *       Copyrighted but permission granted to use code in programs.
 *       Buy their book:
 *
 *          Shanjie Zhang, Jianming Jin,
 *          Computation of Special Functions,
 *          Wiley, 1996,
 *          ISBN: 0-471-11963-6,
 *          LC: QA351.C45.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *       Scipy changes:
 *       - Compiled into a single source file and changed REAL To DBLE throughout.
 *       - Changed according to ERRATA.
 *       - Changed GAMMA to GAMMA2 and PSI to PSI_SPEC to avoid potential conflicts.
 *       - Made functions return sf_error codes in ISFER variables instead
 *         of printing warnings. The codes are
 *         - SF_ERROR_OK        = 0: no error
 *         - SF_ERROR_SINGULAR  = 1: singularity encountered
 *         - SF_ERROR_UNDERFLOW = 2: floating point underflow
 *         - SF_ERROR_OVERFLOW  = 3: floating point overflow
 *         - SF_ERROR_SLOW      = 4: too many iterations required
 *         - SF_ERROR_LOSS      = 5: loss of precision
 *         - SF_ERROR_NO_RESULT = 6: no result obtained
 *         - SF_ERROR_DOMAIN    = 7: out of domain
 *         - SF_ERROR_ARG       = 8: invalid input parameter
 *         - SF_ERROR_OTHER     = 9: unclassified error
 *       - Improved initial guesses for roots in JYZO.
 *
 *
 */

/*
 * Copyright (C) 2024 SciPy developers
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * b. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * c. Names of the SciPy Developers may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <math.h>

namespace specialfunc {

template <typename T>
void sdmn(int m, int n, T c, T cv, int kd, T *df) {

    // =====================================================
    // Purpose: Compute the expansion coefficients of the
    //          prolate and oblate spheroidal functions, dk
    // Input :  m  --- Mode parameter
    //          n  --- Mode parameter
    //          c  --- Spheroidal parameter
    //          cv --- Characteristic value
    //          KD --- Function code
    //                 KD=1 for prolate; KD=-1 for oblate
    // Output:  DF(k) --- Expansion coefficients dk;
    //                    DF(1), DF(2), ... correspond to
    //                    d0, d2, ... for even n-m and d1,
    //                    d3, ... for odd n-m
    // =====================================================

    int nm, ip, k, kb;
    T cs, dk0, dk1, dk2, d2k, f, fs, f1, f0, fl, f2, su1,\
           su2, sw, r1, r3, r4, s0;

    nm = 25 + (int)(0.5 * (n - m) + c);

    if (c < 1e-10) {
        for (int i = 1; i <= nm; ++i) {
            df[i-1] = 0.0;
        }
        df[(n - m) / 2] = 1.0;
        return;
    }

    T *a = (T *) calloc(nm + 2, sizeof(double));
    T *d = (T *) calloc(nm + 2, sizeof(double));
    T *g = (T *) calloc(nm + 2, sizeof(double));
    cs = c*c*kd;
    ip = (n - m) % 2;

    for (int i = 1; i <= nm + 2; ++i) {
        k = (ip == 0 ? 2 * (i - 1) : 2 * i - 1);

        dk0 = m + k;
        dk1 = m + k + 1;
        dk2 = 2 * (m + k);
        d2k = 2 * m + k;

        a[i - 1] = (d2k + 2.0) * (d2k + 1.0) / ((dk2 + 3.0) * (dk2 + 5.0)) * cs;
        d[i - 1] = dk0 * dk1 + (2.0 * dk0 * dk1 - 2.0 * m * m - 1.0) / ((dk2 - 1.0) * (dk2 + 3.0)) * cs;
        g[i - 1] = k * (k - 1.0) / ((dk2 - 3.0) * (dk2 - 1.0)) * cs;
    }

    fs = 1.0;
    f1 = 0.0;
    f0 = 1.0e-100;
    kb = 0;
    df[nm] = 0.0;
    fl = 0.0;

    for (int k = nm; k >= 1; k--) {
        f = -((d[k] - cv) * f0 + a[k] * f1) / g[k];

        if (fabs(f) > fabs(df[k])) {
            df[k-1] = f;
            f1 = f0;
            f0 = f;

            if (fabs(f) > 1.0e+100) {
                for (int k1 = k; k1 <= nm; k1++)
                    df[k1 - 1] *= 1.0e-100;
                f1 *= 1.0e-100;
                f0 *= 1.0e-100;
            }
        } else {
            kb = k;
            fl = df[k];
            f1 = 1.0e-100;
            f2 = -((d[0] - cv) / a[0]) * f1;
            df[0] = f1;

            if (kb == 1) {
                fs = f2;
            } else if (kb == 2) {
                df[1] = f2;
                fs = -((d[1] - cv) * f2 + g[1] * f1) / a[1];
            } else {
                df[1] = f2;

                for (int j = 3; j <= kb + 1; j++) {
                    f = -((d[j - 2] - cv) * f2 + g[j - 2] * f1) / a[j - 2];
                    if (j <= kb) {
                        df[j-1] = f;
                    }
                    if (fabs(f) > 1.0e+100) {
                        for (int k1 = 1; k1 <= j; k1++) {
                            df[k1 - 1] *= 1.0e-100;
                        }
                        f *= 1.0e-100;
                        f2 *= 1.0e-100;
                    }
                    f1 = f2;
                    f2 = f;
                }
                fs = f;
            }
            break;
        }
    }

    su1 = 0.0;
    r1 = 1.0;

    for (int j = m + ip + 1; j <= 2 * (m + ip); j++) {
        r1 *= j;
    }
    su1 = df[0] * r1;

    for (int k = 2; k <= kb; k++) {
        r1 = -r1 * (k + m + ip - 1.5) / (k - 1.0);
        su1 += r1 * df[k - 1];
    }

    su2 = 0.0;
    sw = 0.0;

    for (int k = kb + 1; k <= nm; k++) {
        if (k != 1) {
            r1 = -r1 * (k + m + ip - 1.5) / (k - 1.0);
        }
        su2 += r1 * df[k - 1];

        if (fabs(sw - su2) < fabs(su2) * 1.0e-14) { break; }
        sw = su2;
    }
    r3 = 1.0;

    for (int j = 1; j <= (m + n + ip) / 2; j++) {
        r3 *= (j + 0.5 * (n + m + ip));
    }
    r4 = 1.0;

    for (int j = 1; j <= (n - m - ip) / 2; j++) {
        r4 *= -4.0 * j;
    }
    s0 = r3 / (fl * (su1 / fs) + su2) / r4;

    for (int k = 1; k <= kb; ++k) {
        df[k - 1] *= fl / fs * s0;
    }
    for (int k = kb + 1; k <= nm; ++k) {
        df[k - 1] *= s0;
    }
    free(a);free(d);free(g);
    return;
}

template <typename T>
void sckb(int m, int n, T c, T *df, T *ck) {

    // ======================================================
    // Purpose: Compute the expansion coefficients of the
    //          prolate and oblate spheroidal functions
    // Input :  m  --- Mode parameter
    //          n  --- Mode parameter
    //          c  --- Spheroidal parameter
    //          DF(k) --- Expansion coefficients dk
    // Output:  CK(k) --- Expansion coefficients ck;
    //                    CK(1), CK(2), ... correspond to
    //                    c0, c2, ...
    // ======================================================

    int i, ip, i1, i2, k, nm;
    T reg, fac, sw, r, d1, d2, d3, sum, r1;

    if (c <= 1.0e-10) {
        c = 1.0e-10;
    }
    nm = 25 + (int)(0.5 * (n - m) + c);
    ip = (n - m) % 2;
    reg = ((m + nm) > 80 ? 1.0e-200 : 1.0);
    fac = -pow(0.5, m);
    sw = 0.0;

    for (k = 0; k < nm; k++) {
        fac = -fac;
        i1 = 2 * k + ip + 1;
        r = reg;

        for (i = i1; i <= i1 + 2 * m - 1; i++) {
            r *= i;
        }
        i2 = k + m + ip;
        for (i = i2; i <= i2 + k - 1; i++) {
            r *= (i + 0.5);
        }
        sum = r * df[k];
        for (i = k + 1; i <= nm; i++) {
            d1 = 2.0 * i + ip;
            d2 = 2.0 * m + d1;
            d3 = i + m + ip - 0.5;
            r = r * d2 * (d2 - 1.0) * i * (d3 + k) / (d1 * (d1 - 1.0) * (i - k) * d3);
            sum += r * df[i];
            if (fabs(sw - sum) < fabs(sum) * 1.0e-14) { break; }
            sw = sum;
        }
        r1 = reg;
        for (i = 2; i <= m + k; i++) { r1 *= i; }
        ck[k] = fac * sum / r1;
    }
}

template <typename T>
void segv(int m, int n, T c, int kd, T *cv, T *eg) {

    // =========================================================
    // Purpose: Compute the characteristic values of spheroidal
    //          wave functions
    // Input :  m  --- Mode parameter
    //          n  --- Mode parameter
    //          c  --- Spheroidal parameter
    //          KD --- Function code
    //                 KD=1 for Prolate; KD=-1 for Oblate
    // Output:  CV --- Characteristic value for given m, n and c
    //          EG(L) --- Characteristic value for mode m and n'
    //                    ( L = n' - m + 1 )
    // =========================================================


    int i, icm, j, k, k1, l, nm, nm1;
    T cs, dk0, dk1, dk2, d2k, s, t, t1, x1, xa, xb;
    // eg[<=200] is supplied by the caller

    if (c < 1e-10) {
        for (i = 1; i <= (n-m+1); i++) {
            eg[i-1] = (i+m) * (i + m -1);
        }
        *cv = eg[n-m];
        return;
    }

    // TODO: Following array sizes should be decided dynamically
    T *a = (T *) calloc(300, sizeof(T));
    T *b = (T *) calloc(100, sizeof(T));
    T *cv0 = (T *) calloc(100, sizeof(T));
    T *d = (T *) calloc(300, sizeof(T));
    T *e = (T *) calloc(300, sizeof(T));
    T *f = (T *) calloc(300, sizeof(T));
    T *g = (T *) calloc(300, sizeof(T));
    T *h = (T *) calloc(100, sizeof(T));
    icm = (n-m+2)/2;
    nm = 10 + (int)(0.5*(n-m)+c);
    cs = c*c*kd;
    k = 0;
    for (l = 0; l <= 1; l++) {
        for (i = 1; i <= nm; i++) {
            k = (l == 0 ? 2*(i - 1) : 2*i - 1);
            dk0 = m + k;
            dk1 = m + k + 1;
            dk2 = 2*(m + k);
            d2k = 2*m + k;
            a[i-1] = (d2k+2.0)*(d2k+1.0)/((dk2+3.0)*(dk2+5.0))*cs;
            d[i-1] = dk0*dk1+(2.0*dk0*dk1-2.0*m*m-1.0)/((dk2-1.0)*(dk2+3.0))*cs;
            g[i-1] = k*(k-1.0)/((dk2-3.0)*(dk2-1.0))*cs;
        }
        for (k = 2; k <= nm; k++) {
            e[k-1] = sqrt(a[k-2]*g[k-1]);
            f[k-1] = e[k-1]*e[k-1];
        }
        f[0] = 0.0;
        e[0] = 0.0;
        xa = d[nm-1] + fabs(e[nm-1]);
        xb = d[nm-1] - fabs(e[nm-1]);
        nm1 = nm-1;
        for (i = 1; i <= nm1; i++) {
            t = fabs(e[i-1])+fabs(e[i]);
            t1 = d[i-1] + t;
            if (xa < t1) { xa = t1; }
            t1 = d[i-1] - t;
            if (t1 < xb) { xb = t1; }
        }
        for (i = 1; i <= icm; i++) {
            b[i-1] = xa;
            h[i-1] = xb;
        }
        for (k = 1; k <= icm; k++) {
            for (k1 = k; k1 <= icm; k1++) {
                if (b[k1-1] < b[k-1]) {
                    b[k-1] = b[k1-1];
                    break;
                }
            }
            if (k != 1) {
                if(h[k-1] < h[k-2]) { h[k-1] = h[k-2]; }
            }
            while (1) {
                x1 = (b[k-1]+h[k-1])/2.0;
                cv0[k-1] = x1;
                if (fabs((b[k-1] - h[k-1])/x1) < 1e-14) { break; }
                j = 0;
                s = 1.0;
                for (i = 1; i <= nm; i++) {
                    if (s == 0.0) { s += 1e-30; }
                    t = f[i-1]/s;
                    s = d[i-1] - t - x1;
                    if (s < 0.0) { j += 1; }
                }
                if (j < k) {
                    h[k-1] = x1;
                } else {
                    b[k-1] = x1;
                    if (j >= icm) {
                        b[icm - 1] = x1;
                    } else {
                        if (h[j] < x1) { h[j] = x1; }
                        if (x1 < b[j-1]) { b[j-1] = x1; }
                    }
                }
            }
            cv0[k-1] = x1;
            if (l == 0) eg[2*k-2] = cv0[k-1];
            if (l == 1) eg[2*k-1] = cv0[k-1];
        }
    }
    *cv = eg[n-m];
    free(a);free(b);free(cv0);free(d);free(e);free(f);free(g);free(h);
    return;
}

template <typename T>
void aswfa(T x, int m, int n, T c, int kd, T cv, T *s1f, T *s1d) {

    // ===========================================================
    // Purpose: Compute the prolate and oblate spheroidal angular
    //          functions of the first kind and their derivatives
    // Input :  m  --- Mode parameter,  m = 0,1,2,...
    //          n  --- Mode parameter,  n = m,m+1,...
    //          c  --- Spheroidal parameter
    //          x  --- Argument of angular function, |x| < 1.0
    //          KD --- Function code
    //                 KD=1 for prolate;  KD=-1 for oblate
    //          cv --- Characteristic value
    // Output:  S1F --- Angular function of the first kind
    //          S1D --- Derivative of the angular function of
    //                  the first kind
    // Routine called:
    //          SCKB for computing expansion coefficients ck
    // ===========================================================

    int ip, k, nm, nm2;
    T a0, d0, d1, r, su1, su2, x0, x1;
    T *ck = (T *) calloc(200, sizeof(T));
    T *df = (T *) calloc(200, sizeof(T));
    const T eps = 1e-14;
    x0 = x;
    x = fabs(x);
    ip = ((n-m) % 2 == 0 ? 0 : 1);
    nm = 40 + (int)((n-m)/2 + c);
    nm2 = nm/2 - 2;
    sdmn(m, n, c, cv, kd, df);
    sckb(m, n, c, df, ck);
    x1 = 1.0 - x*x;
    if ((m == 0) && (x1 == 0.0)) {
        a0 = 1.0;
    } else {
        a0 = pow(x1, 0.5*m);
    }
    su1 = ck[0];
    for (k = 1; k <= nm2; k++) {
        r = ck[k]*pow(x1, k);
        su1 += r;
        if ((k >= 10) && (fabs(r/su1) < eps)) { break; }
    }
    *s1f = a0*pow(x, ip)*su1;
    if (x == 1.0) {
        if (m == 0) {
            *s1d = ip*ck[0] - 2.0*ck[1];
        } else if (m == 1) {
            *s1d = -1e100;
        } else if (m == 2) {
            *s1d = -2.0*ck[0];
        } else if (m >= 3) {
            *s1d = 0.0;
        }
    } else {
        d0 = ip - m/x1*pow(x, ip+1.0);
        d1 = -2.0*a0*pow(x, ip+1.0);
        su2 = ck[1];
        for (k = 2; k <= nm2; k++) {
            r = k*ck[k]*pow(x1, (k-1.0));
            su2 += r;
            if ((k >= 10) && (fabs(r/su2) < eps)) { break; }
        }
        *s1d = d0*a0*su1 + d1*su2;
    }
    if ((x0 < 0.0) && (ip == 0)) { *s1d = -*s1d; }
    if ((x0 < 0.0) && (ip == 1)) { *s1f = -*s1f; }
    x = x0;
    free(ck); free(df);
    return;
}

}